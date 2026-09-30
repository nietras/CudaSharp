using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using static CudaSharp.nvcuda;
using static CudaSharp.nvrtc;

namespace CudaSharp.TileGym.Test;

[TestClass]
public class TileCppTest
{
    [TestMethod]
    public void TileCppTest_ConfigValidatesCompilerHints()
    {
        var config = CreateConfig(64, numCtas: 4, occupancy: 2);

        Assert.AreEqual("64", config["BLOCK_SIZE"]);
        Assert.AreEqual(4, config.NumCtas);
        Assert.AreEqual(2, config.Occupancy);
        Assert.AreEqual(3, config.OptimizationLevel);
        Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => CreateConfig(64, numCtas: 3));
        Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => CreateConfig(64, occupancy: 33));
        Assert.ThrowsExactly<ArgumentOutOfRangeException>(() =>
            new TileCppConfig([], numWorkerWarps: 5));
    }

    [TestMethod]
    public void TileCppTest_CompileRejectsNullAndDisposedKernel()
    {
        var kernel = new TileCppKernel(new TileCppCompiler(120, installBundledHeaders: false),
            string.Empty, "test.cu", "test_kernel");
        Assert.ThrowsExactly<ArgumentNullException>(() => kernel.Compile(null!));
        kernel.Dispose();
        Assert.ThrowsExactly<ObjectDisposedException>(() => kernel.Compile(new TileCppConfig([])));
    }

    /// <summary>Checks stage diagnostics for compilation, cache hits, and compilation failures.</summary>
    /// <seealso href="https://docs.nvidia.com/cuda/nvrtc/index.html#group__compilation" />
    [TestMethod]
    [DoNotParallelize]
    public void TileCppTest_CUDA13_3TracesCompilationStages()
    {
        nvrtcVersion(out var major, out var minor).Ok();
        if (major < 13 || major == 13 && minor < 3)
        {
            Assert.Inconclusive($"CUDA Tile C++ requires NVRTC 13.3 or later; found {major}.{minor}.");
        }
        var architectures = nvrtcGetSupportedArchs();
        var compiler = new TileCppCompiler(architectures[^1]);
        compiler.PrepareBundledHeaders();
        const string source = """
            #include <cuda_tile.h>
            template<int Variant> __tile_global__ void timing_kernel() {}
            template __tile_global__ void timing_kernel<1>();
            """;
        var config = new TileCppConfig([]);
        using var writer = new StringWriter(CultureInfo.InvariantCulture);
        using var listener = new TextWriterTraceListener(writer);
        Trace.Listeners.Add(listener);
        try
        {
            using var kernel = new TileCppKernel(compiler, source, "timing_kernel.cu", "timing_kernel",
                nameExpression: "&timing_kernel<1>");
            var bytes = kernel.Compile(config);
            Assert.IsGreaterThan(0, bytes);
            var compiledTrace = writer.ToString();
            Assert.Contains("thread=", compiledTrace);
            Assert.Contains("&timing_kernel<1>", compiledTrace);
            Assert.Contains("_compilations.GetOrAdd:", compiledTrace);
            Assert.Contains("Lazy.Value (inclusive of compiler stages):", compiledTrace);
            Assert.Contains("_headersInstalled.Value:", compiledTrace);
            Assert.Contains(".nvrtcCreateProgram:", compiledTrace);
            Assert.Contains(".nvrtcAddNameExpression:", compiledTrace);
            Assert.Contains(".nvrtcCompileProgram[program=0x", compiledTrace);
            Assert.Contains(".nvrtcGetTileIR:", compiledTrace);
            Assert.Contains(".nvrtcGetLoweredNameString:", compiledTrace);
            Assert.Contains(".nvrtcDestroyProgram:", compiledTrace);
            var reports = Regex.Matches(compiledTrace, "Compile timing:");
            Assert.HasCount(1, reports);

            var cachedBytes = kernel.Compile(config);
            Assert.AreEqual(bytes, cachedBytes);
            var cachedTrace = writer.ToString()[compiledTrace.Length..];
            Assert.Contains("Lazy.Value (inclusive of compiler stages):", cachedTrace);
            Assert.DoesNotContain(".nvrtcCompileProgram[", cachedTrace);

            var output = compiler.CompileToTileIr(source, "timing_output.cu", config);
            Assert.IsNotEmpty(output);
            var outputTrace = writer.ToString();
            Assert.Contains("CompileOutput[timing_output.cu].nvrtcGetTileIR:", outputTrace);

            Assert.ThrowsExactly<CudaException<nvrtcResult>>(() =>
                compiler.CompileToTileIr("invalid CUDA source", "timing_failure.cu", config));
            var failedTrace = writer.ToString()[outputTrace.Length..];
            Assert.Contains(".nvrtcGetProgramLogString:", failedTrace);
            Assert.Contains(".nvrtcDestroyProgram:", failedTrace);

            var parallelStart = writer.GetStringBuilder().Length;
            Parallel.For(0, 4, i =>
            {
                using var parallelKernel = new TileCppKernel(compiler, source, $"timing_parallel_{i}.cu",
                    "timing_kernel", nameExpression: "&timing_kernel<1>");
                var parallelBytes = parallelKernel.Compile(config);
                Assert.IsGreaterThan(0, parallelBytes);
            });
            var parallelTrace = writer.ToString()[parallelStart..];
            var parallelReports = Regex.Matches(parallelTrace,
                @"Compile timing:.*?(?=\r?\nCompile timing:|\z)", RegexOptions.Singleline);
            Assert.HasCount(4, parallelReports);
            foreach (Match report in parallelReports)
            {
                var sourceNames = Regex.Matches(report.Value, @"timing_parallel_\d+\.cu")
                    .Select(static match => match.Value).Distinct(StringComparer.Ordinal).ToArray();
                Assert.HasCount(1, sourceNames, "Parallel compilation timings must not mix specialization records.");
                Assert.Contains(".nvrtcCompileProgram[program=0x", report.Value);
            }
        }
        finally
        {
            Trace.Listeners.Remove(listener);
        }
    }

    [TestMethod]
    public void TileCppTest_TileGymTesterCoversEveryVendoredKernel()
    {
        var root = Path.Combine(AppContext.BaseDirectory, "src-tilecpp", "tilegym");
        var sourceRoot = GetTileGymSourceRoot();
        var kernels = Directory.EnumerateFiles(root, "*.cuh", SearchOption.AllDirectories)
            .SelectMany(path =>
            {
                var source = File.ReadAllText(path);
                var matches = Regex.Matches(source, @"__tile_global__\s+void\s+(\w+)");
                return matches.Select(match => match.Groups[1].Value);
            })
            .Distinct(StringComparer.Ordinal).OrderBy(name => name).ToArray();
        var scenarioPaths = Directory.EnumerateFiles(sourceRoot, "TileGym*Scenarios.cs");
        var scenarioSources = scenarioPaths.Select(File.ReadAllText);
        var scenarios = string.Join('\n', scenarioSources);
        var missing = kernels.Where(kernel => !scenarios.Contains($"\"{kernel}\"", StringComparison.Ordinal)).ToArray();

        Assert.HasCount(60, kernels);
        var missingKernels = string.Join(", ", missing);
        Assert.IsEmpty(missing, $"Missing Tester scenarios: {missingKernels}");
    }

    [TestMethod]
    public void TileCppTest_CUDA13_3CompilesRepresentativeTileGymFamiliesToTileIr()
    {
        try
        {
            nvrtcVersion(out var major, out var minor).Ok();
            if (major < 13 || major == 13 && minor < 3) Assert.Inconclusive($"CUDA Tile C++ requires NVRTC 13.3 or later; found {major}.{minor}.");
            var root = Path.Combine(AppContext.BaseDirectory, "src-tilecpp", "tilegym");
            var cases = new[]
            {
                ("softmax.cuh","softmax_kernel","float, 64, 0","float*, const float*, int, int, int, int, int"),
                ("matmul.cuh","matmul_kernel","float, 64, 64, 64, 64, 64, 32, 8, 2, false, false, 1, 2","const float*, const float*, float*"),
                ("bmm.cuh","bmm_kernel","float, 64, 64, 32, 8, false, false","const float*, const float*, float*, int, int, int, int"),
                ("dropout.cuh","seeded_dropout_kernel","float, 256, 4096, 0.25f, 2654435761u","const float*, float*"),
                ("rope.cuh","rope_kernel","float, float, float, 1, 2, 1, 2, 1, 32, 32, 64, 1, 4, 512, 256, 64, 256, 256, 64","float*, float*, const float*, const float*"),
                ("mla_decoding.cuh","naive_absorb_mla_transpose","float, 64, 1, 64, 16, 0, 64, true","float*, float*, float*, float*, float*, float*, float, long long, int, long long, int, long long, int, long long, int, long long, int, int, int"),
                ("moe_align_block.cuh","moe_align_block_size_stage2","int, 4, 4, 4","int*"),
                ("moe_align_block.cuh","moe_align_block_size_stage3","int, 4, 4, 4","int*, int*, const int*, int*"),
            };
            var supportedArchitectures = nvrtcGetSupportedArchs();
            var highestArchitecture = supportedArchitectures[^1];
            var compiler = new TileCppCompiler(highestArchitecture);
            foreach (var item in cases)
            {
                var headerPath = Path.Combine(root, item.Item1);
                var headerSource = File.ReadAllText(headerPath);
                var header = new TileCppHeader(item.Item1, headerSource);
                var source = $"using int32_t = int; using uint32_t = unsigned int;\nnamespace std {{ template<class A,class B> struct is_same {{ static constexpr bool value=false; }}; template<class A> struct is_same<A,A> {{ static constexpr bool value=true; }}; template<class A,class B> inline constexpr bool is_same_v=is_same<A,B>::value; template<bool B,class T,class F> struct conditional {{ using type=T; }}; template<class T,class F> struct conditional<false,T,F> {{ using type=F; }}; template<bool B,class T,class F> using conditional_t=typename conditional<B,T,F>::type; }}\n#include <cmath>\n#include \"{item.Item1}\"\ntemplate __tile_global__ void {item.Item2}<{item.Item3}>({item.Item4});";
                var compilation = compiler.CompileKernel(source, $"{item.Item2}.cu", $"&{item.Item2}<{item.Item3}>", new TileCppConfig([]), [header, new TileCppHeader("type_traits", string.Empty), new TileCppHeader("cmath", "#ifndef INFINITY\n#define INFINITY __builtin_bit_cast(float, 0x7f800000u)\n#endif\n")]);
                Assert.IsNotEmpty(compilation.TileIr, item.Item2);
            }
        }
        catch (AssertInconclusiveException) { throw; }
        catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException) { Assert.Inconclusive($"CUDA Tile C++ NVRTC components are unavailable: {ex.Message}"); }
    }

    [TestMethod]
    public void TileCppTest_SearchSpaceRequiresMatchingParameterNames()
    {
        var first = CreateConfig(64);
        var second = new TileCppConfig([new("OTHER", "128")]);

        Assert.ThrowsExactly<ArgumentException>(() => new TileCppSearchSpace([first, second]));
    }

    [TestMethod]
    public void TileCppTest_AutotunerSelectsAndCachesFastestConfiguration()
    {
        var slow = CreateConfig(64);
        var fast = CreateConfig(128);
        var timer = new FakeTimer(new Dictionary<TileCppConfig, float>
        {
            [slow] = 2,
            [fast] = 1,
        });
        var tuner = new TileCppAutotuner(new TileCppSearchSpace([slow, fast]), timer);
        TileCppConfig? current = null;
        var launches = 0;

        void Launch(TileCppConfig config)
        {
            current = config;
            launches++;
        }

        timer.GetCurrent = () => current ?? throw new InvalidOperationException();
        var first = tuner.Tune(default, ("relu", 4096), Launch,
            static (_, config) =>
            {
                var blockSize = uint.Parse(config["BLOCK_SIZE"]);
                return new TileCppGrid(4096 / blockSize);
            }, seed: 42);
        var measuresAfterTuning = timer.MeasureCount;
        var second = tuner.Tune(default, ("relu", 4096), Launch,
            static (_, config) =>
            {
                var blockSize = uint.Parse(config["BLOCK_SIZE"]);
                return new TileCppGrid(4096 / blockSize);
            }, seed: 42);

        Assert.AreSame(fast, first.Config);
        Assert.AreEqual(new TileCppGrid(32), first.Grid);
        Assert.AreSame(first, second);
        Assert.AreEqual(2, measuresAfterTuning);
        Assert.AreEqual(measuresAfterTuning, timer.MeasureCount);
        Assert.AreEqual(1, timer.SynchronizeCount);
        Assert.IsGreaterThanOrEqualTo(6, launches);
    }

    [TestMethod]
    public void TileCppTest_AutotunerCompilesCandidatesBeforeTimingWithoutLaunching()
    {
        var slow = CreateConfig(64);
        var fast = CreateConfig(128);
        var broken = CreateConfig(256);
        var timer = new FakeTimer(new Dictionary<TileCppConfig, float>
        {
            [slow] = 2,
            [fast] = 1,
        });
        var tuner = new TileCppAutotuner(new TileCppSearchSpace([slow, fast, broken]), timer);
        var compiled = new System.Collections.Concurrent.ConcurrentBag<TileCppConfig>();
        TileCppConfig? current = null;
        var launches = 0;

        void Compile(TileCppConfig config)
        {
            compiled.Add(config);
            if (ReferenceEquals(config, broken))
            {
                throw new InvalidOperationException("Compilation failed.");
            }
        }

        void Launch(TileCppConfig config)
        {
            current = config;
            launches++;
        }

        timer.GetCurrent = () => current ?? throw new InvalidOperationException();
        var result = tuner.Tune(default, ("relu", 4096), Launch,
            static (_, config) =>
            {
                var blockSize = uint.Parse(config["BLOCK_SIZE"]);
                return new TileCppGrid(4096 / blockSize);
            }, seed: 42, compile: Compile);

        Assert.AreSame(fast, result.Config);
        Assert.HasCount(3, compiled);
        Assert.AreEqual(2, timer.MeasureCount);
        Assert.AreEqual(0, timer.SynchronizeCount);
        Assert.AreEqual(3, launches);
    }

    [TestMethod]
    public void TileCppTest_CUDA13_3CompilesTileGymReluToTileIr()
    {
        try
        {
            nvrtcVersion(out var major, out var minor).Ok();
            if (major < 13 || major == 13 && minor < 3)
                Assert.Inconclusive($"CUDA Tile C++ requires NVRTC 13.3 or later; found {major}.{minor}.");

            var headerPath = Path.Combine(AppContext.BaseDirectory,
                "src-tilecpp", "tilegym", "activation", "relu.cuh");
            var headerSource = File.ReadAllText(headerPath);
            var header = new TileCppHeader("relu.cuh", headerSource);
            const string source = """
                using int32_t = int;
                #include "relu.cuh"
                template __tile_global__ void relu_activation_fwd_kernel<float, 64, 0>(
                    const float*, float*, int, float, float, float, bool);
                """;
            var supportedArchitectures = nvrtcGetSupportedArchs();
            var highestArchitecture = supportedArchitectures[^1];
            var compiler = new TileCppCompiler(highestArchitecture);

            var compilation = compiler.CompileKernel(source, "relu.cu",
                "&relu_activation_fwd_kernel<float, 64, 0>", new TileCppConfig([]), [header]);

            Assert.IsNotEmpty(compilation.TileIr);
            var entryPointIsEmpty = string.IsNullOrWhiteSpace(compilation.EntryPoint);
            Assert.IsFalse(entryPointIsEmpty);
        }
        catch (AssertInconclusiveException)
        {
            throw;
        }
        catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException)
        {
            Assert.Inconclusive($"CUDA Tile C++ NVRTC components are unavailable: {ex.Message}");
        }
    }

    [TestMethod]
    public void TileCppTest_CUDA13_3CompilesSoftmaxWithInfinity()
    {
        nvrtcVersion(out var major, out var minor).Ok();
        if (major < 13 || major == 13 && minor < 3)
            Assert.Inconclusive($"CUDA Tile C++ requires NVRTC 13.3 or later; found {major}.{minor}.");

        var path = Path.Combine(AppContext.BaseDirectory, "src-tilecpp", "tilegym", "softmax.cuh");
        var headerSource = File.ReadAllText(path);
        var header = new TileCppHeader("softmax.cuh", headerSource);
        const string source = """
            #include <cmath>
            #include "softmax.cuh"
            template __tile_global__ void softmax_kernel<float, 256, 0>(
                float*, const float*, int, int, int, int, int);
            """;
        var supportedArchitectures = nvrtcGetSupportedArchs();
        var highestArchitecture = supportedArchitectures[^1];
        var compiler = new TileCppCompiler(highestArchitecture);
        var compilation = compiler.CompileKernel(source, "softmax_infinity.cu",
            "&softmax_kernel<float, 256, 0>", new TileCppConfig([]),
            [header, new TileCppHeader("cmath", "#ifndef INFINITY\n#define INFINITY __builtin_bit_cast(float, 0x7f800000u)\n#endif\n")]);
        Assert.IsNotEmpty(compilation.TileIr);
    }

    [TestMethod]
    public unsafe void TileCppTest_CUDA13_3LaunchesTileGymReluWithoutUsedAttribute()
    {
        nvrtcVersion(out var major, out var minor).Ok();
        if (major < 13 || major == 13 && minor < 3)
            Assert.Inconclusive($"CUDA Tile C++ requires NVRTC 13.3 or later; found {major}.{minor}.");

        CuInit.EnsureInit();
        cuDeviceGet(out var device, 0).Ok();
        cuCtxGetCurrent(out var previousContext).Ok();
        cuDevicePrimaryCtxRetain(out var context, device).Ok();
        try
        {
            cuCtxSetCurrent(context).Ok();
            var headerPath = Path.Combine(AppContext.BaseDirectory, "src-tilecpp", "tilegym", "activation", "relu.cuh");
            var headerSource = File.ReadAllText(headerPath);
            var header = new TileCppHeader("relu.cuh", headerSource);
            var sourceContainsUsedAttribute = header.Source.Contains("__attribute__((used))", StringComparison.Ordinal);
            Assert.IsFalse(sourceContainsUsedAttribute);
            const string source = """
                using int32_t = int;
                #include "relu.cuh"
                template __tile_global__ void relu_activation_fwd_kernel<float, 64, 0>(
                    const float*, float*, int, float, float, float, bool);
                """;
            var deviceArchitecture = device.GetArchitecture();
            var compiler = new TileCppCompiler(deviceArchitecture);
            compiler.PrepareBundledHeaders();
            var config = new TileCppConfig([]);
            var compilation = compiler.CompileKernel(source, "relu_without_used.cu",
                "&relu_activation_fwd_kernel<float, 64, 0>", config, [header]);
            using var kernel = new TileCppKernel(compiler, source, "relu_without_used.cu",
                "relu_activation_fwd_kernel", [header], nameExpression: "&relu_activation_fwd_kernel<float, 64, 0>");
            Assert.ThrowsExactly<InvalidOperationException>(() => kernel.LoadFunction(config));
            var otherConfig = new TileCppConfig([new("VARIANT", "1")]);
            Parallel.Invoke(() => kernel.Compile(config), () => kernel.Compile(otherConfig),
                () => kernel.Compile(config));
            var loadedFunction = kernel.LoadFunction(config);
            var otherLoadedFunction = kernel.LoadFunction(otherConfig);
            Assert.AreNotEqual(default, otherLoadedFunction);
            var cachedFunction = kernel.GetFunction(config);
            Assert.AreEqual(loadedFunction, cachedFunction);
            Assert.ThrowsExactly<InvalidOperationException>(() => kernel.LoadFunction(new TileCppConfig([new("VARIANT", "2")])));

            cuModuleLoadData(out var module, compilation.TileIr).Ok();
            try
            {
                // Do not use TileCppKernel's substring-name fallback: verify the exact lowered entry exists.
                cuModuleGetFunction(out var function, module, compilation.EntryPoint).Ok();
                const int count = 64;
                var input = new float[count];
                for (var i = 0; i < count; i++)
                    input[i] = i - 32;
                cuMemAlloc_v2(out var x, count * sizeof(float)).Ok();
                try
                {
                    cuMemAlloc_v2(out var y, count * sizeof(float)).Ok();
                    try
                    {
                        fixed (float* pointer = input)
                            cuMemcpyHtoD_v2(x, (IntPtr)pointer, count * sizeof(float)).Ok();
                        var px = x.Value;
                        var py = y.Value;
                        var length = count;
                        float alpha = 0, lower = 0, upper = 0;
                        byte training = 0;
                        var args = stackalloc void*[] { &px, &py, &length, &alpha, &lower, &upper, &training };
                        cuLaunchKernel(loadedFunction, 1, 1, 1, 1, 1, 1, 0, default, args, null).Ok();
                        var result = new float[count];
                        fixed (float* pointer = result)
                            cuMemcpyDtoH_v2((IntPtr)pointer, y, count * sizeof(float)).Ok();
                        for (var i = 0; i < count; i++)
                        {
                            var expectedValue = Math.Max(input[i], 0);
                            Assert.AreEqual(expectedValue, result[i], $"ReLU element {i}");
                        }
                    }
                    finally
                    {
                        cuMemFree_v2(y).Ok();
                    }
                }
                finally
                {
                    cuMemFree_v2(x).Ok();
                }
            }
            finally
            {
                cuModuleUnload(module).Ok();
            }
        }
        finally
        {
            cuCtxSetCurrent(previousContext).Ok();
            cuDevicePrimaryCtxRelease(device).Ok();
        }
    }

    static string GetTileGymSourceRoot([CallerFilePath] string filePath = "")
    {
        var sourceDirectory = Path.GetDirectoryName(filePath)!;
        var tileGymDirectory = Path.Combine(sourceDirectory, "..", "CudaSharp.TileGym");
        return Path.GetFullPath(tileGymDirectory);
    }

    static TileCppConfig CreateConfig(int blockSize, int? numCtas = null, int? occupancy = null)
    {
        var blockSizeText = blockSize.ToString();
        KeyValuePair<string, string> blockSizeParameter = new("BLOCK_SIZE", blockSizeText);
        return new TileCppConfig([blockSizeParameter], numCtas, occupancy);
    }

    sealed class FakeTimer(IReadOnlyDictionary<TileCppConfig, float> timings) : ITileCppTimer
    {
        public Func<TileCppConfig>? GetCurrent { get; set; }
        public int MeasureCount { get; private set; }
        public int SynchronizeCount { get; private set; }

        public float Measure(Action launch, CUstream stream, TileCppTimingOptions options)
        {
            launch();
            MeasureCount++;
            return timings[GetCurrent!()];
        }

        public void Synchronize(CUstream stream) => SynchronizeCount++;
    }
}
