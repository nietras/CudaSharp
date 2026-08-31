using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using CudaSharp.Tile;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using static CudaSharp.nvcuda;
using static CudaSharp.nvrtc;

namespace CudaSharp.Test;

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
    public void TileCppTest_TileGymTesterCoversEveryVendoredKernel()
    {
        var root = Path.Combine(AppContext.BaseDirectory, "src-tilecpp", "tilegym");
        var sourceRoot = Path.GetFullPath(Path.Combine(root, "..", "..", "..", "..", "..", "..", "src", "CudaSharp.Tester"));
        var kernels = Directory.EnumerateFiles(root, "*.cuh", SearchOption.AllDirectories)
            .SelectMany(path => Regex.Matches(File.ReadAllText(path), @"__tile_global__\s+void\s+(\w+)").Select(match => match.Groups[1].Value))
            .Distinct(StringComparer.Ordinal).OrderBy(name => name).ToArray();
        var scenarios = string.Join('\n', Directory.EnumerateFiles(sourceRoot, "TileGym*Scenarios.cs").Select(File.ReadAllText));
        var missing = kernels.Where(kernel => !scenarios.Contains($"\"{kernel}\"", StringComparison.Ordinal)).ToArray();

        Assert.HasCount(52, kernels);
        Assert.IsEmpty(missing, $"Missing Tester scenarios: {string.Join(", ", missing)}");
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
                ("softmax.cuh","softmax_kernel","float, 64","float*, const float*, int, int, int, int, int"),
                ("matmul.cuh","matmul_kernel","float, 64, 64, 64, 64, 64, 32, 8, 2, false, false, 1, 2","const float*, const float*, float*"),
                ("bmm.cuh","bmm_kernel","float, 64, 64, 32, 8, false, false","const float*, const float*, float*, int, int, int, int"),
            };
            var compiler = new TileCppCompiler(nvrtcGetSupportedArchs()[^1]);
            foreach (var item in cases)
            {
                var header = new TileCppHeader(item.Item1, File.ReadAllText(Path.Combine(root, item.Item1)).Replace("INFINITY", "3.402823466e+38F", StringComparison.Ordinal));
                var source = $"using int32_t = int; using uint32_t = unsigned int;\nnamespace std {{ template<class A,class B> struct is_same {{ static constexpr bool value=false; }}; template<class A> struct is_same<A,A> {{ static constexpr bool value=true; }}; template<class A,class B> inline constexpr bool is_same_v=is_same<A,B>::value; template<bool B,class T,class F> struct conditional {{ using type=T; }}; template<class T,class F> struct conditional<false,T,F> {{ using type=F; }}; template<bool B,class T,class F> using conditional_t=typename conditional<B,T,F>::type; }}\n#include \"{item.Item1}\"\ntemplate __tile_global__ void {item.Item2}<{item.Item3}>({item.Item4});";
                var compilation = compiler.CompileKernel(source, $"{item.Item2}.cu", $"&{item.Item2}<{item.Item3}>", new TileCppConfig([]), [header, new TileCppHeader("type_traits", string.Empty)]);
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
            static (_, config) => new TileCppGrid(4096 / uint.Parse(config["BLOCK_SIZE"])), seed: 42);
        var measuresAfterTuning = timer.MeasureCount;
        var second = tuner.Tune(default, ("relu", 4096), Launch,
            static (_, config) => new TileCppGrid(4096 / uint.Parse(config["BLOCK_SIZE"])), seed: 42);

        Assert.AreSame(fast, first.Config);
        Assert.AreEqual(new TileCppGrid(32), first.Grid);
        Assert.AreSame(first, second);
        Assert.AreEqual(2, measuresAfterTuning);
        Assert.AreEqual(measuresAfterTuning, timer.MeasureCount);
        Assert.AreEqual(1, timer.SynchronizeCount);
        Assert.IsGreaterThanOrEqualTo(6, launches);
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
            var header = new TileCppHeader("relu.cuh", File.ReadAllText(headerPath));
            const string source = """
                using int32_t = int;
                #include "relu.cuh"
                template __tile_global__ void relu_activation_fwd_kernel<float, 64, 0>(
                    const float*, float*, int, float, float, float, bool);
                """;
            var compiler = new TileCppCompiler(nvrtcGetSupportedArchs()[^1]);

            var compilation = compiler.CompileKernel(source, "relu.cu",
                "&relu_activation_fwd_kernel<float, 64, 0>", new TileCppConfig([]), [header]);

            Assert.IsNotEmpty(compilation.TileIr);
            Assert.IsFalse(string.IsNullOrWhiteSpace(compilation.EntryPoint));
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

    static TileCppConfig CreateConfig(int blockSize, int? numCtas = null, int? occupancy = null) =>
        new([new("BLOCK_SIZE", blockSize.ToString())], numCtas, occupancy);

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
