using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using CudaSharp.Tester;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using static CudaSharp.nvcuda;
using static CudaSharp.nvrtc;

namespace CudaSharp.TileGym.Test;

/// <summary>Tests hierarchical preparation, shared batch accounting, and specialization failure isolation.</summary>
/// <seealso href="https://docs.nvidia.com/cuda/nvrtc/index.html#group__compilation" />
[TestClass]
public class TileGymKernelCacheTest
{
    const string Header = "activation/relu.cuh";
    const string Forward = "relu_activation_fwd_kernel";
    const string ForwardSignature = "const float*, float*, int, float, float, float, bool";
    const string Backward = "relu_activation_bwd_kernel";
    const string BackwardSignature = "const float*, const float*, float*, int, float, float, float, bool";
    static readonly TileGymKernelDefinition ForwardDefinition = new(Header, Forward, ForwardSignature);

    /// <summary>Merges duplicate kernel requests without losing the header, kernel, specialization hierarchy.</summary>
    /// <seealso href="https://docs.nvidia.com/cuda/nvrtc/index.html#group__compilation" />
    [TestMethod]
    public void TileGymKernelCacheTest_PlanGroupsAndDeduplicatesRequests()
    {
        string[] arguments = ["float, 64, 0", "float, 128, 0"];
        var first = new TileGymKernelRequest(ForwardDefinition, arguments);
        arguments[0] = "modified";
        var second = new TileGymKernelRequest(ForwardDefinition, ["float, 128, 0", "float, 256, 0"]);
        var backward = new TileGymKernelRequest(new(Header, Backward, BackwardSignature), ["float, 64, 0"]);
        var other = new TileGymKernelRequest(new("softmax.cuh", "softmax_kernel", "float*"), ["float, 64, 0"]);
        var plan = TileGymCompilationPlan.Create([first, second, backward, other]);

        Assert.HasCount(2, plan.Units);
        var unit = plan.Units[0];
        Assert.AreEqual(Header, unit.Header);
        Assert.HasCount(2, unit.Kernels);
        Assert.HasCount(3, unit.Kernels[0].Specializations);
        Assert.HasCount(4, unit.Specializations);
        Assert.AreEqual("float, 64, 0", unit.Kernels[0].Specializations[0].TemplateArguments);
        Assert.AreEqual(Backward, unit.Kernels[1].Definition.Name);
        var filtered = plan.Where(static spec => spec.TemplateArguments == "float, 256, 0");
        Assert.HasCount(1, filtered.Units);
        Assert.HasCount(1, filtered.Units[0].Kernels);
        Assert.HasCount(1, filtered.Units[0].Specializations);
        Assert.ThrowsExactly<ArgumentException>(() => new TileGymKernelRequest(ForwardDefinition, []));
    }

    /// <summary>Handles empty plans, argument validation, and disposal without loading any CUDA libraries.</summary>
    /// <seealso href="https://docs.nvidia.com/cuda/nvrtc/index.html#group__compilation" />
    [TestMethod]
    public void TileGymKernelCacheTest_EmptyPlanRequiresNoCuda()
    {
        var compiler = new TileCppCompiler(120, installBundledHeaders: false);
        var plan = TileGymCompilationPlan.Create([]);
        using var cache = new TileGymKernelCache(compiler, default);
        using var progress = new StringWriter(CultureInfo.InvariantCulture);
        cache.ProgressOutput = progress;
        var summary = cache.Prepare(plan, 1);
        var progressText = progress.ToString();
        Assert.Contains("Preparing 0 uncached specializations", progressText);
        Assert.DoesNotContain("Preparing NVRTC bundled headers", progressText);
        Assert.HasCount(0, summary.Results);
        Assert.HasCount(0, summary.Batches!);
        Assert.AreEqual(0, summary.CompileMilliseconds);
        Assert.AreEqual(0, summary.LoadMilliseconds);
        var missing = new TileGymKernelSpec(Header, Forward, "float, 64, 0", ForwardSignature);
        var error = Assert.ThrowsExactly<InvalidOperationException>(() => cache.Get(missing));
        Assert.Contains("was not prepared", error.Message);
        Assert.Contains("Call Prepare", error.Message);
        Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => cache.Prepare(plan, 0));
        Assert.ThrowsExactly<ArgumentNullException>(() => cache.Prepare((TileGymCompilationPlan)null!, 1));
        cache.Dispose();
        Assert.ThrowsExactly<ObjectDisposedException>(() => cache.Prepare(plan, 1));
    }

    /// <summary>Counts batch and retry work once and writes consistent CSV and Markdown columns.</summary>
    /// <seealso href="https://docs.nvidia.com/cuda/nvrtc/index.html#group__compilation" />
    [TestMethod]
    public void TileGymKernelCacheTest_ReportsSharedCostsAndCacheReuse()
    {
        var first = new TileGymKernelSpec(Header, Forward, "float, 64, 0", ForwardSignature);
        var second = first with { TemplateArguments = "float, 128, 0" };
        var start = Stopwatch.GetTimestamp();
        var shared = new TileGymBatchTiming(1, Header, [first, second], 120, 60,
            CompileStartTimestamp: start, LoadStartTimestamp: start, TileIrBytes: 200);
        var retry = new TileGymBatchTiming(2, Header, [first, second], 40,
            Error: new InvalidOperationException("bad, \"quoted\" | value\nsecond line"), CompileStartTimestamp: start);
        TileGymCompiledKernel[] results =
        [
            new(first, new CUfunction(1), 60, 30, CompileStartTimestamp: start,
                LoadStartTimestamp: start, TileIrBytes: 200) { Batch = shared },
            new(second, new CUfunction(2), 60, 30, CompileStartTimestamp: start,
                LoadStartTimestamp: start, TileIrBytes: 200) { Batch = shared }
        ];
        var summary = new TileGymPrecompileSummary(results, start, 160, 60, 200, 300)
        {
            Batches = [retry, shared],
        };
        Assert.AreEqual(160, summary.CompileMilliseconds);
        Assert.AreEqual(60, summary.LoadMilliseconds);
        using var writer = new StringWriter(CultureInfo.InvariantCulture);
        summary.WriteTo(writer);
        var text = writer.ToString();
        Assert.Contains("Full batch timings", text);
        Assert.Contains("Compile/share", text);
        Assert.DoesNotContain("parallelism", text);
        var legacy = summary with { Batches = null };
        using var legacyWriter = new StringWriter(CultureInfo.InvariantCulture);
        legacy.WriteTo(legacyWriter);
        var legacyText = legacyWriter.ToString();
        Assert.DoesNotContain("CACHED", legacyText);
        Assert.AreEqual(120, legacy.CompileMilliseconds);
        var temporaryRoot = Path.GetTempPath();
        var directoryName = "CudaSharp-batch-report-" + Guid.NewGuid().ToString("N");
        var directory = Path.Combine(temporaryRoot, directoryName);
        try
        {
            summary.Write(directory);
            var specializationPath = Path.Combine(directory, "tilegym-precompile.csv");
            var batchPath = Path.Combine(directory, "tilegym-precompile-batches.csv");
            var markdownPath = Path.Combine(directory, "tilegym-precompile.md");
            var specializationRows = File.ReadAllLines(specializationPath);
            var batchRows = File.ReadAllLines(batchPath);
            Assert.HasCount(3, specializationRows);
            Assert.HasCount(3, batchRows);
            foreach (var row in specializationRows.Skip(1))
            {
                var fields = Regex.Matches(row, "\"(?:[^\"]|\"\")*\"");
                Assert.HasCount(13, fields);
            }
            foreach (var row in batchRows.Skip(1))
            {
                var fields = Regex.Matches(row, "\"(?:[^\"]|\"\")*\"");
                Assert.HasCount(12, fields);
            }
            var markdownRows = File.ReadAllLines(markdownPath);
            var tables = markdownRows.Where(static line => line.StartsWith('|')).ToArray();
            Assert.HasCount(8, tables);
            foreach (var row in tables.Take(4))
            {
                var separators = Regex.Matches(row, @"(?<!\\)\|");
                Assert.HasCount(14, separators);
            }
            foreach (var row in tables.Skip(4))
            {
                var separators = Regex.Matches(row, @"(?<!\\)\|");
                Assert.HasCount(13, separators);
            }

            var cached = summary with { Batches = [], CompileWallMilliseconds = 0, LoadWallMilliseconds = 0 };
            Assert.AreEqual(0, cached.CompileMilliseconds);
            Assert.AreEqual(0, cached.LoadMilliseconds);
            cached.Write(directory);
            var cachedRows = File.ReadAllLines(specializationPath);
            foreach (var row in cachedRows.Skip(1))
            {
                var fields = Regex.Matches(row, "\"(?:[^\"]|\"\")*\"");
                Assert.AreEqual("\"0\"", fields[5].Value);
                Assert.AreEqual("\"0\"", fields[7].Value);
                Assert.AreEqual("\"True\"", fields[12].Value);
            }
        }
        finally
        {
            if (Directory.Exists(directory))
            {
                Directory.Delete(directory, recursive: true);
            }
        }
    }

    /// <summary>Compiles different kernels from one header once, reuses cached functions, and launches each variant.</summary>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html" />
    [TestMethod]
    [DoNotParallelize]
    public void TileGymKernelCacheTest_CUDA13_3BatchesKernelsAndLaunchesSpecializations()
    {
        WithContext((compiler, context) =>
        {
            using var cache = new TileGymKernelCache(compiler, context);
            var forward = new TileGymKernelRequest(ForwardDefinition, ["float, 64, 0", "float, 128, 0"]);
            var backward = new TileGymKernelRequest(new(Header, Backward, BackwardSignature), ["float, 64, 0"]);
            var plan = TileGymCompilationPlan.Create([forward, backward]);
            using var writer = new StringWriter(CultureInfo.InvariantCulture);
            using var progress = new StringWriter(CultureInfo.InvariantCulture);
            cache.ProgressOutput = progress;
            using var listener = new TextWriterTraceListener(writer);
            Trace.Listeners.Add(listener);
            try
            {
                cuCtxSetCurrent(default).Ok();
                var summary = cache.Prepare(plan, 2);
                cuCtxGetCurrent(out var currentContext).Ok();
                Assert.AreEqual(default, currentContext);
                Assert.AreEqual(0, summary.Failed);
                Assert.HasCount(3, summary.Results);
                Assert.HasCount(1, summary.Batches!);
                var progressText = progress.ToString();
                Assert.Contains("Preparing 3 uncached specializations in 1 header batches.", progressText);
                Assert.Contains($"Compiling {Header}: 3 specializations...", progressText);
                Assert.Contains($"Loading batch 1 ({Header})", progressText);
                Assert.Contains("compile ", progressText);
                Assert.Contains("load ", progressText);
                var batch = summary.Batches![0];
                Assert.HasCount(3, batch.Specializations);
                foreach (var result in summary.Results)
                {
                    Assert.AreSame(batch, result.Batch);
                    Assert.AreSame(result, cache.Get(result.Spec));
                    Assert.AreNotEqual(default, result.Function);
                }
                var compileShares = summary.Results.Sum(static result => result.CompileMilliseconds);
                var loadShares = summary.Results.Sum(static result => result.LoadMilliseconds);
                Assert.AreEqual(summary.CompileMilliseconds, compileShares, 0.000001);
                Assert.AreEqual(summary.LoadMilliseconds, loadShares, 0.000001);
                var firstTrace = writer.ToString();
                var compileCalls = Regex.Matches(firstTrace, @"\.nvrtcCompileProgram\[program=0x");
                Assert.HasCount(1, compileCalls);
                var cached = cache.Prepare(plan, 2);
                Assert.HasCount(0, cached.Batches!);
                Assert.AreEqual(0, cached.CompileMilliseconds);
                Assert.AreEqual(0, cached.LoadMilliseconds);
                var cachedProgress = progress.ToString()[progressText.Length..];
                Assert.Contains("Preparing 0 uncached specializations", cachedProgress);
                Assert.DoesNotContain("Loading batch", cachedProgress);
                var cachedTrace = writer.ToString()[firstTrace.Length..];
                Assert.DoesNotContain(".nvrtcCompileProgram[", cachedTrace);

                var incremental = new TileGymKernelRequest(ForwardDefinition, ["float, 64, 0", "float, 256, 0"]);
                var expandedPlan = TileGymCompilationPlan.Create([incremental]);
                var expanded = cache.Prepare(expandedPlan, 1);
                Assert.HasCount(1, expanded.Batches!);
                Assert.HasCount(1, expanded.Batches![0].Specializations);
                Assert.AreSame(summary.Results[0], expanded.Results[0]);
                cuCtxSetCurrent(context).Ok();
                foreach (var result in summary.Results)
                {
                    LaunchAndValidateRelu(result);
                }
                LaunchAndValidateRelu(expanded.Results[1]);
                var unprepared = forward.Specializations[0] with { TemplateArguments = "float, 512, 0" };
                var lookupTraceStart = writer.GetStringBuilder().Length;
                Assert.ThrowsExactly<InvalidOperationException>(() => cache.Get(unprepared));
                var lookupTrace = writer.ToString()[lookupTraceStart..];
                Assert.DoesNotContain(".nvrtcCompileProgram[", lookupTrace);
                cache.Dispose();
                cache.Dispose();
                Assert.ThrowsExactly<ObjectDisposedException>(() => cache.Get(forward.Specializations[0]));
            }
            finally
            {
                Trace.Listeners.Remove(listener);
            }
        });
    }

    /// <summary>Retries a failed batch so an invalid specialization cannot reject its valid peers.</summary>
    /// <seealso href="https://docs.nvidia.com/cuda/nvrtc/index.html#group__compilation" />
    [TestMethod]
    public void TileGymKernelCacheTest_CUDA13_3RetriesInvalidSpecializationsIndividually()
    {
        WithContext((compiler, context) =>
        {
            using var cache = new TileGymKernelCache(compiler, context);
            var request = new TileGymKernelRequest(ForwardDefinition, ["float, 64, 0", "missing_type, 64, 0"]);
            var flat = request.Specializations.Reverse().ToArray();
            var summary = cache.Prepare(flat, 1);
            Assert.AreEqual(1, summary.Failed);
            Assert.HasCount(3, summary.Batches!);
            Assert.HasCount(2, summary.Batches![0].Specializations);
            Assert.IsNotNull(summary.Batches![0].Error);
            Assert.AreEqual(flat[0], summary.Results[0].Spec);
            Assert.AreEqual(flat[1], summary.Results[1].Spec);
            var valid = summary.Results[1];
            var invalid = summary.Results[0];
            Assert.IsNull(valid.Error);
            Assert.AreNotEqual(default, valid.Function);
            Assert.IsNotNull(invalid.Error);
            Assert.AreEqual(default, invalid.Function);
            Assert.AreSame(invalid, cache.Get(invalid.Spec));
            var unprepared = invalid.Spec with { TemplateArguments = "missing_type, 128, 0" };
            Assert.ThrowsExactly<InvalidOperationException>(() => cache.Get(unprepared));
            var total = summary.Batches!.Sum(static batch => batch.CompileMilliseconds);
            Assert.AreEqual(total, summary.CompileMilliseconds);
            Assert.IsGreaterThan(valid.CompileMilliseconds + invalid.CompileMilliseconds, total);
            LaunchAndValidateRelu(valid);
        });
    }

    /// <summary>Preserves runtime variant selection when building a hierarchical compilation plan.</summary>
    /// <seealso href="https://docs.nvidia.com/cuda/nvrtc/index.html#group__compilation" />
    [TestMethod]
    public void TileGymKernelCacheTest_CUDA13_3RuntimeRespectsNoAutotune()
    {
        RequireNvrtc();
        using var runtime = new TileGymRuntime(0) { EnableAutotuning = false };
        var problem = new ReluProblem();
        TileGymCandidate[] candidates = [new([new("BlockSize", "64")]), new([new("BlockSize", "128")])];
        var kernel = TileGymKernel.Tuned(Header, Forward, ForwardSignature, problem, candidates);
        var benchmark = new TileGymBenchmark(kernel, static (_, _) => { });
        var summary = runtime.Precompile([benchmark, benchmark], 1);
        Assert.HasCount(1, summary.Results);
        Assert.AreEqual("float, 64, 0", summary.Results[0].Spec.TemplateArguments);
        Assert.HasCount(1, summary.Batches!);
        Assert.AreEqual(0, summary.Failed);
    }

    /// <summary>Compiles every catalog specialization in exactly one batch per distinct source header.</summary>
    /// <seealso href="https://docs.nvidia.com/cuda/nvrtc/index.html#group__compilation" />
    [TestMethod]
    public void TileGymKernelCacheTest_CUDA13_3PreparesEntireCatalogByHeader()
    {
        RequireNvrtc();
        using var runtime = new TileGymRuntime(0);
        var options = new TileGymOptions();
        var benchmarks = TileGymCatalog.Scenarios.SelectMany(scenario => scenario.Create(runtime, options)).ToArray();
        var specs = benchmarks.SelectMany(static benchmark => benchmark.Kernels)
            .SelectMany(static kernel => kernel.Variants).Select(static variant => variant.Spec).Distinct().ToArray();
        var headers = specs.Select(static spec => spec.Header).Distinct(StringComparer.Ordinal).ToArray();
        var summary = runtime.Precompile(benchmarks, 1);
        Assert.HasCount(specs.Length, summary.Results);
        Assert.HasCount(headers.Length, summary.Batches!);
        foreach (var result in summary.Results)
        {
            Assert.IsNull(result.Error, result.Spec.ToString());
            Assert.AreNotEqual(default, result.Function, result.Spec.ToString());
        }
        foreach (var batch in summary.Batches!)
        {
            Assert.IsNull(batch.Error, batch.Header);
            Assert.IsGreaterThan(0, batch.TileIrBytes);
        }
    }

    static void RequireNvrtc()
    {
        nvrtcVersion(out var major, out var minor).Ok();
        if (major < 13 || major == 13 && minor < 3)
        {
            Assert.Inconclusive($"CUDA Tile C++ requires NVRTC 13.3 or later; found {major}.{minor}.");
        }
    }

    static void WithContext(Action<TileCppCompiler, CUcontext> action)
    {
        RequireNvrtc();
        CuInit.EnsureInit();
        cuDeviceGet(out var device, 0).Ok();
        cuCtxGetCurrent(out var previousContext).Ok();
        cuDevicePrimaryCtxRetain(out var context, device).Ok();
        try
        {
            cuCtxSetCurrent(context).Ok();
            var architecture = device.GetArchitecture();
            var compiler = new TileCppCompiler(architecture);
            action(compiler, context);
        }
        finally
        {
            cuCtxSetCurrent(previousContext).Ok();
            cuDevicePrimaryCtxRelease(device).Ok();
        }
    }

    static unsafe void LaunchAndValidateRelu(TileGymCompiledKernel kernel)
    {
        const int count = 64;
        var input = new float[count];
        var gradient = new float[count];
        for (var i = 0; i < count; i++)
        {
            input[i] = i - 32;
            gradient[i] = 2;
        }
        using var x = new CudaBuffer<float>(count);
        using var dy = new CudaBuffer<float>(count);
        using var y = new CudaBuffer<float>(count);
        x.CopyFrom(input);
        dy.CopyFrom(gradient);
        var px = x.Pointer.Value;
        var py = y.Pointer.Value;
        var pdy = dy.Pointer.Value;
        var length = count;
        float alpha = 0, lower = 0, upper = 0;
        byte training = 0;
        var forward = stackalloc void*[] { &px, &py, &length, &alpha, &lower, &upper, &training };
        var backward = stackalloc void*[] { &pdy, &px, &py, &length, &alpha, &lower, &upper, &training };
        var isBackward = kernel.Spec.Name == Backward;
        var args = isBackward ? backward : forward;
        cuLaunchKernel(kernel.Function, 1, 1, 1, 1, 1, 1, 0, default, args, null).Ok();
        var result = y.CopyToHost();
        for (var i = 0; i < count; i++)
        {
            var expected = isBackward ? (input[i] > 0 ? 2f : 0f) : Math.Max(input[i], 0);
            Assert.AreEqual(expected, result[i], $"{kernel.Spec}, element {i}");
        }
    }

    sealed class ReluProblem : ITileGymProblem
    {
        public string TemplateArguments(TileGymCandidate candidate) => $"float, {candidate["BlockSize"]}, 0";
        public TileCppGrid Grid(TileGymCandidate candidate) => new(1);
    }
}
