using System.Diagnostics;
using System.IO;
using CudaSharp.TileGym;
using static CudaSharp.nvcuda;

namespace CudaSharp.Tester;

readonly record struct TileGymPhaseTiming(double CompileMilliseconds, double LoadMilliseconds,
    double FirstLaunchMilliseconds, double? HostMilliseconds, double KernelMilliseconds);
readonly record struct TileGymTiming(double FirstUseMilliseconds, double KernelMilliseconds,
    double? HostMilliseconds = null);

static class TileGymKernel
{
    public static TileCppKernel Create(TileCppCompiler compiler, string relativeHeader, string kernelName,
        string templateArguments, string signature)
    {
        ArgumentNullException.ThrowIfNull(compiler);
        var headerPath = Path.Combine(AppContext.BaseDirectory, "src-tilecpp", "tilegym", relativeHeader);
        var headerSource = File.ReadAllText(headerPath);
        var headerName = Path.GetFileName(relativeHeader);
        var source = $$"""
            using int32_t = int;
            using uint32_t = unsigned int;
            using int64_t = long long;
            using uint64_t = unsigned long long;
            #include <cmath>
            #include "{{headerName}}"
            template __tile_global__ void {{kernelName}}<{{templateArguments}}>({{signature}});
            """;
        const string typeTraits =
            "namespace std { template<bool B, class T, class F> struct conditional { using type = T; }; " +
            "template<class T, class F> struct conditional<false, T, F> { using type = F; }; " +
            "template<bool B, class T, class F> using conditional_t = typename conditional<B, T, F>::type; " +
            "template<class A, class B> struct is_same { static constexpr bool value = false; }; " +
            "template<class A> struct is_same<A, A> { static constexpr bool value = true; }; " +
            "template<class A, class B> inline constexpr bool is_same_v = is_same<A, B>::value; }";
        var headers = new System.Collections.Generic.List<TileCppHeader>
        {
            new(headerName, headerSource),
            new("type_traits", typeTraits),
            new("cmath", "#ifndef INFINITY\n#define INFINITY __builtin_bit_cast(float, 0x7f800000u)\n#endif\n")
        };
        if (relativeHeader.StartsWith("conv", StringComparison.Ordinal))
        {
            headers.Add(new TileCppHeader("convolution_common.cuh",
                File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "src-tilecpp", "tilegym", "convolution_common.cuh"))));
        }
        return new TileCppKernel(compiler,
            source, $"{kernelName}.cu", kernelName, headers,
            nameExpression: $"&{kernelName}<{templateArguments}>");
    }

    public static TileGymPhaseTiming MeasurePhases(TileGymRuntime runtime, TileCppKernel kernel,
        TileCppConfig config, Action<CUfunction> launch)
    {
        var watch = Stopwatch.StartNew();
        kernel.Compile(config);
        watch.Stop();
        var compile = watch.Elapsed.TotalMilliseconds;

        watch.Restart();
        var function = kernel.LoadFunction(config);
        watch.Stop();
        var load = watch.Elapsed.TotalMilliseconds;

        watch.Restart();
        launch(function);
        cuStreamSynchronize(runtime.Stream).Ok();
        watch.Stop();
        var firstLaunch = watch.Elapsed.TotalMilliseconds;

        void Launch() => launch(function);
        var device = new CudaEventTileCppTimer().Measure(Launch, runtime.Stream, new TileCppTimingOptions());
        var host = MeasureHost(runtime, Launch);
        return new(compile, load, firstLaunch, host, device);
    }

    public static TileGymPhaseTiming MeasureOncePhases(TileGymRuntime runtime, TileCppKernel kernel,
        TileCppConfig config, Action launch)
    {
        var watch = Stopwatch.StartNew();
        kernel.Compile(config);
        watch.Stop();
        var compile = watch.Elapsed.TotalMilliseconds;
        watch.Restart();
        kernel.LoadFunction(config);
        watch.Stop();
        var load = watch.Elapsed.TotalMilliseconds;

        cuEventCreate(out var start, 0).Ok();
        cuEventCreate(out var end, 0).Ok();
        try
        {
            watch.Restart();
            cuEventRecord(start, runtime.Stream).Ok();
            launch();
            cuEventRecord(end, runtime.Stream).Ok();
            cuEventSynchronize(end).Ok();
            watch.Stop();
            cuEventElapsedTime(out var milliseconds, start, end).Ok();
            return new(compile, load, watch.Elapsed.TotalMilliseconds, null, milliseconds);
        }
        finally
        {
            cuEventDestroy(start).Ok();
            cuEventDestroy(end).Ok();
        }
    }

    public static TileGymPhaseTiming MeasurePhases(TileGymRuntime runtime, TileCppKernel kernel,
        TileCppConfig config, Action launch, int? warmupCount = null, int? iterationCount = null)
    {
        var watch = Stopwatch.StartNew();
        kernel.Compile(config);
        watch.Stop();
        var compile = watch.Elapsed.TotalMilliseconds;

        watch.Restart();
        kernel.LoadFunction(config);
        watch.Stop();
        var load = watch.Elapsed.TotalMilliseconds;

        watch.Restart();
        launch();
        cuStreamSynchronize(runtime.Stream).Ok();
        watch.Stop();
        var firstLaunch = watch.Elapsed.TotalMilliseconds;

        if (warmupCount is { } warmups && iterationCount is { } iterations)
        {
            var timing = MeasureFixed(runtime, launch, warmups, iterations);
            return new(compile, load, firstLaunch, timing.HostMilliseconds!.Value, timing.KernelMilliseconds);
        }
        var device = new CudaEventTileCppTimer().Measure(launch, runtime.Stream, new TileCppTimingOptions());
        var host = MeasureHost(runtime, launch);
        return new(compile, load, firstLaunch, host, device);
    }

    static double MeasureHost(TileGymRuntime runtime, Action launch, int iterations = 100)
    {
        cuStreamSynchronize(runtime.Stream).Ok();
        var watch = Stopwatch.StartNew();
        for (var i = 0; i < iterations; i++)
            launch();
        cuStreamSynchronize(runtime.Stream).Ok();
        watch.Stop();
        return watch.Elapsed.TotalMilliseconds / iterations;
    }

    public static TileGymTiming MeasureFixed(
        TileGymRuntime runtime, Action launch, int warmupCount, int iterationCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(warmupCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(iterationCount);

        var compile = Stopwatch.StartNew();
        launch();
        cuStreamSynchronize(runtime.Stream).Ok();
        compile.Stop();

        for (var i = 0; i < warmupCount; i++)
        {
            launch();
        }
        cuStreamSynchronize(runtime.Stream).Ok();

        cuEventCreate(out var start, 0).Ok();
        cuEventCreate(out var end, 0).Ok();
        try
        {
            cuEventRecord(start, runtime.Stream).Ok();
            for (var i = 0; i < iterationCount; i++)
            {
                launch();
            }
            cuEventRecord(end, runtime.Stream).Ok();
            cuEventSynchronize(end).Ok();
            cuEventElapsedTime(out var milliseconds, start, end).Ok();
            var host = MeasureHost(runtime, launch, iterationCount);
            return new(compile.Elapsed.TotalMilliseconds, milliseconds / iterationCount, host);
        }
        finally
        {
            cuEventDestroy(start).Ok();
            cuEventDestroy(end).Ok();
        }
    }

    public static TileGymTiming Measure(
        TileGymRuntime runtime, Action launch)
    {
        var compile = Stopwatch.StartNew();
        launch();
        cuStreamSynchronize(runtime.Stream).Ok();
        compile.Stop();
        var milliseconds = new CudaEventTileCppTimer().Measure(
            launch, runtime.Stream, new TileCppTimingOptions());
        return new(compile.Elapsed.TotalMilliseconds, milliseconds, MeasureHost(runtime, launch));
    }

    public static TileGymTiming MeasureOnce(
        TileGymRuntime runtime, Action compile, Action launch)
    {
        var compilation = Stopwatch.StartNew();
        compile();
        cuStreamSynchronize(runtime.Stream).Ok();
        compilation.Stop();
        cuEventCreate(out var start, 0).Ok();
        cuEventCreate(out var end, 0).Ok();
        try
        {
            cuEventRecord(start, runtime.Stream).Ok();
            launch();
            cuEventRecord(end, runtime.Stream).Ok();
            cuEventSynchronize(end).Ok();
            cuEventElapsedTime(out var milliseconds, start, end).Ok();
            return new(compilation.Elapsed.TotalMilliseconds, milliseconds);
        }
        finally
        {
            cuEventDestroy(start).Ok();
            cuEventDestroy(end).Ok();
        }
    }

    public static void Validate(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected,
        string kernel, float absoluteTolerance = 2e-5f, float relativeTolerance = 2e-5f)
    {
        if (actual.Length != expected.Length)
        {
            throw new InvalidOperationException($"{kernel} validation length mismatch.");
        }
        for (var i = 0; i < actual.Length; i++)
        {
            var tolerance = absoluteTolerance + relativeTolerance * Math.Abs(expected[i]);
            if (!float.IsFinite(actual[i]) || Math.Abs(actual[i] - expected[i]) > tolerance)
            {
                throw new InvalidOperationException(
                    $"{kernel} validation failed at {i}: {actual[i]} != {expected[i]} (tolerance {tolerance}).");
            }
        }
    }

    public static void Report(TileGymReport report, string family, string kernel, string shape,
        string configuration, nuint bytes, TileGymTiming timing)
    {
        var throughput = bytes / (timing.KernelMilliseconds * 1_000_000.0);
        report.Add(new TileGymResult(
            family,
            kernel,
            shape,
            configuration,
            "Passed",
            null,
            0,
            timing.KernelMilliseconds,
            throughput,
            "GB/s",
            null,
            timing.HostMilliseconds,
            FirstUseMilliseconds: timing.FirstUseMilliseconds));
    }

    public static void Report(TileGymReport report, string family, string kernel, string shape,
        string configuration, nuint bytes, TileGymPhaseTiming timing)
    {
        var throughput = bytes / (timing.KernelMilliseconds * 1_000_000.0);
        report.Add(new TileGymResult(
            family, kernel, shape, configuration, "Passed",
            timing.CompileMilliseconds, 0, timing.KernelMilliseconds, throughput, "GB/s", null,
            timing.HostMilliseconds, timing.LoadMilliseconds, timing.FirstLaunchMilliseconds));
    }

    public static void Report(TileGymReport report, string family, string kernel, string shape,
        nuint bytes, TileGymTiming timing, TileGymTunedResult tuned)
    {
        var throughput = bytes / (timing.KernelMilliseconds * 1_000_000.0);
        report.Add(new TileGymResult(
            family,
            kernel,
            shape,
            tuned.Candidate.ToString(),
            tuned.CandidateCount == 1 ? "Passed (fixed)" : tuned.CacheHit ? "Passed (cached)" : "Passed (searched)",
            tuned.CompileMilliseconds,
            tuned.TuneMilliseconds,
            timing.KernelMilliseconds,
            throughput,
            "GB/s",
            tuned.CandidateCount == 1 ? "Compilation and initial measurement included in tuning time." :
                $"{tuned.CandidateCount} candidates offered; rejected variants may be skipped. Compilation included in tuning time.",
            timing.HostMilliseconds,
            tuned.LoadMilliseconds,
            tuned.FirstLaunchMilliseconds,
            timing.FirstUseMilliseconds));
    }
}
