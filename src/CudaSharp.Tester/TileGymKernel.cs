using System;
using System.Diagnostics;
using System.IO;
using CudaSharp.Tile;
using static CudaSharp.nvcuda;

namespace CudaSharp.Tester;

static class TileGymKernel
{
    public static TileCppKernel Create(TileGymRuntime runtime, string relativeHeader, string kernelName,
        string templateArguments, string signature)
    {
        var headerPath = Path.Combine(AppContext.BaseDirectory, "src-tilecpp", "tilegym", relativeHeader);
        var headerSource = File.ReadAllText(headerPath)
            .Replace(
                $"__tile_global__ void {kernelName}",
                $"__attribute__((used)) __tile_global__ void {kernelName}",
                StringComparison.Ordinal)
            .Replace("INFINITY", "3.402823466e+38F", StringComparison.Ordinal);
        var headerName = Path.GetFileName(relativeHeader);
        var source = $$"""
            using int32_t = int;
            using uint32_t = unsigned int;
            using int64_t = long long;
            using uint64_t = unsigned long long;
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
        return new TileCppKernel(
            new TileCppCompiler(runtime.Architecture),
            source,
            $"{kernelName}.cu",
            kernelName,
            [
                new TileCppHeader(headerName, headerSource),
                new TileCppHeader("type_traits", typeTraits),
                new TileCppHeader("cmath", string.Empty)
            ],
            nameExpression: $"&{kernelName}<{templateArguments}>");
    }

    public static (double CompileMilliseconds, double KernelMilliseconds) Measure(
        TileGymRuntime runtime, Action launch)
    {
        var compile = Stopwatch.StartNew();
        launch();
        cuStreamSynchronize(runtime.Stream).Ok();
        compile.Stop();
        var milliseconds = new CudaEventTileCppTimer().Measure(
            launch, runtime.Stream, new TileCppTimingOptions());
        return (compile.Elapsed.TotalMilliseconds, milliseconds);
    }

    public static (double CompileMilliseconds, double KernelMilliseconds) MeasureOnce(
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
            return (compilation.Elapsed.TotalMilliseconds, milliseconds);
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
        string configuration, nuint bytes, (double CompileMilliseconds, double KernelMilliseconds) timing)
    {
        var throughput = bytes / (timing.KernelMilliseconds * 1_000_000.0);
        report.Add(new TileGymResult(
            family,
            kernel,
            shape,
            configuration,
            "Passed",
            timing.CompileMilliseconds,
            0,
            timing.KernelMilliseconds,
            throughput,
            "GB/s",
            null));
    }
}
