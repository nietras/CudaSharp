using System;
using CudaSharp.Tile;
using static CudaSharp.nvcuda;

namespace CudaSharp.Tester;

static class TileGymMatrixScenarios
{
    static readonly TileCppConfig Config = new([]);

    public static void RunAll(TileGymRuntime runtime, TileGymReport report)
        => RunAll(runtime, report, 64);

    public static void RunAll(TileGymRuntime runtime, TileGymReport report, int matmulSize)
    {
        RunMatmul(runtime, report, false, matmulSize);
        RunMatmul(runtime, report, true, matmulSize);
        RunBmm(runtime, report, false);
        RunBmm(runtime, report, true);
    }

    public static void RunMatmulOnly(TileGymRuntime runtime, TileGymReport report, int matmulSize,
        int? warmupCount = null, int? iterationCount = null, bool skipValidation = false,
        int? tileM = null, int? tileN = null, int? tileK = null, int? occupancy = null)
        => RunMatmul(runtime, report, false, matmulSize, warmupCount, iterationCount, skipValidation,
            tileM, tileN, tileK, occupancy);

    static unsafe void RunMatmul(TileGymRuntime runtime, TileGymReport report, bool persistent, int matmulSize,
        int? warmupCount = null, int? iterationCount = null, bool skipValidation = false,
        int? configuredTileM = null, int? configuredTileN = null, int? configuredTileK = null,
        int? configuredOccupancy = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(matmulSize);
        var m = matmulSize;
        var n = matmulSize;
        var k = matmulSize;
        cuDeviceGetAttribute(out var smCount,
            CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, runtime.Device).Ok();
        var problem = new TileGymMatmulProblem(m, n, k, "float", false, false,
            persistent, runtime.Architecture, smCount);
        var name = persistent ? "static_persistent_matmul_kernel" : "matmul_kernel";
        var candidates = TileGymMatmulCandidates.Select(problem, configuredTileM, configuredTileN,
            configuredTileK, configuredOccupancy);
        using var a = runtime.Allocate<float>(m * k);
        using var b = runtime.Allocate<float>(k * n);
        using var c = runtime.Allocate<float>(m * n);

        var ha = Values(a.Length, .02f);
        var hb = Values(b.Length, .015f);
        a.CopyFrom(ha);
        b.CopyFrom(hb);

        void Launch(TileCppKernel kernel, TileCppConfig config, TileCppGrid grid)
        {
            kernel.Launch(config, grid, runtime.Stream, a.Pointer, b.Pointer, c.Pointer);
        }

        var expected = skipValidation ? null : Gemm(ha, hb, m, n, k);
        var tuned = TileGymTuning.Tune(runtime, problem, candidates,
            problem.Header, name, "const float*, const float*, float*",
            static (p, candidate) => p.TemplateArguments(candidate),
            static (p, candidate) => p.Grid(candidate), Launch,
            validate: expected is null ? null : _ =>
                TileGymKernel.Validate(c.CopyToHost(), expected, name, 4e-3f, 4e-3f));
        var selected = tuned.Candidate;
        void LaunchSelected() => Launch(tuned.Kernel, selected.CompilerConfig, tuned.Grid);
        var timing = warmupCount is { } warmups && iterationCount is { } iterations
            ? TileGymKernel.MeasureFixed(runtime, LaunchSelected, warmups, iterations)
            : TileGymKernel.Measure(runtime, LaunchSelected);
        if (expected is not null)
        {
            TileGymKernel.Validate(c.CopyToHost(), expected, name, 4e-3f, 4e-3f);
        }
        TileGymKernel.Report(report, "matmul", name, $"{m}x{k} @ {k}x{n}",
            a.ByteLength + b.ByteLength + c.ByteLength, timing, tuned);
    }

    static unsafe void RunBmm(TileGymRuntime runtime, TileGymReport report, bool persistent)
    {
        const int batch = 2, m = 64, n = 64, k = 64;
        var name = persistent ? "bmm_static_persistent_kernel" : "bmm_kernel";
        var signature = persistent ? "const float*, const float*, float*" : "const float*, const float*, float*, int, int, int, int";
        cuDeviceGetAttribute(out var smCount,
            CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, runtime.Device).Ok();
        var problem = new TileGymBmmProblem(batch, m, n, k, "float", false, false,
            persistent, runtime.Architecture, smCount);
        var candidates = TileGymBmmCandidates.For(problem);

        using var a = runtime.Allocate<float>(batch * m * k);
        using var b = runtime.Allocate<float>(batch * k * n);
        using var c = runtime.Allocate<float>(batch * m * n);

        var ha = Values(a.Length, .02f);
        var hb = Values(b.Length, .015f);
        a.CopyFrom(ha);
        b.CopyFrom(hb);

        var expected = new float[c.Length];
        for (var q = 0; q < batch; q++)
        {
            Gemm(ha.AsSpan(q * m * k, m * k), hb.AsSpan(q * k * n, k * n), expected.AsSpan(q * m * n, m * n), m, n, k);
        }

        void Launch(TileCppKernel kernel, TileCppConfig config, TileCppGrid grid)
        {
            var pa = a.Pointer.Value;
            var pb = b.Pointer.Value;
            var pc = c.Pointer.Value;
            if (persistent)
            {
                var args = stackalloc IntPtr[] { (IntPtr)(&pa), (IntPtr)(&pb), (IntPtr)(&pc) };
                kernel.Launch(config, grid, runtime.Stream, new(args, 3));
            }
            else
            {
                var q = batch;
                var mm = m;
                var nn = n;
                var kk = k;
                var args = stackalloc IntPtr[]
                {
                    (IntPtr)(&pa), (IntPtr)(&pb), (IntPtr)(&pc), (IntPtr)(&q),
                    (IntPtr)(&mm), (IntPtr)(&nn), (IntPtr)(&kk)
                };
                kernel.Launch(config, grid, runtime.Stream, new(args, 7));
            }
        }

        var tuned = TileGymTuning.Tune(runtime, problem, candidates, "bmm.cuh", name, signature,
            static (p, candidate) => p.TemplateArguments(candidate),
            static (p, candidate) => p.Grid(candidate), Launch,
            validate: _ => TileGymKernel.Validate(c.CopyToHost(), expected, name, 4e-3f, 4e-3f));
        void LaunchSelected() => Launch(tuned.Kernel, tuned.Candidate.CompilerConfig, tuned.Grid);
        var timing = TileGymKernel.Measure(runtime, LaunchSelected);
        TileGymKernel.Validate(c.CopyToHost(), expected, name, 4e-3f, 4e-3f);
        TileGymKernel.Report(report, "bmm", name, $"B={batch},{m}x{k} @ {k}x{n}",
            a.ByteLength + b.ByteLength + c.ByteLength, timing, tuned);
    }

    static float[] Gemm(float[] a, float[] b, int m, int n, int k)
    {
        var c = new float[m * n];
        Gemm(a, b, c, m, n, k);
        return c;
    }

    static void Gemm(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int n, int k)
    {
        for (var row = 0; row < m; row++)
        {
            for (var col = 0; col < n; col++)
            {
                var sum = 0f;
                for (var x = 0; x < k; x++)
                {
                    sum += a[row * k + x] * b[x * n + col];
                }
                c[row * n + col] = sum;
            }
        }
    }

    static float[] Values(int count, float scale)
    {
        var a = new float[count];
        for (var i = 0; i < count; i++)
        {
            a[i] = (i % 19 - 9) * scale;
        }
        return a;
    }
}
