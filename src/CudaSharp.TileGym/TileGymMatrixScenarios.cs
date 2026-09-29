using System.Collections.Generic;
using CudaSharp.TileGym;
using static CudaSharp.nvcuda;

namespace CudaSharp.Tester;

static class TileGymMatrixScenarios
{
    /// <summary>
    /// Creates the matmul and bmm benchmarks. Non-default matmul options select a matmul-only run so tile
    /// configurations and measurement counts can be explored for a single kernel.
    /// </summary>
    public static IEnumerable<TileGymBenchmark> Create(TileGymRuntime runtime, TileGymOptions options)
    {
        var matmul = options.Matmul;
        yield return Matmul(runtime, persistent: false, matmul);
        if (!matmul.IsDefault)
        {
            yield break;
        }
        yield return Matmul(runtime, persistent: true, matmul);
        yield return Bmm(runtime, persistent: false);
        yield return Bmm(runtime, persistent: true);
    }

    static TileGymBenchmark Matmul(TileGymRuntime runtime, bool persistent, TileGymMatmulOptions options)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.Size);
        var m = options.Size;
        var n = options.Size;
        var k = options.Size;
        var problem = new TileGymMatmulProblem(m, n, k, "float", false, false,
            persistent, runtime.Architecture, runtime.SmCount);
        var name = persistent ? "static_persistent_matmul_kernel" : "matmul_kernel";
        var candidates = TileGymMatmulCandidates.Select(problem, options.TileM, options.TileN,
            options.TileK, options.Occupancy);
        var kernel = TileGymKernel.Tuned(problem.Header, name, "const float*, const float*, float*",
            problem, candidates);
        return new(kernel, (runtime, report) =>
        {
            using var a = runtime.Allocate<float>(m * k);
            using var b = runtime.Allocate<float>(k * n);
            using var c = runtime.Allocate<float>(m * n);
            var ha = Values(a.Length, .02f);
            var hb = Values(b.Length, .015f);
            a.CopyFrom(ha);
            b.CopyFrom(hb);
            void Launch(CUfunction function, TileCppGrid grid) =>
                cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                    a.Pointer, b.Pointer, c.Pointer).Ok();
            Action? validate = null;
            if (!options.SkipValidation)
            {
                var expected = Gemm(ha, hb, m, n, k);
                validate = () => c.Validate(expected, name, 4e-3f, 4e-3f);
            }
            var measure = new TileGymMeasureOptions(options.Warmup, options.Iterations);
            var run = runtime.Run(kernel, Launch, validate, options: measure);
            report.Add("matmul", $"{m}x{k} @ {k}x{n}", a.ByteLength + b.ByteLength + c.ByteLength, run);
        });
    }

    static TileGymBenchmark Bmm(TileGymRuntime runtime, bool persistent)
    {
        const int batch = 8, m = 256, n = 256, k = 256;
        var name = persistent ? "bmm_static_persistent_kernel" : "bmm_kernel";
        var signature = persistent
            ? "const float*, const float*, float*"
            : "const float*, const float*, float*, int, int, int, int";
        var problem = new TileGymBmmProblem(batch, m, n, k, "float", false, false,
            persistent, runtime.Architecture, runtime.SmCount);
        var candidates = TileGymBmmCandidates.For(problem);
        var kernel = TileGymKernel.Tuned("bmm.cuh", name, signature, problem, candidates);
        return new(kernel, (runtime, report) =>
        {
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
                var aq = ha.AsSpan(q * m * k, m * k);
                var bq = hb.AsSpan(q * k * n, k * n);
                var cq = expected.AsSpan(q * m * n, m * n);
                Gemm(aq, bq, cq, m, n, k);
            }
            void Launch(CUfunction function, TileCppGrid grid)
            {
                if (persistent)
                {
                    cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                        a.Pointer, b.Pointer, c.Pointer).Ok();
                }
                else
                {
                    cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                        a.Pointer, b.Pointer, c.Pointer, batch, m, n, k).Ok();
                }
            }
            var run = runtime.Run(kernel, Launch, () => c.Validate(expected, name, 4e-3f, 4e-3f));
            report.Add("bmm", $"B={batch},{m}x{k} @ {k}x{n}", a.ByteLength + b.ByteLength + c.ByteLength, run);
        });
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
