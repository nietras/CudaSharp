using System;
using CudaSharp.Tile;

namespace CudaSharp.Tester;

static class TileGymMatrixScenarios
{
    static readonly TileCppConfig Config = new([]);

    public static void RunAll(TileGymRuntime runtime, TileGymReport report)
    {
        RunMatmul(runtime, report, false);
        RunMatmul(runtime, report, true);
        RunBmm(runtime, report, false);
        RunBmm(runtime, report, true);
    }

    static unsafe void RunMatmul(TileGymRuntime runtime, TileGymReport report, bool persistent)
    {
        const int m = 64, n = 64, k = 64;
        var name = persistent ? "static_persistent_matmul_kernel" : "matmul_kernel";
        var templates = persistent
            ? "float, 64, 64, 64, 64, 64, 32, 8, false, false, 1, 2"
            : "float, 64, 64, 64, 64, 64, 32, 8, 2, false, false, 1, 2";
        var header = persistent ? "persistent_matmul.cuh" : "matmul.cuh";

        using var kernel = TileGymKernel.Create(runtime, header, name, templates, "const float*, const float*, float*");
        using var a = runtime.Allocate<float>(m * k);
        using var b = runtime.Allocate<float>(k * n);
        using var c = runtime.Allocate<float>(m * n);

        var ha = Values(a.Length, .02f);
        var hb = Values(b.Length, .015f);
        a.CopyFrom(ha);
        b.CopyFrom(hb);

        void Launch()
        {
            var pa = a.Pointer.Value;
            var pb = b.Pointer.Value;
            var pc = c.Pointer.Value;
            var args = stackalloc IntPtr[] { (IntPtr)(&pa), (IntPtr)(&pb), (IntPtr)(&pc) };
            kernel.Launch(Config, new(1), runtime.Stream, new(args, 3));
        }

        var timing = TileGymKernel.Measure(runtime, Launch);
        TileGymKernel.Validate(c.CopyToHost(), Gemm(ha, hb, m, n, k), name, 4e-3f, 4e-3f);
        TileGymKernel.Report(report, "matmul", name, $"{m}x{k} @ {k}x{n}", templates, a.ByteLength + b.ByteLength + c.ByteLength, timing);
    }

    static unsafe void RunBmm(TileGymRuntime runtime, TileGymReport report, bool persistent)
    {
        const int batch = 2, m = 64, n = 64, k = 64;
        var name = persistent ? "bmm_static_persistent_kernel" : "bmm_kernel";
        var templates = persistent ? "float, 64, 64, 32, 8, false, false, 2, 64, 64, 64, 1, 2" : "float, 64, 64, 32, 8, false, false";
        var signature = persistent ? "const float*, const float*, float*" : "const float*, const float*, float*, int, int, int, int";

        using var kernel = TileGymKernel.Create(runtime, "bmm.cuh", name, templates, signature);
        using var a = runtime.Allocate<float>(batch * m * k);
        using var b = runtime.Allocate<float>(batch * k * n);
        using var c = runtime.Allocate<float>(batch * m * n);

        var ha = Values(a.Length, .02f);
        var hb = Values(b.Length, .015f);
        a.CopyFrom(ha);
        b.CopyFrom(hb);

        void Launch()
        {
            var pa = a.Pointer.Value;
            var pb = b.Pointer.Value;
            var pc = c.Pointer.Value;
            if (persistent)
            {
                var args = stackalloc IntPtr[] { (IntPtr)(&pa), (IntPtr)(&pb), (IntPtr)(&pc) };
                kernel.Launch(Config, new(2), runtime.Stream, new(args, 3));
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
                kernel.Launch(Config, new(1, batch), runtime.Stream, new(args, 7));
            }
        }

        var timing = TileGymKernel.Measure(runtime, Launch);
        var expected = new float[c.Length];
        for (var q = 0; q < batch; q++)
        {
            Gemm(ha.AsSpan(q * m * k, m * k), hb.AsSpan(q * k * n, k * n), expected.AsSpan(q * m * n, m * n), m, n, k);
        }
        TileGymKernel.Validate(c.CopyToHost(), expected, name, 4e-3f, 4e-3f);
        TileGymKernel.Report(
            report,
            "bmm",
            name,
            $"B={batch},{m}x{k} @ {k}x{n}",
            templates,
            a.ByteLength + b.ByteLength + c.ByteLength,
            timing);
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
