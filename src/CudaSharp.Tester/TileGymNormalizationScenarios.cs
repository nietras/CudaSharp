using System;
using CudaSharp.Tile;

namespace CudaSharp.Tester;

static class TileGymNormalizationScenarios
{
    static readonly TileCppConfig Config = new([]);

    public static void RunAll(TileGymRuntime runtime, TileGymReport report)
    {
        RunLayerNorm(runtime, report, persistent: false);
        RunLayerNorm(runtime, report, persistent: true);
        RunRmsNorm(
            runtime,
            report,
            "rms_norm_kernel",
            "float, float, 256, 256, 0.00001f, 0.0f",
            "const float*, const float*, float*, float*, int",
            new(8)
        );
        RunRmsNorm(
            runtime,
            report,
            "rms_norm_multi_wave_cached_kernel",
            "float, float, 256, 256, 0.00001f, 0.0f",
            "const float*, const float*, float*, float*, int",
            new(8)
        );
        RunRmsNormPv(runtime, report);
        RunRmsNormPersistent(runtime, report);
        RunRmsNormBackward(runtime, report);
    }

    static unsafe void RunLayerNorm(TileGymRuntime runtime, TileGymReport report, bool persistent)
    {
        const int rows = 8;
        const int columns = 256;
        var name = persistent ? "persistent_layer_norm_fwd_kernel" : "layer_norm_fwd_fused_kernel";
        var templates = persistent
            ? "float, float, float, 2, 256, false, true, true, 8, 256, 4, 0.00001f"
            : "float, float, float, 256, 256";
        var signature = persistent
            ? "const float*, float*, const float*, const float*, float*, float*"
            : "float*, float*, const float*, const float*, float*, float*, float, float";
        var header = persistent ? "persistent_layer_norm.cuh" : "layer_norm_legacy.cuh";
        using var kernel = TileGymKernel.Create(runtime, header, name, templates, signature);
        using var x = runtime.Allocate<float>(rows * columns);
        using var y = runtime.Allocate<float>(x.Length);
        using var w = runtime.Allocate<float>(columns);
        using var b = runtime.Allocate<float>(columns);
        using var mean = runtime.Allocate<float>(rows);
        using var rstd = runtime.Allocate<float>(rows);
        var hx = Values(x.Length);
        var hw = new float[columns];
        var hb = new float[columns];
        for (var i = 0; i < columns; i++)
        {
            hw[i] = .75f + i / 1024f;
            hb[i] = (i % 17 - 8) / 128f;
        }

        x.CopyFrom(hx);
        w.CopyFrom(hw);
        b.CopyFrom(hb);

        void Launch()
        {
            var px = x.Pointer.Value;
            var py = y.Pointer.Value;
            var pw = w.Pointer.Value;
            var pb = b.Pointer.Value;
            var pm = mean.Pointer.Value;
            var pr = rstd.Pointer.Value;
            var eps = .00001f;
            var shift = 0f;
            if (persistent)
            {
                var args = stackalloc IntPtr[]
                {
                    (IntPtr)(&px),
                    (IntPtr)(&py),
                    (IntPtr)(&pw),
                    (IntPtr)(&pb),
                    (IntPtr)(&pm),
                    (IntPtr)(&pr)
                };
                kernel.Launch(Config, new(4), runtime.Stream, new(args, 6));
            }
            else
            {
                var args = stackalloc IntPtr[]
                {
                    (IntPtr)(&px),
                    (IntPtr)(&py),
                    (IntPtr)(&pw),
                    (IntPtr)(&pb),
                    (IntPtr)(&pm),
                    (IntPtr)(&pr),
                    (IntPtr)(&eps),
                    (IntPtr)(&shift)
                };
                kernel.Launch(Config, new(rows), runtime.Stream, new(args, 8));
            }
        }

        var timing = TileGymKernel.Measure(runtime, Launch);
        var expected = LayerNorm(hx, hw, hb, rows, columns, .00001f);
        TileGymKernel.Validate(y.CopyToHost(), expected, name, 8e-4f, 8e-4f);
        TileGymKernel.Report(
            report,
            "normalization",
            name,
            $"{rows}x{columns}",
            templates,
            x.ByteLength + y.ByteLength + w.ByteLength + b.ByteLength,
            timing
        );
    }

    static unsafe void RunRmsNorm(TileGymRuntime runtime, TileGymReport report, string name,
        string templates, string signature, TileCppGrid grid)
    {
        const int rows = 8;
        const int columns = 256;
        using var kernel = TileGymKernel.Create(runtime, "rms_norm.cuh", name, templates, signature);
        using var x = runtime.Allocate<float>(rows * columns);
        using var y = runtime.Allocate<float>(x.Length);
        using var w = runtime.Allocate<float>(columns);
        using var rstd = runtime.Allocate<float>(rows);
        var hx = Values(x.Length);
        var hw = Weights(columns);
        x.CopyFrom(hx);
        w.CopyFrom(hw);

        void Launch()
        {
            var px = x.Pointer.Value;
            var pw = w.Pointer.Value;
            var py = y.Pointer.Value;
            var pr = rstd.Pointer.Value;
            var stride = columns;
            var args = stackalloc IntPtr[]
            {
                (IntPtr)(&px),
                (IntPtr)(&pw),
                (IntPtr)(&py),
                (IntPtr)(&pr),
                (IntPtr)(&stride)
            };
            kernel.Launch(Config, grid, runtime.Stream, new(args, 5));
        }

        var timing = TileGymKernel.Measure(runtime, Launch);
        TileGymKernel.Validate(y.CopyToHost(), RmsNorm(hx, hw, rows, columns), name, 8e-4f, 8e-4f);
        TileGymKernel.Report(report, "normalization", name, $"{rows}x{columns}", templates, x.ByteLength + y.ByteLength + w.ByteLength, timing);
    }

    static unsafe void RunRmsNormPv(TileGymRuntime runtime, TileGymReport report)
    {
        const int rows = 8;
        const int columns = 256;
        const string name = "rms_norm_kernel_pv";
        using var kernel = TileGymKernel.Create(
            runtime,
            "rms_norm.cuh",
            name,
            "float, float, 8, 256, 256",
            "const float*, const float*, float*, float*, float"
        );
        using var x = runtime.Allocate<float>(rows * columns);
        using var y = runtime.Allocate<float>(x.Length);
        using var w = runtime.Allocate<float>(columns);
        using var r = runtime.Allocate<float>(rows);
        var hx = Values(x.Length);
        var hw = Weights(columns);
        x.CopyFrom(hx);
        w.CopyFrom(hw);

        void Launch()
        {
            var px = x.Pointer.Value;
            var pw = w.Pointer.Value;
            var py = y.Pointer.Value;
            var pr = r.Pointer.Value;
            var eps = .00001f;
            var args = stackalloc IntPtr[]
            {
                (IntPtr)(&px),
                (IntPtr)(&pw),
                (IntPtr)(&py),
                (IntPtr)(&pr),
                (IntPtr)(&eps)
            };
            kernel.Launch(Config, new(rows), runtime.Stream, new(args, 5));
        }

        var timing = TileGymKernel.Measure(runtime, Launch);
        TileGymKernel.Validate(y.CopyToHost(), RmsNorm(hx, hw, rows, columns), name, 8e-4f, 8e-4f);
        TileGymKernel.Report(
            report,
            "normalization",
            name,
            $"{rows}x{columns}",
            "float,BLOCK_SIZE=256",
            x.ByteLength + y.ByteLength + w.ByteLength,
            timing
        );
    }

    static unsafe void RunRmsNormPersistent(TileGymRuntime runtime, TileGymReport report)
    {
        const int rows = 8;
        const int columns = 256;
        const string name = "rms_norm_static_persistent_kernel";
        using var kernel = TileGymKernel.Create(
            runtime,
            "rms_norm.cuh",
            name,
            "float, float, 2, 256, 1, 8, 256, 4, 0.00001f, 0.0f",
            "const float*, float*, const float*, float*"
        );
        using var x = runtime.Allocate<float>(rows * columns);
        using var y = runtime.Allocate<float>(x.Length);
        using var w = runtime.Allocate<float>(columns);
        using var r = runtime.Allocate<float>(rows);
        var hx = Values(x.Length);
        var hw = Weights(columns);
        x.CopyFrom(hx);
        w.CopyFrom(hw);

        void Launch()
        {
            var px = x.Pointer.Value;
            var py = y.Pointer.Value;
            var pw = w.Pointer.Value;
            var pr = r.Pointer.Value;
            var args = stackalloc IntPtr[]
            {
                (IntPtr)(&px),
                (IntPtr)(&py),
                (IntPtr)(&pw),
                (IntPtr)(&pr)
            };
            kernel.Launch(Config, new(4), runtime.Stream, new(args, 4));
        }

        var timing = TileGymKernel.Measure(runtime, Launch);
        TileGymKernel.Validate(y.CopyToHost(), RmsNorm(hx, hw, rows, columns), name, 8e-4f, 8e-4f);
        TileGymKernel.Report(
            report,
            "normalization",
            name,
            $"{rows}x{columns}",
            "float,TILE_M=2,TILE_N=256",
            x.ByteLength + y.ByteLength + w.ByteLength,
            timing
        );
    }

    static unsafe void RunRmsNormBackward(TileGymRuntime runtime, TileGymReport report)
    {
        const int rows = 8;
        const int columns = 256;
        const string name = "rms_norm_backward_dx_kernel";
        using var kernel = TileGymKernel.Create(
            runtime,
            "rms_norm.cuh",
            name,
            "float, float, 256",
            "float*, const float*, const float*, const float*, const float*, float*, int, int"
        );
        using var dx = runtime.Allocate<float>(rows * columns);
        using var dy = runtime.Allocate<float>(dx.Length);
        using var x = runtime.Allocate<float>(dx.Length);
        using var w = runtime.Allocate<float>(columns);
        using var r = runtime.Allocate<float>(rows);
        using var temp = runtime.Allocate<float>(dx.Length);
        var hx = Values(x.Length);
        var hw = Weights(columns);
        var hdy = new float[dy.Length];
        Array.Fill(hdy, 1f);
        var hr = Rstd(hx, rows, columns);
        x.CopyFrom(hx);
        w.CopyFrom(hw);
        dy.CopyFrom(hdy);
        r.CopyFrom(hr);

        void Launch()
        {
            var pdx = dx.Pointer.Value;
            var pdy = dy.Pointer.Value;
            var px = x.Pointer.Value;
            var pw = w.Pointer.Value;
            var pr = r.Pointer.Value;
            var pt = temp.Pointer.Value;
            var stride = columns;
            var n = columns;
            var args = stackalloc IntPtr[]
            {
                (IntPtr)(&pdx),
                (IntPtr)(&pdy),
                (IntPtr)(&px),
                (IntPtr)(&pw),
                (IntPtr)(&pr),
                (IntPtr)(&pt),
                (IntPtr)(&stride),
                (IntPtr)(&n)
            };
            kernel.Launch(Config, new(rows), runtime.Stream, new(args, 8));
        }

        var timing = TileGymKernel.Measure(runtime, Launch);
        var expected = new float[dx.Length];
        for (var row = 0; row < rows; row++)
        {
            var dot = 0f;
            for (var c = 0; c < columns; c++) dot += hw[c] * hx[row * columns + c];
            for (var c = 0; c < columns; c++)
            {
                var value = hx[row * columns + c];
                expected[row * columns + c] = hw[c] * hr[row] - value * (hr[row] * hr[row] * hr[row] / columns) * dot;
            }
        }

        TileGymKernel.Validate(dx.CopyToHost(), expected, name, 1e-3f, 1e-3f);
        TileGymKernel.Report(report, "normalization", name, $"{rows}x{columns}", "float,BLOCK_SIZE=256",
            dx.ByteLength + dy.ByteLength + x.ByteLength + w.ByteLength + r.ByteLength + temp.ByteLength, timing);
    }

    static float[] Values(int count) { var a = new float[count]; for (var i = 0; i < count; i++) a[i] = (i % 251 - 125) / 64f; return a; }
    static float[] Weights(int n) { var a = new float[n]; for (var i = 0; i < n; i++) a[i] = .75f + i / 1024f; return a; }
    static float[] Rstd(float[] x, int rows, int n)
    {
        var r = new float[rows];

        for (var row = 0; row < rows; row++)
        {
            var s = 0f;

            for (var c = 0; c < n; c++)
            {
                var v = x[row * n + c];
                s += v * v;
            }

            r[row] = 1f / MathF.Sqrt(s / n + .00001f);
        }

        return r;
    }
    static float[] RmsNorm(float[] x, float[] w, int rows, int n)
    {
        var r = Rstd(x, rows, n);
        var y = new float[x.Length];

        for (var row = 0; row < rows; row++)
        {
            for (var c = 0; c < n; c++)
            {
                y[row * n + c] = x[row * n + c] * r[row] * w[c];
            }
        }

        return y;
    }
    static float[] LayerNorm(float[] x, float[] w, float[] b, int rows, int n, float eps)
    {
        var y = new float[x.Length];

        for (var row = 0; row < rows; row++)
        {
            var mean = 0f;

            for (var c = 0; c < n; c++)
            {
                mean += x[row * n + c];
            }

            mean /= n;

            var variance = 0f;

            for (var c = 0; c < n; c++)
            {
                var d = x[row * n + c] - mean;
                variance += d * d;
            }

            var rs = 1f / MathF.Sqrt(variance / n + eps);

            for (var c = 0; c < n; c++)
            {
                y[row * n + c] = (x[row * n + c] - mean) * rs * w[c] + b[c];
            }
        }

        return y;
    }
}
