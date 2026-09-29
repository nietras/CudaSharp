using System.Collections.Generic;
using CudaSharp.TileGym;
using static CudaSharp.nvcuda;

namespace CudaSharp.Tester;

static class TileGymNormalizationScenarios
{
    const int Rows = 8;
    const int Columns = 256;
    const string Shape = "8x256";

    public static IEnumerable<TileGymBenchmark> Create(TileGymRuntime runtime, TileGymOptions options)
    {
        yield return LayerNorm();
        yield return PersistentLayerNorm(runtime.SmCount);
        yield return RmsNorm("rms_norm_kernel");
        yield return RmsNorm("rms_norm_multi_wave_cached_kernel");
        yield return RmsNormPv();
        yield return RmsNormPersistent(runtime.SmCount);
        yield return RmsNormBackward();
    }

    static TileGymBenchmark LayerNorm()
    {
        const string name = "layer_norm_fwd_fused_kernel";
        var kernel = TileGymKernel.Fixed("layer_norm_legacy.cuh", name, "float, float, float, 256, 256",
            "float*, float*, const float*, const float*, float*, float*, float, float", new TileCppGrid(Rows));
        return new(kernel, (runtime, report) =>
        {
            using var buffers = new LayerNormBuffers(runtime);
            void Launch(CUfunction function, TileCppGrid grid)
            {
                var eps = .00001f;
                var shift = 0f;
                cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                    buffers.X.Pointer, buffers.Y.Pointer, buffers.W.Pointer, buffers.B.Pointer,
                    buffers.Mean.Pointer, buffers.Rstd.Pointer, eps, shift).Ok();
            }
            var run = runtime.Run(kernel, Launch, () => buffers.Y.Validate(buffers.Expected, name, 8e-4f, 8e-4f));
            report.Add("normalization", Shape, buffers.ByteLength, run);
        });
    }

    static TileGymBenchmark PersistentLayerNorm(int smCount)
    {
        const string name = "persistent_layer_norm_fwd_kernel";
        var problem = new TileGymPersistentLayerNormProblem(Rows, Columns, smCount);
        var candidates = TileGymPersistentLayerNormCandidates.For(problem);
        var kernel = TileGymKernel.Tuned("persistent_layer_norm.cuh", name,
            "const float*, float*, const float*, const float*, float*, float*", problem, candidates);
        return new(kernel, (runtime, report) =>
        {
            using var buffers = new LayerNormBuffers(runtime);
            void Launch(CUfunction function, TileCppGrid grid) =>
                cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                    buffers.X.Pointer, buffers.Y.Pointer, buffers.W.Pointer, buffers.B.Pointer,
                    buffers.Mean.Pointer, buffers.Rstd.Pointer).Ok();
            var run = runtime.Run(kernel, Launch, () => buffers.Y.Validate(buffers.Expected, name, 8e-4f, 8e-4f));
            report.Add("normalization", Shape, buffers.ByteLength, run);
        });
    }

    static TileGymBenchmark RmsNorm(string name)
    {
        var kernel = TileGymKernel.Fixed("rms_norm.cuh", name, "float, float, 256, 256, 0.00001f, 0.0f",
            "const float*, const float*, float*, float*, int", new TileCppGrid(Rows));
        return new(kernel, (runtime, report) =>
        {
            using var buffers = new RmsNormBuffers(runtime);
            void Launch(CUfunction function, TileCppGrid grid) =>
                cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                    buffers.X.Pointer, buffers.W.Pointer, buffers.Y.Pointer, buffers.Rstd.Pointer, Columns).Ok();
            var run = runtime.Run(kernel, Launch, () => buffers.Y.Validate(buffers.Expected, name, 8e-4f, 8e-4f));
            report.Add("normalization", Shape, buffers.ByteLength, run);
        });
    }

    static TileGymBenchmark RmsNormPv()
    {
        const string name = "rms_norm_kernel_pv";
        var kernel = TileGymKernel.Fixed("rms_norm.cuh", name, "float, float, 8, 256, 256",
            "const float*, const float*, float*, float*, float", new TileCppGrid(Rows), "float,BLOCK_SIZE=256");
        return new(kernel, (runtime, report) =>
        {
            using var buffers = new RmsNormBuffers(runtime);
            void Launch(CUfunction function, TileCppGrid grid)
            {
                var eps = .00001f;
                cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                    buffers.X.Pointer, buffers.W.Pointer, buffers.Y.Pointer, buffers.Rstd.Pointer, eps).Ok();
            }
            var run = runtime.Run(kernel, Launch, () => buffers.Y.Validate(buffers.Expected, name, 8e-4f, 8e-4f));
            report.Add("normalization", Shape, buffers.ByteLength, run);
        });
    }

    static TileGymBenchmark RmsNormPersistent(int smCount)
    {
        const string name = "rms_norm_static_persistent_kernel";
        var problem = new TileGymPersistentRmsNormProblem(Rows, Columns, smCount);
        var candidates = TileGymPersistentRmsNormCandidates.For(problem);
        var kernel = TileGymKernel.Tuned("rms_norm.cuh", name, "const float*, float*, const float*, float*",
            problem, candidates);
        return new(kernel, (runtime, report) =>
        {
            using var buffers = new RmsNormBuffers(runtime);
            void Launch(CUfunction function, TileCppGrid grid) =>
                cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                    buffers.X.Pointer, buffers.Y.Pointer, buffers.W.Pointer, buffers.Rstd.Pointer).Ok();
            var run = runtime.Run(kernel, Launch, () => buffers.Y.Validate(buffers.Expected, name, 8e-4f, 8e-4f));
            report.Add("normalization", Shape, buffers.ByteLength, run);
        });
    }

    static TileGymBenchmark RmsNormBackward()
    {
        const string name = "rms_norm_backward_dx_kernel";
        var kernel = TileGymKernel.Fixed("rms_norm.cuh", name, "float, float, 256",
            "float*, const float*, const float*, const float*, const float*, float*, int, int",
            new TileCppGrid(Rows), "float,BLOCK_SIZE=256");
        return new(kernel, (runtime, report) =>
        {
            using var dx = runtime.Allocate<float>(Rows * Columns);
            using var dy = runtime.Allocate<float>(dx.Length);
            using var x = runtime.Allocate<float>(dx.Length);
            using var w = runtime.Allocate<float>(Columns);
            using var r = runtime.Allocate<float>(Rows);
            using var temp = runtime.Allocate<float>(dx.Length);
            var hx = Values(x.Length);
            var hw = Weights(Columns);
            var hdy = new float[dy.Length];
            Array.Fill(hdy, 1f);
            var hr = Rstd(hx, Rows, Columns);
            x.CopyFrom(hx);
            w.CopyFrom(hw);
            dy.CopyFrom(hdy);
            r.CopyFrom(hr);
            void Launch(CUfunction function, TileCppGrid grid) =>
                cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                    dx.Pointer, dy.Pointer, x.Pointer, w.Pointer, r.Pointer, temp.Pointer, Columns, Columns).Ok();
            var expected = new float[dx.Length];
            for (var row = 0; row < Rows; row++)
            {
                var dot = 0f;
                for (var c = 0; c < Columns; c++)
                {
                    dot += hw[c] * hx[row * Columns + c];
                }
                for (var c = 0; c < Columns; c++)
                {
                    var value = hx[row * Columns + c];
                    expected[row * Columns + c] =
                        hw[c] * hr[row] - value * (hr[row] * hr[row] * hr[row] / Columns) * dot;
                }
            }
            var run = runtime.Run(kernel, Launch, () => dx.Validate(expected, name, 1e-3f, 1e-3f));
            report.Add("normalization", Shape,
                dx.ByteLength + dy.ByteLength + x.ByteLength + w.ByteLength + r.ByteLength + temp.ByteLength, run);
        });
    }

    sealed class LayerNormBuffers : IDisposable
    {
        public LayerNormBuffers(TileGymRuntime runtime)
        {
            X = runtime.Allocate<float>(Rows * Columns);
            Y = runtime.Allocate<float>(X.Length);
            W = runtime.Allocate<float>(Columns);
            B = runtime.Allocate<float>(Columns);
            Mean = runtime.Allocate<float>(Rows);
            Rstd = runtime.Allocate<float>(Rows);
            var hx = Values(X.Length);
            var hw = new float[Columns];
            var hb = new float[Columns];
            for (var i = 0; i < Columns; i++)
            {
                hw[i] = .75f + i / 1024f;
                hb[i] = (i % 17 - 8) / 128f;
            }
            X.CopyFrom(hx);
            W.CopyFrom(hw);
            B.CopyFrom(hb);
            Expected = TileGymNormalizationScenarios.LayerNorm(hx, hw, hb, Rows, Columns, .00001f);
        }

        public CudaBuffer<float> X { get; }
        public CudaBuffer<float> Y { get; }
        public CudaBuffer<float> W { get; }
        public CudaBuffer<float> B { get; }
        public CudaBuffer<float> Mean { get; }
        public CudaBuffer<float> Rstd { get; }
        public float[] Expected { get; }
        public nuint ByteLength => X.ByteLength + Y.ByteLength + W.ByteLength + B.ByteLength;

        public void Dispose()
        {
            X.Dispose();
            Y.Dispose();
            W.Dispose();
            B.Dispose();
            Mean.Dispose();
            Rstd.Dispose();
        }
    }

    sealed class RmsNormBuffers : IDisposable
    {
        public RmsNormBuffers(TileGymRuntime runtime)
        {
            X = runtime.Allocate<float>(Rows * Columns);
            Y = runtime.Allocate<float>(X.Length);
            W = runtime.Allocate<float>(Columns);
            Rstd = runtime.Allocate<float>(Rows);
            var hx = Values(X.Length);
            var hw = Weights(Columns);
            X.CopyFrom(hx);
            W.CopyFrom(hw);
            Expected = RmsNorm(hx, hw, Rows, Columns);
        }

        public CudaBuffer<float> X { get; }
        public CudaBuffer<float> Y { get; }
        public CudaBuffer<float> W { get; }
        public CudaBuffer<float> Rstd { get; }
        public float[] Expected { get; }
        public nuint ByteLength => X.ByteLength + Y.ByteLength + W.ByteLength;

        public void Dispose()
        {
            X.Dispose();
            Y.Dispose();
            W.Dispose();
            Rstd.Dispose();
        }
    }

    static float[] Values(int count)
    {
        var a = new float[count];
        for (var i = 0; i < count; i++)
        {
            a[i] = (i % 251 - 125) / 64f;
        }
        return a;
    }
    static float[] Weights(int n)
    {
        var a = new float[n];
        for (var i = 0; i < n; i++)
        {
            a[i] = .75f + i / 1024f;
        }
        return a;
    }
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
