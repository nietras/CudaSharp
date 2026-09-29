using System.Collections.Generic;
using CudaSharp.TileGym;
using static CudaSharp.nvcuda;

namespace CudaSharp.Tester;

static class TileGymRecurrentScenarios
{
    const int T = 1024, Kd = 16, Vd = 16;
    const float Scale = .25f;
    const string Shape = "B=1,T=1024,H=1,K=16,V=16";

    public static IEnumerable<TileGymBenchmark> Create(TileGymRuntime runtime, TileGymOptions options)
    {
        yield return Dropout();
        yield return Recurrent();
        yield return Chunk();
    }

    static TileGymBenchmark Dropout()
    {
        const int n = 4096;
        const float probability = .25f;
        const uint seed = 2654435761;
        const string name = "seeded_dropout_kernel";
        var problem = new TileGymDropoutProblem(n, probability, seed);
        var candidates = TileGymDropoutCandidates.For();
        var kernel = TileGymKernel.Tuned("dropout.cuh", name, "const float*, float*", problem, candidates);
        return new(kernel, (runtime, report) =>
        {
            using var x = runtime.Allocate<float>(n);
            using var y = runtime.Allocate<float>(n);
            var hx = new float[n];
            Array.Fill(hx, 1f);
            x.CopyFrom(hx);
            void Launch(CUfunction function, TileCppGrid grid) =>
                cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                    x.Pointer, y.Pointer).Ok();
            var expected = new float[n];
            for (var i = 0; i < n; i++)
            {
                var combined = unchecked((int)((uint)i * 1103515245u + seed));
                var hash = combined ^ (combined >> 16);
                hash ^= hash << 8;
                hash ^= hash >> 4;
                var random = (hash & 0x7fffffff) / 2147483647f;
                expected[i] = random > probability ? 1f / (1f - probability) : 0f;
            }
            var run = runtime.Run(kernel, Launch, () => y.Validate(expected, name, 1e-6f, 1e-6f));
            report.Add("dropout", $"{n},p={probability},seed=1", x.ByteLength + y.ByteLength, run);
        });
    }

    static unsafe TileGymBenchmark Recurrent()
    {
        const string name = "recurrent_gated_delta_rule_fwd_kernel";
        var kernel = TileGymKernel.Fixed("recurrent_gated_delta_rule.cuh", name,
            $"float, float, float, {Kd}, {Vd}, false, true, false",
            "const float*, const float*, const float*, const float*, const float*, float*, " +
            "const float*, float*, float, int, int, int, int, int",
            new TileCppGrid(1), "float,OUTPUT_FINAL_STATE=true");
        return new(kernel, (runtime, report) =>
        {
            using var inputs = new Inputs(runtime);
            using var output = runtime.Allocate<float>(T * Vd);
            using var final = runtime.Allocate<float>(Kd * Vd);
            void Launch(CUfunction function, TileCppGrid grid)
            {
                var pq = inputs.Q.Pointer;
                var pk = inputs.K.Pointer;
                var pv = inputs.V.Pointer;
                var pg = inputs.G.Pointer;
                var pb = inputs.Beta.Pointer;
                var po = output.Pointer;
                var init = IntPtr.Zero;
                var pf = final.Pointer;
                var sc = Scale;
                var b = 1;
                var seq = T;
                var h = 1;
                var kd = Kd;
                var vd = Vd;
                var args = stackalloc void*[] { &pq, &pk, &pv, &pg, &pb, &po, &init, &pf, &sc, &b, &seq, &h, &kd, &vd };
                cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream, args, null).Ok();
            }
            void Validate()
            {
                output.Validate(inputs.Expected, name, 2e-3f, 2e-3f);
                final.Validate(inputs.ExpectedState, name, 2e-3f, 2e-3f);
            }
            var run = runtime.Run(kernel, Launch, Validate);
            report.Add("recurrent", Shape, inputs.ByteLength + output.ByteLength + final.ByteLength, run);
        });
    }

    static unsafe TileGymBenchmark Chunk()
    {
        const int chunk = 4, chunks = T / chunk;
        const string interName = "chunk_gated_delta_rule_inter_kernel";
        var label = $"CHUNK={chunk}";
        var intra = TileGymKernel.Fixed("chunk_gated_delta_rule.cuh", "chunk_gated_delta_rule_intra_kernel",
            $"float, float, float, {chunk}, {Kd}, false, 1",
            "const float*, const float*, const float*, const float*, const float*, float*, " +
            "float*, float*, float*, float*, float, int, int, int, int, int, int",
            new TileCppGrid(1, chunks), label);
        var inter = TileGymKernel.Fixed("chunk_gated_delta_rule.cuh", interName,
            $"float, {chunk}, {Kd}, {Vd}, false, true, 1",
            "const float*, const float*, const float*, const float*, const float*, float*, " +
            "const float*, float*, int, int, int, int, int",
            new TileCppGrid(1), label);
        return new([intra, inter], (runtime, report) =>
        {
            using var inputs = new Inputs(runtime);
            using var qo = runtime.Allocate<float>(T * Kd);
            using var ko = runtime.Allocate<float>(T * Kd);
            using var vc = runtime.Allocate<float>(T * Vd);
            using var kc = runtime.Allocate<float>(T * Kd);
            using var gc = runtime.Allocate<float>(T);
            using var output = runtime.Allocate<float>(T * Vd);
            using var final = runtime.Allocate<float>(Kd * Vd);
            void Intra(CUfunction function, TileCppGrid grid)
            {
                var pq = inputs.Q.Pointer;
                var pk = inputs.K.Pointer;
                var pv = inputs.V.Pointer;
                var pb = inputs.Beta.Pointer;
                var pg = inputs.G.Pointer;
                var pqo = qo.Pointer;
                var pko = ko.Pointer;
                var pvc = vc.Pointer;
                var pkc = kc.Pointer;
                var pgc = gc.Pointer;
                var sc = Scale;
                var b = 1;
                var seq = T;
                var h = 1;
                var nc = chunks;
                var kd = Kd;
                var vd = Vd;
                var args = stackalloc void*[]
                {
                    &pq, &pk, &pv, &pb, &pg, &pqo, &pko, &pvc, &pkc, &pgc, &sc, &b, &seq, &h, &nc, &kd, &vd
                };
                cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream, args, null).Ok();
            }
            void Inter(CUfunction function, TileCppGrid grid)
            {
                var pqo = qo.Pointer;
                var pko = ko.Pointer;
                var pvc = vc.Pointer;
                var pkc = kc.Pointer;
                var pgc = gc.Pointer;
                var po = output.Pointer;
                var init = IntPtr.Zero;
                var pf = final.Pointer;
                var b = 1;
                var nc = chunks;
                var h = 1;
                var kd = Kd;
                var vd = Vd;
                var args = stackalloc void*[] { &pqo, &pko, &pvc, &pkc, &pgc, &po, &init, &pf, &b, &nc, &h, &kd, &vd };
                cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream, args, null).Ok();
            }
            var chunked = qo.ByteLength + ko.ByteLength + vc.ByteLength + kc.ByteLength + gc.ByteLength;
            var intraRun = runtime.Run(intra, Intra);
            report.Add("recurrent", Shape,
                inputs.Q.ByteLength + inputs.K.ByteLength + inputs.V.ByteLength + chunked, intraRun);
            void Validate()
            {
                output.Validate(inputs.Expected, interName, 3e-3f, 3e-3f);
                final.Validate(inputs.ExpectedState, interName, 3e-3f, 3e-3f);
            }
            var interRun = runtime.Run(inter, Inter, Validate);
            report.Add("recurrent", Shape, chunked + output.ByteLength + final.ByteLength, interRun);
        });
    }

    sealed class Inputs : IDisposable
    {
        public Inputs(TileGymRuntime runtime)
        {
            Q = runtime.Allocate<float>(T * Kd);
            K = runtime.Allocate<float>(Q.Length);
            V = runtime.Allocate<float>(T * Vd);
            G = runtime.Allocate<float>(T);
            Beta = runtime.Allocate<float>(T);
            var hq = Values(Q.Length, .03f);
            var hk = Values(K.Length, .025f);
            var hv = Values(V.Length, .04f);
            var hg = new float[T];
            var hb = new float[T];
            Array.Fill(hg, -.05f);
            Array.Fill(hb, .6f);
            Q.CopyFrom(hq);
            K.CopyFrom(hk);
            V.CopyFrom(hv);
            G.CopyFrom(hg);
            Beta.CopyFrom(hb);
            Expected = Reference(hq, hk, hv, hg, hb, T, Kd, Vd, Scale, out var state);
            ExpectedState = state;
        }

        public CudaBuffer<float> Q { get; }
        public CudaBuffer<float> K { get; }
        public CudaBuffer<float> V { get; }
        public CudaBuffer<float> G { get; }
        public CudaBuffer<float> Beta { get; }
        public float[] Expected { get; }
        public float[] ExpectedState { get; }
        public nuint ByteLength => Q.ByteLength + K.ByteLength + V.ByteLength + G.ByteLength + Beta.ByteLength;

        public void Dispose()
        {
            Q.Dispose();
            K.Dispose();
            V.Dispose();
            G.Dispose();
            Beta.Dispose();
        }
    }

    static float[] Reference(float[] q, float[] k, float[] v, float[] g, float[] beta, int t, int kd, int vd, float scale, out float[] state)
    {
        state = new float[kd * vd];
        var output = new float[t * vd];
        for (var step = 0; step < t; step++)
        {
            var decay = MathF.Exp(g[step]);
            for (var i = 0; i < state.Length; i++)
            {
                state[i] *= decay;
            }
            for (var col = 0; col < vd; col++)
            {
                var memory = 0f;
                for (var row = 0; row < kd; row++)
                {
                    memory += state[row * vd + col] * k[step * kd + row];
                }
                var delta = (v[step * vd + col] - memory) * beta[step];
                for (var row = 0; row < kd; row++)
                {
                    state[row * vd + col] += k[step * kd + row] * delta;
                }
                var value = 0f;
                for (var row = 0; row < kd; row++)
                {
                    value += state[row * vd + col] * q[step * kd + row] * scale;
                }
                output[step * vd + col] = value;
            }
        }

        return output;
    }

    static float[] Values(int count, float scale)
    {
        var a = new float[count];
        for (var i = 0; i < count; i++)
        {
            a[i] = (i % 23 - 11) * scale;
        }
        return a;
    }
}
