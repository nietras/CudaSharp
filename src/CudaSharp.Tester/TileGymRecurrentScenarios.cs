using System;
using CudaSharp.Tile;
using static CudaSharp.nvcuda;

namespace CudaSharp.Tester;

static class TileGymRecurrentScenarios
{
    static readonly TileCppConfig Config = new([]);

    public static void RunAll(TileGymRuntime runtime, TileGymReport report)
    {
        RunDropout(runtime, report);
        RunRecurrent(runtime, report);
        RunChunk(runtime, report);
    }

    static unsafe void RunDropout(TileGymRuntime runtime, TileGymReport report)
    {
        const int n = 4096;
        const float probability = .25f;
        const ulong seed = 2654435761;
        const string name = "seeded_dropout_kernel";

        using var kernel = TileGymKernel.Create(runtime, "dropout.cuh", name, "float, 1024", "const float*, float*, float, uint64_t, int");
        using var x = runtime.Allocate<float>(n);
        using var y = runtime.Allocate<float>(n);

        var hx = new float[n];
        Array.Fill(hx, 1f);
        x.CopyFrom(hx);

        void Launch()
        {
            var px = x.Pointer.Value;
            var py = y.Pointer.Value;
            var p = probability;
            var s = seed;
            var count = n;

            var args = stackalloc IntPtr[]
            {
                (IntPtr)(&px),
                (IntPtr)(&py),
                (IntPtr)(&p),
                (IntPtr)(&s),
                (IntPtr)(&count)
            };

            kernel.Launch(Config, new(n / 1024), runtime.Stream, new(args, 5));
        }

        var timing = TileGymKernel.Measure(runtime, Launch);
        var expected = new float[n];
        for (var i = 0; i < n; i++)
        {
            var combined = unchecked((uint)i * 1103515245u + (uint)seed);
            var hash = combined ^ (combined >> 16);
            hash ^= hash << 8;
            hash ^= hash >> 4;
            var random = (hash & 0x7fffffffu) / 2147483647f;
            expected[i] = random > probability ? 1f / (1f - probability) : 0f;
        }

        TileGymKernel.Validate(y.CopyToHost(), expected, name, 1e-6f, 1e-6f);
        TileGymKernel.Report(report, "dropout", name, $"{n},p={probability},seed=1", "float,BLOCK_SIZE=1024", x.ByteLength + y.ByteLength, timing);
    }

    static unsafe void RunRecurrent(TileGymRuntime runtime, TileGymReport report)
    {
        const int t = 8, kd = 16, vd = 16;
        const float scale = .25f;
        const string name = "recurrent_gated_delta_rule_fwd_kernel";

        using var kernel = TileGymKernel.Create(
            runtime, "recurrent_gated_delta_rule.cuh", name,
            $"float, float, float, {kd}, {vd}, false, true, false",
            "const float*, const float*, const float*, const float*, const float*, float*, " +
            "const float*, float*, float, int, int, int, int, int");
        using var q = runtime.Allocate<float>(t * kd);
        using var k = runtime.Allocate<float>(q.Length);
        using var v = runtime.Allocate<float>(t * vd);
        using var g = runtime.Allocate<float>(t);
        using var beta = runtime.Allocate<float>(t);
        using var output = runtime.Allocate<float>(t * vd);
        using var final = runtime.Allocate<float>(kd * vd);

        var hq = Values(q.Length, .03f);
        var hk = Values(k.Length, .025f);
        var hv = Values(v.Length, .04f);
        var hg = new float[t];
        var hb = new float[t];
        for (var i = 0; i < t; i++)
        {
            hg[i] = -.05f;
            hb[i] = .6f;
        }

        q.CopyFrom(hq);
        k.CopyFrom(hk);
        v.CopyFrom(hv);
        g.CopyFrom(hg);
        beta.CopyFrom(hb);

        void Launch()
        {
            var pq = q.Pointer.Value;
            var pk = k.Pointer.Value;
            var pv = v.Pointer.Value;
            var pg = g.Pointer.Value;
            var pb = beta.Pointer.Value;
            var po = output.Pointer.Value;
            var init = IntPtr.Zero;
            var pf = final.Pointer.Value;
            var sc = scale;
            var b = 1;
            var seq = t;
            var h = 1;
            var kd0 = kd;
            var vd0 = vd;

            var args = stackalloc IntPtr[]
            {
                (IntPtr)(&pq),
                (IntPtr)(&pk),
                (IntPtr)(&pv),
                (IntPtr)(&pg),
                (IntPtr)(&pb),
                (IntPtr)(&po),
                (IntPtr)(&init),
                (IntPtr)(&pf),
                (IntPtr)(&sc),
                (IntPtr)(&b),
                (IntPtr)(&seq),
                (IntPtr)(&h),
                (IntPtr)(&kd0),
                (IntPtr)(&vd0)
            };

            kernel.Launch(Config, new(1, 1), runtime.Stream, new(args, 14));
        }

        var timing = TileGymKernel.Measure(runtime, Launch);
        var expected = Reference(hq, hk, hv, hg, hb, t, kd, vd, scale, out var state);

        TileGymKernel.Validate(output.CopyToHost(), expected, name, 2e-3f, 2e-3f);
        TileGymKernel.Validate(final.CopyToHost(), state, name, 2e-3f, 2e-3f);
        TileGymKernel.Report(report, "recurrent", name, $"B=1,T={t},H=1,K={kd},V={vd}", "float,OUTPUT_FINAL_STATE=true", q.ByteLength + k.ByteLength + v.ByteLength + g.ByteLength + beta.ByteLength + output.ByteLength + final.ByteLength, timing);
    }

    static unsafe void RunChunk(TileGymRuntime runtime, TileGymReport report)
    {
        const int t = 8, chunk = 4, chunks = 2, kd = 16, vd = 16;
        const float scale = .25f;

        using var intra = TileGymKernel.Create(
            runtime, "chunk_gated_delta_rule.cuh", "chunk_gated_delta_rule_intra_kernel",
            $"float, float, float, {chunk}, {kd}, false, 1",
            "const float*, const float*, const float*, const float*, const float*, float*, " +
            "float*, float*, float*, float*, float, int, int, int, int, int, int");
        using var inter = TileGymKernel.Create(
            runtime, "chunk_gated_delta_rule.cuh", "chunk_gated_delta_rule_inter_kernel",
            $"float, {chunk}, {kd}, {vd}, false, true, 1",
            "const float*, const float*, const float*, const float*, const float*, float*, " +
            "const float*, float*, int, int, int, int, int");
        using var q = runtime.Allocate<float>(t * kd);
        using var k = runtime.Allocate<float>(q.Length);
        using var v = runtime.Allocate<float>(t * vd);
        using var beta = runtime.Allocate<float>(t);
        using var g = runtime.Allocate<float>(t);
        using var qo = runtime.Allocate<float>(t * kd);
        using var ko = runtime.Allocate<float>(t * kd);
        using var vc = runtime.Allocate<float>(t * vd);
        using var kc = runtime.Allocate<float>(t * kd);
        using var gc = runtime.Allocate<float>(t);
        using var output = runtime.Allocate<float>(t * vd);
        using var final = runtime.Allocate<float>(kd * vd);

        var hq = Values(q.Length, .03f);
        var hk = Values(k.Length, .025f);
        var hv = Values(v.Length, .04f);
        var hg = new float[t];
        var hb = new float[t];
        for (var i = 0; i < t; i++)
        {
            hg[i] = -.05f;
            hb[i] = .6f;
        }

        q.CopyFrom(hq);
        k.CopyFrom(hk);
        v.CopyFrom(hv);
        g.CopyFrom(hg);
        beta.CopyFrom(hb);

        void Intra()
        {
            var pq = q.Pointer.Value;
            var pk = k.Pointer.Value;
            var pv = v.Pointer.Value;
            var pb = beta.Pointer.Value;
            var pg = g.Pointer.Value;
            var pqo = qo.Pointer.Value;
            var pko = ko.Pointer.Value;
            var pvc = vc.Pointer.Value;
            var pkc = kc.Pointer.Value;
            var pgc = gc.Pointer.Value;
            var sc = scale;
            var b = 1;
            var seq = t;
            var h = 1;
            var nc = chunks;
            var kd0 = kd;
            var vd0 = vd;

            var args = stackalloc IntPtr[]
            {
                (IntPtr)(&pq),
                (IntPtr)(&pk),
                (IntPtr)(&pv),
                (IntPtr)(&pb),
                (IntPtr)(&pg),
                (IntPtr)(&pqo),
                (IntPtr)(&pko),
                (IntPtr)(&pvc),
                (IntPtr)(&pkc),
                (IntPtr)(&pgc),
                (IntPtr)(&sc),
                (IntPtr)(&b),
                (IntPtr)(&seq),
                (IntPtr)(&h),
                (IntPtr)(&nc),
                (IntPtr)(&kd0),
                (IntPtr)(&vd0)
            };

            intra.Launch(Config, new(1, chunks), runtime.Stream, new(args, 17));
        }

        void Inter()
        {
            var pqo = qo.Pointer.Value;
            var pko = ko.Pointer.Value;
            var pvc = vc.Pointer.Value;
            var pkc = kc.Pointer.Value;
            var pgc = gc.Pointer.Value;
            var po = output.Pointer.Value;
            var init = IntPtr.Zero;
            var pf = final.Pointer.Value;
            var b = 1;
            var nc = chunks;
            var h = 1;
            var kd0 = kd;
            var vd0 = vd;

            var args = stackalloc IntPtr[]
            {
                (IntPtr)(&pqo),
                (IntPtr)(&pko),
                (IntPtr)(&pvc),
                (IntPtr)(&pkc),
                (IntPtr)(&pgc),
                (IntPtr)(&po),
                (IntPtr)(&init),
                (IntPtr)(&pf),
                (IntPtr)(&b),
                (IntPtr)(&nc),
                (IntPtr)(&h),
                (IntPtr)(&kd0),
                (IntPtr)(&vd0)
            };

            inter.Launch(Config, new(1, 1), runtime.Stream, new(args, 13));
        }

        var intraTiming = TileGymKernel.Measure(runtime, Intra);
        var interTiming = TileGymKernel.Measure(runtime, Inter);
        var expected = Reference(hq, hk, hv, hg, hb, t, kd, vd, scale, out var state);

        TileGymKernel.Validate(output.CopyToHost(), expected, "chunk_gated_delta_rule_inter_kernel", 3e-3f, 3e-3f);
        TileGymKernel.Validate(final.CopyToHost(), state, "chunk_gated_delta_rule_inter_kernel", 3e-3f, 3e-3f);
        TileGymKernel.Report(report, "recurrent", "chunk_gated_delta_rule_intra_kernel", $"B=1,T={t},H=1,K={kd},V={vd}", $"CHUNK={chunk}", q.ByteLength + k.ByteLength + v.ByteLength + qo.ByteLength + ko.ByteLength + vc.ByteLength + kc.ByteLength + gc.ByteLength, intraTiming);
        TileGymKernel.Report(report, "recurrent", "chunk_gated_delta_rule_inter_kernel", $"B=1,T={t},H=1,K={kd},V={vd}", $"CHUNK={chunk}", qo.ByteLength + ko.ByteLength + vc.ByteLength + kc.ByteLength + gc.ByteLength + output.ByteLength + final.ByteLength, interTiming);
    }

    static float[] Reference(float[] q, float[] k, float[] v, float[] g, float[] beta, int t, int kd, int vd, float scale, out float[] state)
    {
        state = new float[kd * vd];
        var output = new float[t * vd];
        for (var step = 0; step < t; step++)
        {
            var decay = MathF.Exp(g[step]);
            for (var i = 0; i < state.Length; i++) state[i] *= decay;
            for (var col = 0; col < vd; col++)
            {
                var memory = 0f;
                for (var row = 0; row < kd; row++) memory += state[row * vd + col] * k[step * kd + row];
                var delta = (v[step * vd + col] - memory) * beta[step];
                for (var row = 0; row < kd; row++) state[row * vd + col] += k[step * kd + row] * delta;
                var value = 0f;
                for (var row = 0; row < kd; row++) value += state[row * vd + col] * q[step * kd + row] * scale;
                output[step * vd + col] = value;
            }
        }

        return output;
    }

    static float[] Values(int count, float scale)
    {
        var a = new float[count];
        for (var i = 0; i < count; i++) a[i] = (i % 23 - 11) * scale;
        return a;
    }
}
