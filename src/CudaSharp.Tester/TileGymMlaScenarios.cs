using System;
using CudaSharp.Tile;

namespace CudaSharp.Tester;

static class TileGymMlaScenarios
{
    static readonly TileCppConfig Config = new([]);

    public static void RunAll(TileGymRuntime runtime, TileGymReport report)
    {
        RunPrefill(runtime, report);
        RunDecode(runtime, report, false);
        RunDecode(runtime, report, true);
        RunSplitDecode(runtime, report);
        RunSplitReduce(runtime, report);
    }

    static unsafe void RunPrefill(TileGymRuntime runtime, TileGymReport report)
    {
        const int s = 64, d = 64, kd = 16;
        const float scale = .1f;
        const string name = "prefill_mla_kernel";

        using var kernel = TileGymKernel.Create(runtime, "mla.cuh", name, $"float, 1, 1, 1, {s}, {s}, {d}, {kd}, 64, 64, 0, true", "const float*, const float*, const float*, const float*, const float*, float*, float");
        using var q = runtime.Allocate<float>(s * d);
        using var qpe = runtime.Allocate<float>(s * kd);
        using var k = runtime.Allocate<float>(s * d);
        using var kpe = runtime.Allocate<float>(s * kd);
        using var v = runtime.Allocate<float>(s * d);
        using var o = runtime.Allocate<float>(s * d);

        var hq = Values(q.Length, .03f);
        var hqp = Values(qpe.Length, .02f);
        var hk = Values(k.Length, .025f);
        var hkp = Values(kpe.Length, .015f);
        var hv = Values(v.Length, .04f);

        q.CopyFrom(hq);
        qpe.CopyFrom(hqp);
        k.CopyFrom(hk);
        kpe.CopyFrom(hkp);
        v.CopyFrom(hv);

        void Launch()
        {
            var pq = q.Pointer.Value;
            var pqp = qpe.Pointer.Value;
            var pk = k.Pointer.Value;
            var pkp = kpe.Pointer.Value;
            var pv = v.Pointer.Value;
            var po = o.Pointer.Value;
            var sc = scale;

            var args = stackalloc IntPtr[] { (IntPtr)(&pq), (IntPtr)(&pqp), (IntPtr)(&pk), (IntPtr)(&pkp), (IntPtr)(&pv), (IntPtr)(&po), (IntPtr)(&sc) };

            kernel.Launch(Config, new(1, 1), runtime.Stream, new(args, 7));
        }

        var timing = TileGymKernel.Measure(runtime, Launch);

        TileGymKernel.Validate(o.CopyToHost(), Mla(hq, hqp, hk, hkp, hv, s, d, kd, scale, true), name, 3e-3f, 3e-3f);
        TileGymKernel.Report(report, "mla", name, $"B=1,H=1,S={s},D={d},KPE={kd}", "TILE_M=64,TILE_N=64", q.ByteLength + qpe.ByteLength + k.ByteLength + kpe.ByteLength + v.ByteLength + o.ByteLength, timing);
    }

    static unsafe void RunDecode(TileGymRuntime runtime, TileGymReport report, bool transpose)
    {
        const int heads = 1, s = 64, d = 64, kd = 16;
        const float scale = .1f;
        var name = transpose ? "naive_absorb_mla_transpose" : "naive_absorb_mla";

        using var kernel = TileGymKernel.Create(
            runtime, "mla_decoding.cuh", name, $"float, {d}, 1, 64, {kd}",
            "float*, float*, float*, float*, float*, float*, float, long long, int, long long, int, " +
            "long long, int, long long, int, long long, int, int, int, int");
        using var q = runtime.Allocate<float>(heads * d);
        using var qpe = runtime.Allocate<float>(heads * kd);
        using var kv = runtime.Allocate<float>(s * d);
        using var kpe = runtime.Allocate<float>(s * kd);
        using var o = runtime.Allocate<float>(heads * d);
        using var l = runtime.Allocate<float>(heads);

        var hq = Values(q.Length, .03f);
        var hqp = Values(qpe.Length, .02f);
        var hkv = Values(kv.Length, .025f);
        var hkp = Values(kpe.Length, .015f);

        q.CopyFrom(hq);
        qpe.CopyFrom(hqp);
        kv.CopyFrom(hkv);
        kpe.CopyFrom(hkp);

        void Launch()
        {
            var pq = q.Pointer.Value;
            var pqp = qpe.Pointer.Value;
            var pkv = kv.Pointer.Value;
            var pkp = kpe.Pointer.Value;
            var po = o.Pointer.Value;
            var pl = l.Pointer.Value;
            var sc = scale;

            long qbs = heads * d, qpbs = heads * kd, kvbs = s * d, kpbs = s * kd, obs = heads * d;
            var qhs = d;
            var qphs = kd;
            var kvs = d;
            var kps = kd;
            var os = d;
            var b = 1;
            var h = heads;
            var seq = s;

            var args = stackalloc IntPtr[]
            {
                (IntPtr)(&pq), (IntPtr)(&pqp), (IntPtr)(&pkv), (IntPtr)(&pkp),
                (IntPtr)(&po), (IntPtr)(&pl), (IntPtr)(&sc), (IntPtr)(&qbs),
                (IntPtr)(&qhs), (IntPtr)(&qpbs), (IntPtr)(&qphs), (IntPtr)(&kvbs),
                (IntPtr)(&kvs), (IntPtr)(&kpbs), (IntPtr)(&kps), (IntPtr)(&obs),
                (IntPtr)(&os), (IntPtr)(&b), (IntPtr)(&h), (IntPtr)(&seq)
            };

            kernel.Launch(Config, new(1, 1), runtime.Stream, new(args, 20));
        }

        var timing = TileGymKernel.Measure(runtime, Launch);

        TileGymKernel.Validate(o.CopyToHost(), Mla(hq, hqp, hkv, hkp, hkv, 1, d, kd, scale, false), name, 3e-3f, 3e-3f);
        TileGymKernel.Report(report, "mla", name, $"B=1,H={heads},S={s},D={d},KPE={kd}", "BLOCK_H=1,BLOCK_N=64", q.ByteLength + qpe.ByteLength + kv.ByteLength + kpe.ByteLength + o.ByteLength + l.ByteLength, timing);
    }

    static unsafe void RunSplitDecode(TileGymRuntime runtime, TileGymReport report)
    {
        const int s = 128, d = 64, kd = 16;
        const float scale = .1f;
        const string name = "naive_absorb_mla_transpose";

        using var kernel = TileGymKernel.Create(
            runtime, "mla_decoding_split_kv.cuh", name,
            $"float, 1, 1, {s}, {d}, 16, 128, {kd}, 1, 128, true",
            "const float*, const float*, const float*, const float*, const float*, float*, float*, float");
        using var q = runtime.Allocate<float>(d);
        using var qpe = runtime.Allocate<float>(kd);
        using var kv = runtime.Allocate<float>(s * d);
        using var kpe = runtime.Allocate<float>(s * kd);
        using var o = runtime.Allocate<float>(d);
        using var l = runtime.Allocate<float>(1);

        var hq = Values(d, .03f);
        var hqp = Values(kd, .02f);
        var hkv = Values(kv.Length, .025f);
        var hkp = Values(kpe.Length, .015f);

        q.CopyFrom(hq);
        qpe.CopyFrom(hqp);
        kv.CopyFrom(hkv);
        kpe.CopyFrom(hkp);

        void Launch()
        {
            var pq = q.Pointer.Value;
            var pqp = qpe.Pointer.Value;
            var pk = kv.Pointer.Value;
            var pv = kv.Pointer.Value;
            var pkp = kpe.Pointer.Value;
            var po = o.Pointer.Value;
            var pl = l.Pointer.Value;
            var sc = scale;

            var args = stackalloc IntPtr[]
            {
                (IntPtr)(&pq), (IntPtr)(&pqp), (IntPtr)(&pk), (IntPtr)(&pv),
                (IntPtr)(&pkp), (IntPtr)(&po), (IntPtr)(&pl), (IntPtr)(&sc)
            };

            kernel.Launch(Config, new(1, 1, 1), runtime.Stream, new(args, 8));
        }

        var timing = TileGymKernel.Measure(runtime, Launch);

        TileGymKernel.Validate(o.CopyToHost(), Mla(hq, hqp, hkv, hkp, hkv, 1, d, kd, scale, false), name, 3e-3f, 3e-3f);
        TileGymKernel.Report(report, "mla-split", name, $"B=1,H=1,S={s},D={d},KPE={kd}", "TILE_H=16,TILE_N=128,SPLITS=1", q.ByteLength + qpe.ByteLength + kv.ByteLength + kpe.ByteLength + o.ByteLength + l.ByteLength, timing);
    }

    static unsafe void RunSplitReduce(TileGymRuntime runtime, TileGymReport report)
    {
        const int d = 64, splits = 2;
        const string name = "splitk_reduce_kernel";

        using var kernel = TileGymKernel.Create(
            runtime, "splitk_reduce.cuh", name,
            $"float, 1, 1, {d}, {splits}, {splits}, {d}, false",
            "const float*, const float*, float*");
        using var input = runtime.Allocate<float>(splits * d);
        using var lse = runtime.Allocate<float>(splits);
        using var output = runtime.Allocate<float>(d);

        var hi = Values(input.Length, .1f);
        var hl = new[] { -.3f, .2f };

        input.CopyFrom(hi);
        lse.CopyFrom(hl);

        void Launch()
        {
            var pi = input.Pointer.Value;
            var pl = lse.Pointer.Value;
            var po = output.Pointer.Value;

            var args = stackalloc IntPtr[] { (IntPtr)(&pi), (IntPtr)(&pl), (IntPtr)(&po) };

            kernel.Launch(Config, new(1, 1, 1), runtime.Stream, new(args, 3));
        }

        var timing = TileGymKernel.Measure(runtime, Launch);
        var expected = new float[d];
        var max = Math.Max(hl[0], hl[1]);
        var a = MathF.Pow(2, hl[0] - max);
        var b = MathF.Pow(2, hl[1] - max);

        for (var i = 0; i < d; i++)
            expected[i] = (a * hi[i] + b * hi[d + i]) / (a + b);

        TileGymKernel.Validate(output.CopyToHost(), expected, name, 5e-4f, 5e-4f);
        TileGymKernel.Report(report, "reduction", name, $"B=1,H=1,SPLITS={splits},D={d}", "BLOCK_D=64,USE_DOT=false", input.ByteLength + lse.ByteLength + output.ByteLength, timing);
    }

    static float[] Mla(float[] q, float[] qpe, float[] k, float[] kpe, float[] v, int sq, int d, int kd, float scale, bool causal)
    {
        var sk = k.Length / d;
        var o = new float[sq * d];
        var scores = new float[sk];

        for (var i = 0; i < sq; i++)
        {
            var max = float.NegativeInfinity;

            for (var j = 0; j < sk; j++)
            {
                if (causal && j > i)
                {
                    scores[j] = float.NegativeInfinity;
                    continue;
                }

                var dot = 0f;

                for (var x = 0; x < d; x++)
                    dot += q[i * d + x] * k[j * d + x];

                for (var x = 0; x < kd; x++)
                    dot += qpe[i * kd + x] * kpe[j * kd + x];

                scores[j] = dot * scale;
                max = Math.Max(max, scores[j]);
            }

            var sum = 0f;

            for (var j = 0; j < sk; j++)
            {
                scores[j] = MathF.Exp(scores[j] - max);
                sum += scores[j];
            }

            for (var j = 0; j < sk; j++)
            {
                var p = scores[j] / sum;

                for (var x = 0; x < d; x++)
                    o[i * d + x] += p * v[j * d + x];
            }
        }

        return o;
    }

    static float[] Values(int count, float scale)
    {
        var a = new float[count];

        for (var i = 0; i < count; i++)
            a[i] = (i % 29 - 14) * scale;

        return a;
    }
}
