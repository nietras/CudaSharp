using System.Collections.Generic;
using CudaSharp.TileGym;
using static CudaSharp.nvcuda;

namespace CudaSharp.Tester;

static class TileGymMlaScenarios
{
    const float Scale = .1f;
    static readonly TileCppGrid Single = new(1);

    public static IEnumerable<TileGymBenchmark> Create(TileGymRuntime runtime, TileGymOptions options)
    {
        yield return Prefill();
        yield return Decode(transpose: false);
        yield return Decode(transpose: true);
        yield return SplitDecode();
        yield return SplitReduce();
    }

    static TileGymBenchmark Prefill()
    {
        const int s = 64, d = 64, kd = 16;
        const string name = "prefill_mla_kernel";
        var kernel = TileGymKernel.Fixed("mla.cuh", name, $"float, 1, 1, 1, {s}, {s}, {d}, {kd}, 64, 64, 0, true",
            "const float*, const float*, const float*, const float*, const float*, float*, float",
            Single, "TILE_M=64,TILE_N=64");
        return new(kernel, (runtime, report) =>
        {
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
            void Launch(CUfunction function, TileCppGrid grid) =>
                cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                    q.Pointer, qpe.Pointer, k.Pointer, kpe.Pointer, v.Pointer, o.Pointer, Scale).Ok();
            var expected = Mla(hq, hqp, hk, hkp, hv, s, d, kd, Scale, true);
            var run = runtime.Run(kernel, Launch, () => o.Validate(expected, name, 3e-3f, 3e-3f));
            report.Add("mla", $"B=1,H=1,S={s},D={d},KPE={kd}", q.ByteLength + qpe.ByteLength + k.ByteLength +
                kpe.ByteLength + v.ByteLength + o.ByteLength, run);
        });
    }

    static unsafe TileGymBenchmark Decode(bool transpose)
    {
        const int heads = 1, s = 64, d = 64, kd = 16;
        var name = transpose ? "naive_absorb_mla_transpose" : "naive_absorb_mla";
        var kernel = TileGymKernel.Fixed("mla_decoding.cuh", name,
            transpose ? $"float, {d}, 1, 64, {kd}, 0, {s}, true" : $"float, {d}, 1, 64, {kd}",
            "float*, float*, float*, float*, float*, float*, float, long long, int, long long, int, " +
            "long long, int, long long, int, long long, int, int, int" + (transpose ? "" : ", int"),
            Single, "BLOCK_H=1,BLOCK_N=64");
        return new(kernel, (runtime, report) =>
        {
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
            void Launch(CUfunction function, TileCppGrid grid)
            {
                var pq = q.Pointer;
                var pqp = qpe.Pointer;
                var pkv = kv.Pointer;
                var pkp = kpe.Pointer;
                var po = o.Pointer;
                var pl = l.Pointer;
                var sc = Scale;
                long qbs = heads * d, qpbs = heads * kd, kvbs = s * d, kpbs = s * kd, obs = heads * d;
                var qhs = d;
                var qphs = kd;
                var kvs = d;
                var kps = kd;
                var os = d;
                var b = 1;
                var h = heads;
                var seq = s;
                var args = stackalloc void*[]
                {
                    &pq, &pqp, &pkv, &pkp, &po, &pl, &sc, &qbs, &qhs, &qpbs, &qphs, &kvbs,
                    &kvs, &kpbs, &kps, &obs, &os, &b, &h, &seq
                };
                cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream, args, null).Ok();
            }
            var expected = Mla(hq, hqp, hkv, hkp, hkv, 1, d, kd, Scale, false);
            var run = runtime.Run(kernel, Launch, () => o.Validate(expected, name, 3e-3f, 3e-3f));
            report.Add("mla", $"B=1,H={heads},S={s},D={d},KPE={kd}", q.ByteLength + qpe.ByteLength +
                kv.ByteLength + kpe.ByteLength + o.ByteLength + l.ByteLength, run);
        });
    }

    static TileGymBenchmark SplitDecode()
    {
        const int s = 128, d = 64, kd = 16;
        const string name = "naive_absorb_mla_transpose";
        var kernel = TileGymKernel.Fixed("mla_decoding_split_kv.cuh", name,
            $"float, 1, 1, {s}, {d}, 16, 128, {kd}, 1, 128, true",
            "const float*, const float*, const float*, const float*, const float*, float*, float*, float",
            Single, "TILE_H=16,TILE_N=128,SPLITS=1");
        return new(kernel, (runtime, report) =>
        {
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
            void Launch(CUfunction function, TileCppGrid grid) =>
                cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                    q.Pointer, qpe.Pointer, kv.Pointer, kv.Pointer, kpe.Pointer, o.Pointer, l.Pointer, Scale).Ok();
            var expected = Mla(hq, hqp, hkv, hkp, hkv, 1, d, kd, Scale, false);
            var run = runtime.Run(kernel, Launch, () => o.Validate(expected, name, 3e-3f, 3e-3f));
            report.Add("mla-split", $"B=1,H=1,S={s},D={d},KPE={kd}", q.ByteLength + qpe.ByteLength +
                kv.ByteLength + kpe.ByteLength + o.ByteLength + l.ByteLength, run);
        });
    }

    static TileGymBenchmark SplitReduce()
    {
        const int d = 64, splits = 2;
        const string name = "splitk_reduce_kernel";
        var kernel = TileGymKernel.Fixed("splitk_reduce.cuh", name, $"float, 1, 1, {d}, {splits}, {splits}, {d}, false",
            "const float*, const float*, float*", Single, "BLOCK_D=64,USE_DOT=false");
        return new(kernel, (runtime, report) =>
        {
            using var input = runtime.Allocate<float>(splits * d);
            using var lse = runtime.Allocate<float>(splits);
            using var output = runtime.Allocate<float>(d);
            var hi = Values(input.Length, .1f);
            var hl = new[] { -.3f, .2f };
            input.CopyFrom(hi);
            lse.CopyFrom(hl);
            void Launch(CUfunction function, TileCppGrid grid) =>
                cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                    input.Pointer, lse.Pointer, output.Pointer).Ok();
            var expected = new float[d];
            var max = Math.Max(hl[0], hl[1]);
            var a = MathF.Pow(2, hl[0] - max);
            var b = MathF.Pow(2, hl[1] - max);
            for (var i = 0; i < d; i++)
            {
                expected[i] = (a * hi[i] + b * hi[d + i]) / (a + b);
            }
            var run = runtime.Run(kernel, Launch, () => output.Validate(expected, name, 5e-4f, 5e-4f));
            report.Add("reduction", $"B=1,H=1,SPLITS={splits},D={d}",
                input.ByteLength + lse.ByteLength + output.ByteLength, run);
        });
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
                {
                    dot += q[i * d + x] * k[j * d + x];
                }

                for (var x = 0; x < kd; x++)
                {
                    dot += qpe[i * kd + x] * kpe[j * kd + x];
                }

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
                {
                    o[i * d + x] += p * v[j * d + x];
                }
            }
        }

        return o;
    }

    static float[] Values(int count, float scale)
    {
        var a = new float[count];

        for (var i = 0; i < count; i++)
        {
            a[i] = (i % 29 - 14) * scale;
        }

        return a;
    }
}
