using System.Collections.Generic;
using CudaSharp.TileGym;
using static CudaSharp.nvcuda;

namespace CudaSharp.Tester;

static class TileGymAttentionScenarios
{
    const int Sequence = 64;
    const int Dimension = 64;
    const float Scale = .125f;
    const string ForwardSignature = "const float*, const float*, const float*, float*, float*, float";
    const string PreprocessSignature = "const float*, const float*, const float*, float*, float*, float";

    public static IEnumerable<TileGymBenchmark> Create(TileGymRuntime runtime, TileGymOptions options)
    {
        yield return Prefill(runtime, "attention.cuh", "prefill_fmha_fwd_kernel", ForwardSignature, gemma: false);
        yield return BackwardPreprocess(runtime);
        yield return BackwardMain(runtime);
        yield return Sink();
        yield return Prefill(runtime, "gemma_attention.cuh", "gemma_attention_fwd_kernel",
            "const float*, const float*, const float*, float*, float, float", gemma: true);
        yield return Decode("flash_decode.cuh", "attention_decode_kernel_optimized",
            "float, 1, 1, 64, 1, 8, 64, 64, 64, 1", sink: false, gemma: false);
        yield return Decode("attention_sink_decode.cuh", "attention_sink_decode_kernel",
            "float, 1, 1, 1, 64, 1, 8, 64, 64, 64, 1, 0, false, 1", sink: true, gemma: false);
        yield return Decode("gemma_attention_decode.cuh", "gemma_attention_decode_kernel",
            "float, 1, 1, 64, 1, 8, 64, 64, 64, 1, 0, false, 1", sink: false, gemma: true);
    }

    static TileGymBenchmark Prefill(TileGymRuntime runtime, string header, string name, string signature, bool gemma)
    {
        var kind = gemma ? TileGymAttentionKind.GemmaForward : TileGymAttentionKind.Forward;
        var problem = new TileGymAttentionProblem(kind, Sequence, Dimension, true, runtime.Architecture);
        var candidates = TileGymAttentionCandidates.For(problem);
        var kernel = TileGymKernel.Tuned(header, name, signature, problem, candidates);
        return new(kernel, (runtime, report) =>
        {
            using var q = runtime.Allocate<float>(Sequence * Dimension);
            using var k = runtime.Allocate<float>(q.Length);
            using var v = runtime.Allocate<float>(q.Length);
            using var output = runtime.Allocate<float>(q.Length);
            using var lse = gemma ? null : runtime.Allocate<float>(Sequence);
            var hq = Values(q.Length, .07f);
            var hk = Values(k.Length, .05f);
            var hv = Values(v.Length, .11f);
            q.CopyFrom(hq);
            k.CopyFrom(hk);
            v.CopyFrom(hv);
            void Launch(CUfunction function, TileCppGrid grid)
            {
                if (lse is null)
                {
                    var softCap = 0f;
                    cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                        q.Pointer, k.Pointer, v.Pointer, output.Pointer, Scale, softCap).Ok();
                }
                else
                {
                    cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                        q.Pointer, k.Pointer, v.Pointer, output.Pointer, lse.Pointer, Scale).Ok();
                }
            }
            var expected = Attention(hq, hk, hv, Sequence, Sequence, Dimension, Scale, true);
            var run = runtime.Run(kernel, Launch, () => output.Validate(expected, name, 2e-3f, 2e-3f));
            report.Add("attention", $"B=1,H=1,Sq={Sequence},Sk={Sequence},D={Dimension}",
                q.ByteLength + k.ByteLength + v.ByteLength + output.ByteLength + (lse?.ByteLength ?? 0), run);
        });
    }

    static unsafe TileGymBenchmark Sink()
    {
        const string name = "attention_sink_fwd_kernel";
        var kernel = TileGymKernel.Fixed("attention_sink.cuh", name, "float, 64, 64, 64, false",
            "float*, float*, float*, float*, float, float*, float*, int, int, int, int, int, int",
            new TileCppGrid(1), "float,BLOCK_M=64,BLOCK_N=64");
        return new(kernel, (runtime, report) =>
        {
            using var q = runtime.Allocate<float>(Sequence * Dimension);
            using var k = runtime.Allocate<float>(q.Length);
            using var v = runtime.Allocate<float>(q.Length);
            using var sinks = runtime.Allocate<float>(1);
            using var m = runtime.Allocate<float>(Sequence);
            using var output = runtime.Allocate<float>(q.Length);
            var hq = Values(q.Length, .07f);
            var hk = Values(k.Length, .05f);
            var hv = Values(v.Length, .11f);
            q.CopyFrom(hq);
            k.CopyFrom(hk);
            v.CopyFrom(hv);
            sinks.CopyFrom([float.NegativeInfinity]);
            void Launch(CUfunction function, TileCppGrid grid)
            {
                var pq = q.Pointer;
                var pk = k.Pointer;
                var pv = v.Pointer;
                var ps = sinks.Pointer;
                var pm = m.Pointer;
                var po = output.Pointer;
                var scale = Scale;
                var start = 0;
                var z = 1;
                var h = 1;
                var nq = Sequence;
                var nk = Sequence;
                var bandwidth = 0;
                var args = stackalloc void*[]
                {
                    &pq, &pk, &pv, &ps, &scale, &pm, &po, &start, &z, &h, &nq, &nk, &bandwidth
                };
                cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream, args, null).Ok();
            }
            var expected = Attention(hq, hk, hv, Sequence, Sequence, Dimension, Scale, true);
            var run = runtime.Run(kernel, Launch, () => output.Validate(expected, name, 2e-3f, 2e-3f));
            report.Add("attention", $"B=1,H=1,S={Sequence},D={Dimension}",
                q.ByteLength + k.ByteLength + v.ByteLength + output.ByteLength, run);
        });
    }

    static TileGymBenchmark Decode(string header, string name, string templates, bool sink, bool gemma)
    {
        var signature = sink
            ? "const float*, const float*, const float*, const float*, float*, float*, const int*, float"
            : gemma
                ? "const float*, const float*, const float*, float*, float*, float, float"
                : "const float*, const float*, const float*, float*, float*, float";
        var kernel = TileGymKernel.Fixed(header, name, templates, signature, new TileCppGrid(1));
        return new(kernel, (runtime, report) =>
        {
            using var q = runtime.Allocate<float>(Dimension);
            using var k = runtime.Allocate<float>(Sequence * Dimension);
            using var v = runtime.Allocate<float>(k.Length);
            using var output = runtime.Allocate<float>(Dimension);
            using var lse = runtime.Allocate<float>(1);
            using var start = runtime.Allocate<int>(1);
            using var sinks = runtime.Allocate<float>(1);
            var hq = Values(q.Length, .07f);
            var hk = Values(k.Length, .05f);
            var hv = Values(v.Length, .11f);
            q.CopyFrom(hq);
            k.CopyFrom(hk);
            v.CopyFrom(hv);
            start.CopyFrom([Sequence - 1]);
            sinks.CopyFrom([float.NegativeInfinity]);
            void Launch(CUfunction function, TileCppGrid grid)
            {
                if (sink)
                {
                    cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                        q.Pointer, k.Pointer, v.Pointer, sinks.Pointer, output.Pointer, lse.Pointer, start.Pointer,
                        Scale).Ok();
                }
                else if (gemma)
                {
                    var softCap = 0f;
                    cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                        q.Pointer, k.Pointer, v.Pointer, output.Pointer, lse.Pointer, Scale, softCap).Ok();
                }
                else
                {
                    cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                        q.Pointer, k.Pointer, v.Pointer, output.Pointer, lse.Pointer, Scale).Ok();
                }
            }
            var expected = Attention(hq, hk, hv, 1, Sequence, Dimension, Scale, false);
            var run = runtime.Run(kernel, Launch, () => output.Validate(expected, name, 2e-3f, 2e-3f));
            report.Add("decode", $"B=1,H=1,S={Sequence},D={Dimension}",
                q.ByteLength + k.ByteLength + v.ByteLength + output.ByteLength + lse.ByteLength, run);
        });
    }

    static TileGymBenchmark BackwardPreprocess(TileGymRuntime runtime)
    {
        const string name = "fmha_bwd_preprocess_kernel";
        var problem = new TileGymAttentionProblem(TileGymAttentionKind.BackwardPreprocess,
            Sequence, Dimension, true, runtime.Architecture);
        var candidates = TileGymAttentionCandidates.For(problem);
        var kernel = TileGymKernel.Tuned("attention.cuh", name, PreprocessSignature, problem, candidates);
        return new(kernel, (runtime, report) =>
        {
            using var o = runtime.Allocate<float>(Sequence * Dimension);
            using var d = runtime.Allocate<float>(o.Length);
            using var l = runtime.Allocate<float>(Sequence);
            using var delta = runtime.Allocate<float>(Sequence);
            using var minusL = runtime.Allocate<float>(Sequence);
            var ho = Values(o.Length, .1f);
            var hd = Values(d.Length, .03f);
            var hl = Values(l.Length, .02f);
            o.CopyFrom(ho);
            d.CopyFrom(hd);
            l.CopyFrom(hl);
            void Launch(CUfunction function, TileCppGrid grid) =>
                cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                    o.Pointer, d.Pointer, l.Pointer, delta.Pointer, minusL.Pointer, Scale).Ok();
            var ed = new float[Sequence];
            var el = new float[Sequence];
            for (var r = 0; r < Sequence; r++)
            {
                var sum = 0f;
                for (var c = 0; c < Dimension; c++)
                {
                    sum += ho[r * Dimension + c] * hd[r * Dimension + c];
                }
                ed[r] = -sum * Scale;
                el[r] = -hl[r];
            }
            void Validate()
            {
                delta.Validate(ed, name, 1e-3f, 1e-3f);
                minusL.Validate(el, name);
            }
            var run = runtime.Run(kernel, Launch, Validate);
            report.Add("attention", $"B=1,H=1,S={Sequence},D={Dimension}",
                o.ByteLength + d.ByteLength + l.ByteLength + delta.ByteLength + minusL.ByteLength, run);
        });
    }

    static unsafe TileGymBenchmark BackwardMain(TileGymRuntime runtime)
    {
        const string name = "fmha_bwd_main_kernel";
        const string signature = "const float*, const float*, const float*, const float*, const float*, " +
            "const float*, float*, float*, float*, float";
        var problem = new TileGymAttentionProblem(TileGymAttentionKind.BackwardMain,
            Sequence, Dimension, true, runtime.Architecture);
        var candidates = TileGymAttentionCandidates.For(problem);
        var kernel = TileGymKernel.Tuned("attention.cuh", name, signature, problem, candidates);
        var forward = TileGymKernel.Fixed("attention.cuh", "prefill_fmha_fwd_kernel",
            "float, 1, 1, 1, 64, 64, 64, 64, 64, true, true, 2, 1", ForwardSignature, new TileCppGrid(1));
        var preprocess = TileGymKernel.Fixed("attention.cuh", "fmha_bwd_preprocess_kernel",
            "float, 1, 1, 64, 64, 64, 2", PreprocessSignature, new TileCppGrid(1));
        return new([kernel, forward, preprocess], (runtime, report) =>
        {
            using var q = runtime.Allocate<float>(Sequence * Dimension);
            using var k = runtime.Allocate<float>(q.Length);
            using var v = runtime.Allocate<float>(q.Length);
            using var o = runtime.Allocate<float>(q.Length);
            using var dout = runtime.Allocate<float>(q.Length);
            using var l = runtime.Allocate<float>(Sequence);
            using var delta = runtime.Allocate<float>(Sequence);
            using var ml = runtime.Allocate<float>(Sequence);
            using var dq = runtime.Allocate<float>(q.Length);
            using var dk = runtime.Allocate<float>(q.Length);
            using var dv = runtime.Allocate<float>(q.Length);
            var hq = Values(q.Length, .02f);
            var hk = Values(k.Length, .018f);
            var hv = Values(v.Length, .025f);
            var hd = Values(dout.Length, .013f);
            q.CopyFrom(hq);
            k.CopyFrom(hk);
            v.CopyFrom(hv);
            dout.CopyFrom(hd);
            runtime.Launch(forward, (function, grid) =>
                cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                    q.Pointer, k.Pointer, v.Pointer, o.Pointer, l.Pointer, Scale).Ok());
            runtime.Launch(preprocess, (function, grid) =>
                cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                    o.Pointer, dout.Pointer, l.Pointer, delta.Pointer, ml.Pointer, Scale).Ok());
            runtime.Synchronize();
            void Launch(CUfunction function, TileCppGrid grid)
            {
                var pq = q.Pointer;
                var pk = k.Pointer;
                var pv = v.Pointer;
                var pd = dout.Pointer;
                var pml = ml.Pointer;
                var pdel = delta.Pointer;
                var pdq = dq.Pointer;
                var pdk = dk.Pointer;
                var pdv = dv.Pointer;
                var scale = Scale;
                var args = stackalloc void*[] { &pq, &pk, &pv, &pd, &pml, &pdel, &pdq, &pdk, &pdv, &scale };
                cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream, args, null).Ok();
            }
            AttentionBackward(hq, hk, hv, hd, out var edq, out var edk, out var edv);
            void Validate()
            {
                dq.Validate(edq, name, 3e-3f, 3e-3f);
                dk.Validate(edk, name, 3e-3f, 3e-3f);
                dv.Validate(edv, name, 3e-3f, 3e-3f);
            }
            var run = runtime.Run(kernel, Launch, Validate, reset: dq.Clear);
            report.Add("attention", $"B=1,H=1,S={Sequence},D={Dimension}", q.ByteLength + k.ByteLength +
                v.ByteLength + dout.ByteLength + dq.ByteLength + dk.ByteLength + dv.ByteLength, run);
        });
    }

    static void AttentionBackward(float[] q, float[] k, float[] v, float[] dO,
        out float[] dQ, out float[] dK, out float[] dV)
    {
        dQ = new float[q.Length];
        dK = new float[k.Length];
        dV = new float[v.Length];
        var p = new float[Sequence * Sequence];
        for (var i = 0; i < Sequence; i++)
        {
            var max = float.NegativeInfinity;
            for (var j = 0; j <= i; j++)
            {
                var dot = 0f;
                for (var x = 0; x < Dimension; x++)
                {
                    dot += q[i * Dimension + x] * k[j * Dimension + x];
                }
                p[i * Sequence + j] = dot * Scale;
                max = Math.Max(max, p[i * Sequence + j]);
            }
            var sum = 0f;
            for (var j = 0; j <= i; j++)
            {
                p[i * Sequence + j] = MathF.Exp(p[i * Sequence + j] - max);
                sum += p[i * Sequence + j];
            }
            for (var j = 0; j <= i; j++)
            {
                p[i * Sequence + j] /= sum;
            }
        }
        for (var i = 0; i < Sequence; i++)
        {
            var rowDot = 0f;
            for (var j = 0; j <= i; j++)
            {
                for (var x = 0; x < Dimension; x++)
                {
                    dV[j * Dimension + x] += p[i * Sequence + j] * dO[i * Dimension + x];
                }
                var dp = 0f;
                for (var x = 0; x < Dimension; x++)
                {
                    dp += dO[i * Dimension + x] * v[j * Dimension + x];
                }
                rowDot += p[i * Sequence + j] * dp;
            }
            for (var j = 0; j <= i; j++)
            {
                var dp = 0f;
                for (var x = 0; x < Dimension; x++)
                {
                    dp += dO[i * Dimension + x] * v[j * Dimension + x];
                }
                var ds = p[i * Sequence + j] * (dp - rowDot) * Scale;
                for (var x = 0; x < Dimension; x++)
                {
                    dQ[i * Dimension + x] += ds * k[j * Dimension + x];
                    dK[j * Dimension + x] += ds * q[i * Dimension + x];
                }
            }
        }
    }

    static float[] Attention(float[] q, float[] k, float[] v, int sq, int sk, int d, float scale, bool causal)
    {
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
            a[i] = (i % 31 - 15) * scale;
        }
        return a;
    }
}
