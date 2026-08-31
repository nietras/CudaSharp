using System;
using CudaSharp.Tile;
using static CudaSharp.nvcuda;

namespace CudaSharp.Tester;

static class TileGymAttentionScenarios
{
    static readonly TileCppConfig Config = new([]);
    const int Sequence = 64;
    const int Dimension = 64;
    const float Scale = .125f;

    public static void RunAll(TileGymRuntime runtime, TileGymReport report)
    {
        RunPrefill(runtime, report, "attention.cuh", "prefill_fmha_fwd_kernel",
            "float, 1, 1, 1, 64, 64, 64, 64, 64, true, true, 2, 1",
            "const float*, const float*, const float*, float*, float*, float", true);
        RunAttentionBackward(runtime, report);
        RunAttentionBackwardMain(runtime, report);
        RunSink(runtime, report);
        RunPrefill(runtime, report, "gemma_attention.cuh", "gemma_attention_fwd_kernel",
            "float, 1, 1, 1, 64, 64, 64, 64, 64, true, 0, false, 2",
            "const float*, const float*, const float*, float*, float, float", true, gemma: true);
        RunDecode(runtime, report, "flash_decode.cuh", "attention_decode_kernel_optimized",
            "float, 1, 1, 64, 1, 8, 64, 64, 64, 1", false, false);
        RunDecode(runtime, report, "attention_sink_decode.cuh", "attention_sink_decode_kernel",
            "float, 1, 1, 1, 64, 1, 8, 64, 64, 64, 1, 0, false, 1", true, false);
        RunDecode(runtime, report, "gemma_attention_decode.cuh", "gemma_attention_decode_kernel",
            "float, 1, 1, 64, 1, 8, 64, 64, 64, 1, 0, false, 1", false, true);
    }

    static unsafe void RunPrefill(TileGymRuntime runtime, TileGymReport report, string header, string name,
        string templates, string signature, bool causal, bool gemma = false)
    {
        using var kernel = TileGymKernel.Create(runtime, header, name, templates, signature);
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

        void Launch()
        {
            var pq = q.Pointer.Value;
            var pk = k.Pointer.Value;
            var pv = v.Pointer.Value;
            var po = output.Pointer.Value;
            var pl = lse?.Pointer.Value ?? IntPtr.Zero;
            var scale = Scale;
            var cap = 0f;
            if (gemma)
            {
                var args = stackalloc IntPtr[] { (IntPtr)(&pq), (IntPtr)(&pk), (IntPtr)(&pv), (IntPtr)(&po), (IntPtr)(&scale), (IntPtr)(&cap) };
                kernel.Launch(Config, new(1, 1), runtime.Stream, new(args, 6));
            }
            else
            {
                var args = stackalloc IntPtr[] { (IntPtr)(&pq), (IntPtr)(&pk), (IntPtr)(&pv), (IntPtr)(&po), (IntPtr)(&pl), (IntPtr)(&scale) };
                kernel.Launch(Config, new(1, 1), runtime.Stream, new(args, 6));
            }
        }

        var timing = TileGymKernel.Measure(runtime, Launch);
        TileGymKernel.Validate(output.CopyToHost(), Attention(hq, hk, hv, Sequence, Sequence, Dimension, Scale, causal), name, 2e-3f, 2e-3f);
        TileGymKernel.Report(
            report, "attention", name,
            $"B=1,H=1,Sq={Sequence},Sk={Sequence},D={Dimension}", templates,
            q.ByteLength + k.ByteLength + v.ByteLength + output.ByteLength + (lse?.ByteLength ?? 0),
            timing);
    }

    static unsafe void RunSink(TileGymRuntime runtime, TileGymReport report)
    {
        const string name = "attention_sink_fwd_kernel";
        using var kernel = TileGymKernel.Create(
            runtime, "attention_sink.cuh", name, "float, 64, 64, 64, false",
            "float*, float*, float*, float*, float, float*, float*, int, int, int, int, int, int");
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

        void Launch()
        {
            var pq = q.Pointer.Value;
            var pk = k.Pointer.Value;
            var pv = v.Pointer.Value;
            var ps = sinks.Pointer.Value;
            var pm = m.Pointer.Value;
            var po = output.Pointer.Value;
            var scale = Scale;
            var start = 0;
            var z = 1;
            var h = 1;
            var nq = Sequence;
            var nk = Sequence;
            var bandwidth = 0;
            var args = stackalloc IntPtr[]
            {
                (IntPtr)(&pq), (IntPtr)(&pk), (IntPtr)(&pv), (IntPtr)(&ps),
                (IntPtr)(&scale), (IntPtr)(&pm), (IntPtr)(&po), (IntPtr)(&start),
                (IntPtr)(&z), (IntPtr)(&h), (IntPtr)(&nq), (IntPtr)(&nk),
                (IntPtr)(&bandwidth)
            };
            kernel.Launch(Config, new(1, 1), runtime.Stream, new(args, 13));
        }

        var timing = TileGymKernel.Measure(runtime, Launch);
        TileGymKernel.Validate(output.CopyToHost(), Attention(hq, hk, hv, Sequence, Sequence, Dimension, Scale, true), name, 2e-3f, 2e-3f);
        TileGymKernel.Report(report, "attention", name, $"B=1,H=1,S={Sequence},D={Dimension}", "float,BLOCK_M=64,BLOCK_N=64", q.ByteLength + k.ByteLength + v.ByteLength + output.ByteLength, timing);
    }

    static unsafe void RunDecode(TileGymRuntime runtime, TileGymReport report, string header, string name, string templates, bool sink, bool gemma)
    {
        var signature = sink
            ? "const float*, const float*, const float*, const float*, float*, float*, const int*, float"
            : gemma
                ? "const float*, const float*, const float*, float*, float*, float, float"
                : "const float*, const float*, const float*, float*, float*, float";
        using var kernel = TileGymKernel.Create(runtime, header, name, templates, signature);
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

        void Launch()
        {
            var pq = q.Pointer.Value;
            var pk = k.Pointer.Value;
            var pv = v.Pointer.Value;
            var po = output.Pointer.Value;
            var pl = lse.Pointer.Value;
            var ps = start.Pointer.Value;
            var sinkp = sinks.Pointer.Value;
            var scale = Scale;
            var cap = 0f;
            if (sink)
            {
                var args = stackalloc IntPtr[]
                {
                    (IntPtr)(&pq), (IntPtr)(&pk), (IntPtr)(&pv), (IntPtr)(&sinkp),
                    (IntPtr)(&po), (IntPtr)(&pl), (IntPtr)(&ps), (IntPtr)(&scale)
                };
                kernel.Launch(Config, new(1, 1, 1), runtime.Stream, new(args, 8));
            }
            else if (gemma)
            {
                var args = stackalloc IntPtr[] { (IntPtr)(&pq), (IntPtr)(&pk), (IntPtr)(&pv), (IntPtr)(&po), (IntPtr)(&pl), (IntPtr)(&scale), (IntPtr)(&cap) };
                kernel.Launch(Config, new(1, 1, 1), runtime.Stream, new(args, 7));
            }
            else
            {
                var args = stackalloc IntPtr[] { (IntPtr)(&pq), (IntPtr)(&pk), (IntPtr)(&pv), (IntPtr)(&po), (IntPtr)(&pl), (IntPtr)(&scale) };
                kernel.Launch(Config, new(1, 1, 1), runtime.Stream, new(args, 6));
            }
        }

        var timing = TileGymKernel.Measure(runtime, Launch);
        TileGymKernel.Validate(output.CopyToHost(), Attention(hq, hk, hv, 1, Sequence, Dimension, Scale, false), name, 2e-3f, 2e-3f);
        TileGymKernel.Report(report, "decode", name, $"B=1,H=1,S={Sequence},D={Dimension}", templates, q.ByteLength + k.ByteLength + v.ByteLength + output.ByteLength + lse.ByteLength, timing);
    }

    static unsafe void RunAttentionBackward(TileGymRuntime runtime, TileGymReport report)
    {
        const string name = "fmha_bwd_preprocess_kernel";
        using var kernel = TileGymKernel.Create(runtime, "attention.cuh", name, "float, 1, 1, 64, 64, 64, 2", "const float*, const float*, const float*, float*, float*, float");
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

        void Launch()
        {
            var po = o.Pointer.Value;
            var pd = d.Pointer.Value;
            var pl = l.Pointer.Value;
            var pdel = delta.Pointer.Value;
            var pml = minusL.Pointer.Value;
            var scale = Scale;
            var args = stackalloc IntPtr[] { (IntPtr)(&po), (IntPtr)(&pd), (IntPtr)(&pl), (IntPtr)(&pdel), (IntPtr)(&pml), (IntPtr)(&scale) };
            kernel.Launch(Config, new(1, 1), runtime.Stream, new(args, 6));
        }

        var timing = TileGymKernel.Measure(runtime, Launch);
        var ed = new float[Sequence];
        var el = new float[Sequence];
        for (var r = 0; r < Sequence; r++)
        {
            var sum = 0f;
            for (var c = 0; c < Dimension; c++) sum += ho[r * Dimension + c] * hd[r * Dimension + c];
            ed[r] = -sum * Scale;
            el[r] = -hl[r];
        }
        TileGymKernel.Validate(delta.CopyToHost(), ed, name, 1e-3f, 1e-3f);
        TileGymKernel.Validate(minusL.CopyToHost(), el, name);
        TileGymKernel.Report(report, "attention", name, $"B=1,H=1,S={Sequence},D={Dimension}", "float,BLOCK_M=64,BLOCK_D=64", o.ByteLength + d.ByteLength + l.ByteLength + delta.ByteLength + minusL.ByteLength, timing);
    }

    static unsafe void RunAttentionBackwardMain(TileGymRuntime runtime, TileGymReport report)
    {
        const string name = "fmha_bwd_main_kernel";
        using var forward = TileGymKernel.Create(
            runtime, "attention.cuh", "prefill_fmha_fwd_kernel",
            "float, 1, 1, 1, 64, 64, 64, 64, 64, true, true, 2, 1",
            "const float*, const float*, const float*, float*, float*, float");
        using var preprocess = TileGymKernel.Create(
            runtime, "attention.cuh", "fmha_bwd_preprocess_kernel",
            "float, 1, 1, 64, 64, 64, 2",
            "const float*, const float*, const float*, float*, float*, float");
        using var kernel = TileGymKernel.Create(
            runtime, "attention.cuh", name, "float, 1, 1, 64, 64, 64, 64, 64, true, 2",
            "const float*, const float*, const float*, const float*, const float*, const float*, " +
            "float*, float*, float*, float");

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

        void Forward()
        {
            var pq = q.Pointer.Value;
            var pk = k.Pointer.Value;
            var pv = v.Pointer.Value;
            var po = o.Pointer.Value;
            var pl = l.Pointer.Value;
            var scale = Scale;
            var args = stackalloc IntPtr[] { (IntPtr)(&pq), (IntPtr)(&pk), (IntPtr)(&pv), (IntPtr)(&po), (IntPtr)(&pl), (IntPtr)(&scale) };
            forward.Launch(Config, new(1, 1), runtime.Stream, new(args, 6));
        }

        void Preprocess()
        {
            var po = o.Pointer.Value;
            var pd = dout.Pointer.Value;
            var pl = l.Pointer.Value;
            var pdel = delta.Pointer.Value;
            var pml = ml.Pointer.Value;
            var scale = Scale;
            var args = stackalloc IntPtr[] { (IntPtr)(&po), (IntPtr)(&pd), (IntPtr)(&pl), (IntPtr)(&pdel), (IntPtr)(&pml), (IntPtr)(&scale) };
            preprocess.Launch(Config, new(1, 1), runtime.Stream, new(args, 6));
        }

        Forward();
        Preprocess();
        cuStreamSynchronize(runtime.Stream).Ok();
        dq.Clear();

        void Launch()
        {
            var pq = q.Pointer.Value;
            var pk = k.Pointer.Value;
            var pv = v.Pointer.Value;
            var pd = dout.Pointer.Value;
            var pml = ml.Pointer.Value;
            var pdel = delta.Pointer.Value;
            var pdq = dq.Pointer.Value;
            var pdk = dk.Pointer.Value;
            var pdv = dv.Pointer.Value;
            var scale = Scale;
            var args = stackalloc IntPtr[]
            {
                (IntPtr)(&pq), (IntPtr)(&pk), (IntPtr)(&pv), (IntPtr)(&pd),
                (IntPtr)(&pml), (IntPtr)(&pdel), (IntPtr)(&pdq), (IntPtr)(&pdk),
                (IntPtr)(&pdv), (IntPtr)(&scale)
            };
            kernel.Launch(Config, new(1, 1), runtime.Stream, new(args, 10));
        }

        var timing = TileGymKernel.Measure(runtime, Launch);
        dq.Clear();
        Launch();
        static void Cpu(float[] q, float[] k, float[] v, float[] dO, out float[] dQ, out float[] dK, out float[] dV)
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
                    for (var x = 0; x < Dimension; x++) dot += q[i * Dimension + x] * k[j * Dimension + x];
                    p[i * Sequence + j] = dot * Scale;
                    max = Math.Max(max, p[i * Sequence + j]);
                }
                var sum = 0f;
                for (var j = 0; j <= i; j++)
                {
                    p[i * Sequence + j] = MathF.Exp(p[i * Sequence + j] - max);
                    sum += p[i * Sequence + j];
                }
                for (var j = 0; j <= i; j++) p[i * Sequence + j] /= sum;
            }
            for (var i = 0; i < Sequence; i++)
            {
                var rowDot = 0f;
                for (var j = 0; j <= i; j++)
                {
                    for (var x = 0; x < Dimension; x++) dV[j * Dimension + x] += p[i * Sequence + j] * dO[i * Dimension + x];
                    var dp = 0f;
                    for (var x = 0; x < Dimension; x++) dp += dO[i * Dimension + x] * v[j * Dimension + x];
                    rowDot += p[i * Sequence + j] * dp;
                }
                for (var j = 0; j <= i; j++)
                {
                    var dp = 0f;
                    for (var x = 0; x < Dimension; x++) dp += dO[i * Dimension + x] * v[j * Dimension + x];
                    var ds = p[i * Sequence + j] * (dp - rowDot) * Scale;
                    for (var x = 0; x < Dimension; x++)
                    {
                        dQ[i * Dimension + x] += ds * k[j * Dimension + x];
                        dK[j * Dimension + x] += ds * q[i * Dimension + x];
                    }
                }
            }
        }
        Cpu(hq, hk, hv, hd, out var edq, out var edk, out var edv);
        TileGymKernel.Validate(dq.CopyToHost(), edq, name, 3e-3f, 3e-3f);
        TileGymKernel.Validate(dk.CopyToHost(), edk, name, 3e-3f, 3e-3f);
        TileGymKernel.Validate(dv.CopyToHost(), edv, name, 3e-3f, 3e-3f);
        TileGymKernel.Report(report, "attention", name, $"B=1,H=1,S={Sequence},D={Dimension}", "float,BLOCK_M=64,BLOCK_N=64", q.ByteLength + k.ByteLength + v.ByteLength + dout.ByteLength + dq.ByteLength + dk.ByteLength + dv.ByteLength, timing);
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
                if (causal && j > i) { scores[j] = float.NegativeInfinity; continue; }
                var dot = 0f;
                for (var x = 0; x < d; x++) dot += q[i * d + x] * k[j * d + x];
                scores[j] = dot * scale;
                max = Math.Max(max, scores[j]);
            }
            var sum = 0f;
            for (var j = 0; j < sk; j++) { scores[j] = MathF.Exp(scores[j] - max); sum += scores[j]; }
            for (var j = 0; j < sk; j++) { var p = scores[j] / sum; for (var x = 0; x < d; x++) o[i * d + x] += p * v[j * d + x]; }
        }
        return o;
    }
    static float[] Values(int count, float scale) { var a = new float[count]; for (var i = 0; i < count; i++) a[i] = (i % 31 - 15) * scale; return a; }
}
