using System;
using CudaSharp.Tile;
using static CudaSharp.nvcuda;

namespace CudaSharp.Tester;

static class TileGymMoeScenarios
{
    static readonly TileCppConfig Config = new([]);

    public static void RunAll(TileGymRuntime runtime, TileGymReport report)
    {
        RunAlignment(runtime, report);
        RunGeneric(runtime, report);
        RunFp8(runtime, report, true);
        RunFp8(runtime, report, false);
    }

    static unsafe void RunAlignment(TileGymRuntime runtime, TileGymReport report)
    {
        const int experts = 4, numel = 8, block = 4, tokensPerThread = 2;
        var ids = new[] { 2, 0, 2, 1, 3, 2, 0, 1 };
        using var top = runtime.Allocate<int>(numel);
        using var counts = runtime.Allocate<int>((experts + 1) * experts);
        using var cumsum = runtime.Allocate<int>(experts + 1);
        using var total = runtime.Allocate<int>(1);
        using var max = runtime.Allocate<int>(1);
        using var sorted = runtime.Allocate<int>(16);
        using var expertIds = runtime.Allocate<int>(4);
        top.CopyFrom(ids);
        sorted.CopyFrom([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]);
        using var s1 = TileGymKernel.Create(
            runtime,
            "moe_align_block.cuh",
            "moe_align_block_size_stage1",
            "int, 1, 4",
            "const int*, int*, int, int"
        );
        using var s2 = TileGymKernel.Create(
            runtime,
            "moe_align_block.cuh",
            "moe_align_block_size_stage2",
            "int, 4, 4",
            "int*"
        );
        using var s3 = TileGymKernel.Create(
            runtime,
            "moe_align_block.cuh",
            "moe_align_block_size_stage3",
            "int, 4, 4",
            "int*, int*, const int*, int*"
        );
        using var s4 = TileGymKernel.Create(
            runtime,
            "moe_align_block.cuh",
            "moe_align_block_size_stage4",
            "int, 4, 4",
            "const int*, int*, int*, int*, const int*, int, int"
        );

        void Stage1()
        {
            var pt = top.Pointer.Value;
            var pc = counts.Pointer.Value;
            var n = numel;
            var per = tokensPerThread;
            var args = stackalloc IntPtr[] { (IntPtr)(&pt), (IntPtr)(&pc), (IntPtr)(&n), (IntPtr)(&per) };
            s1.Launch(Config, new(experts), runtime.Stream, new(args, 4));
        }

        void Stage2()
        {
            var pc = counts.Pointer.Value;
            var args = stackalloc IntPtr[] { (IntPtr)(&pc) };
            s2.Launch(Config, new(experts), runtime.Stream, new(args, 1));
        }

        void Stage3()
        {
            var pt = total.Pointer.Value;
            var pm = max.Pointer.Value;
            var pc = counts.Pointer.Value;
            var ps = cumsum.Pointer.Value;
            var args = stackalloc IntPtr[] { (IntPtr)(&pt), (IntPtr)(&pm), (IntPtr)(&pc), (IntPtr)(&ps) };
            s3.Launch(Config, new(1), runtime.Stream, new(args, 4));
        }

        void Stage4()
        {
            var pt = top.Pointer.Value;
            var ps = sorted.Pointer.Value;
            var pe = expertIds.Pointer.Value;
            var pc = counts.Pointer.Value;
            var pcs = cumsum.Pointer.Value;
            var n = numel;
            var per = tokensPerThread;
            var args = stackalloc IntPtr[] { (IntPtr)(&pt), (IntPtr)(&ps), (IntPtr)(&pe), (IntPtr)(&pc), (IntPtr)(&pcs), (IntPtr)(&n), (IntPtr)(&per) };
            s4.Launch(Config, new(experts), runtime.Stream, new(args, 7));
        }

        var t1 = TileGymKernel.Measure(runtime, () => { counts.Clear(); Stage1(); });
        counts.Clear();
        Stage1();
        cuStreamSynchronize(runtime.Stream).Ok();
        var t2 = TileGymKernel.Measure(runtime, Stage2);
        counts.Clear();
        Stage1();
        Stage2();
        cuStreamSynchronize(runtime.Stream).Ok();
        var t3 = TileGymKernel.Measure(runtime, Stage3);
        counts.Clear();
        Stage1();
        Stage2();
        Stage3();
        cuStreamSynchronize(runtime.Stream).Ok();
        var t4 = TileGymKernel.MeasureOnce(runtime, () => s4.GetFunction(Config), Stage4);

        var cs = cumsum.CopyToHost();
        if (!cs.AsSpan().SequenceEqual([0, 4, 8, 12, 16]))
            throw new InvalidOperationException("MoE alignment cumsum validation failed.");
        if (total.CopyToHost()[0] != 16 || max.CopyToHost()[0] != 3)
            throw new InvalidOperationException("MoE alignment totals validation failed.");
        var actual = sorted.CopyToHost();
        var expected = new[] { 1, 6, 8, 8, 3, 7, 8, 8, 0, 2, 5, 8, 4, 8, 8, 8 };
        if (!actual.AsSpan().SequenceEqual(expected))
            throw new InvalidOperationException("MoE sorted-token validation failed.");

        TileGymKernel.Report(
            report,
            "moe-align",
            "moe_align_block_size_stage1",
            "tokens=8,experts=4",
            "BLOCK_SIZE=1",
            top.ByteLength + counts.ByteLength,
            t1
        );
        TileGymKernel.Report(
            report,
            "moe-align",
            "moe_align_block_size_stage2",
            "tokens=8,experts=4",
            "PADDED_EXPERTS=4",
            counts.ByteLength,
            t2
        );
        TileGymKernel.Report(
            report,
            "moe-align",
            "moe_align_block_size_stage3",
            "tokens=8,experts=4",
            "BLOCK_SIZE=4",
            counts.ByteLength + cumsum.ByteLength,
            t3
        );
        TileGymKernel.Report(
            report,
            "moe-align",
            "moe_align_block_size_stage4",
            "tokens=8,experts=4",
            "BLOCK_SIZE=4",
            top.ByteLength + sorted.ByteLength + expertIds.ByteLength,
            t4
        );
    }

    static unsafe void RunGeneric(TileGymRuntime runtime, TileGymReport report)
    {
        const int m = 16, n = 16, kdim = 16;
        const string name = "fused_moe_kernel";
        var templates = "float, float, 16, 16, 16, 1, false, false, 16, 16, 16, 16, 1, 16, 1, 256, 16, 1, 16, 1, 0, 0, 0, 0, 0, 16";
        using var kernel = TileGymKernel.Create(
            runtime,
            "moe.cuh",
            name,
            templates,
            "const float*, const float*, float*, const float*, const float*, const float*, const int*, const int*, const int*, int"
        );
        using var a = runtime.Allocate<float>(m * kdim);
        using var b = runtime.Allocate<float>(n * kdim);
        using var c = runtime.Allocate<float>(m * n);
        using var weights = runtime.Allocate<float>(m);
        using var sorted = runtime.Allocate<int>(m);
        using var experts = runtime.Allocate<int>(1);
        using var padded = runtime.Allocate<int>(1);
        var ha = Values(a.Length, .03f);
        var hb = Values(b.Length, .02f);
        a.CopyFrom(ha);
        b.CopyFrom(hb);
        var hw = new float[m];
        Array.Fill(hw, 1f);
        weights.CopyFrom(hw);
        var hs = new int[m];
        for (var i = 0; i < m; i++)
            hs[i] = i;
        sorted.CopyFrom(hs);
        experts.CopyFrom([0]);
        padded.CopyFrom([m]);

        void Launch()
        {
            var pa = a.Pointer.Value;
            var pb = b.Pointer.Value;
            var pc = c.Pointer.Value;
            var nil = IntPtr.Zero;
            var pw = weights.Pointer.Value;
            var ps = sorted.Pointer.Value;
            var pe = experts.Pointer.Value;
            var pp = padded.Pointer.Value;
            var valid = m;
            var args = stackalloc IntPtr[]
            {
                (IntPtr)(&pa), (IntPtr)(&pb), (IntPtr)(&pc), (IntPtr)(&nil),
                (IntPtr)(&nil), (IntPtr)(&pw), (IntPtr)(&ps), (IntPtr)(&pe),
                (IntPtr)(&pp), (IntPtr)(&valid)
            };
            kernel.Launch(Config, new(1), runtime.Stream, new(args, 10));
        }

        var timing = TileGymKernel.Measure(runtime, Launch);
        var expected = new float[m * n];
        for (var row = 0; row < m; row++)
            for (var col = 0; col < n; col++)
                for (var x = 0; x < kdim; x++)
                    expected[row * n + col] += ha[row * kdim + x] * hb[col * kdim + x];
        TileGymKernel.Validate(c.CopyToHost(), expected, name, 2e-3f, 2e-3f);
        TileGymKernel.Report(report, "moe", name, $"M={m},N={n},K={kdim},E=1", templates, a.ByteLength + b.ByteLength + c.ByteLength, timing);
    }

    static unsafe void RunFp8(TileGymRuntime runtime, TileGymReport report, bool fc1)
    {
        const int m = 16, n = 16, kdim = 16;
        var name = fc1 ? "fused_moe_fc1_layer_kernel" : "fused_moe_fc2_layer_kernel";
        var signature = fc1 ? Fc1Signature : Fc2Signature;
        using var kernel = TileGymKernel.Create(
            runtime,
            "moe.cuh",
            name,
            "float, __nv_fp8_e4m3, 16, 16, 16, 1, false",
            signature);
        using var a = runtime.Allocate<byte>(m * kdim);
        using var b1 = runtime.Allocate<byte>(n * kdim);
        using var b2 = fc1 ? runtime.Allocate<byte>(n * kdim) : null;
        using var c = runtime.Allocate<float>(m * n);
        using var scaleA = runtime.Allocate<float>(m);
        using var scaleB1 = runtime.Allocate<float>(1);
        using var scaleB2 = fc1 ? runtime.Allocate<float>(1) : null;
        using var weights = runtime.Allocate<float>(m);
        using var sorted = runtime.Allocate<int>(m);
        using var experts = runtime.Allocate<int>(1);
        using var padded = runtime.Allocate<int>(1);
        var ones = new float[m];
        Array.Fill(ones, 1f);
        scaleA.CopyFrom(ones);
        scaleB1.CopyFrom([1f]);
        scaleB2?.CopyFrom([1f]);
        weights.CopyFrom(ones);
        var ids = new int[m];
        for (var i = 0; i < m; i++)
        {
            ids[i] = i;
        }
        sorted.CopyFrom(ids);
        experts.CopyFrom([0]);
        padded.CopyFrom([m]);

        void Launch()
        {
            var pa = a.Pointer.Value;
            var pb1 = b1.Pointer.Value;
            var pb2 = b2?.Pointer.Value ?? IntPtr.Zero;
            var pc = c.Pointer.Value;
            var psa = scaleA.Pointer.Value;
            var psb1 = scaleB1.Pointer.Value;
            var psb2 = scaleB2?.Pointer.Value ?? IntPtr.Zero;
            var pw = weights.Pointer.Value;
            var ps = sorted.Pointer.Value;
            var pe = experts.Pointer.Value;
            var pp = padded.Pointer.Value;
            var M = m;
            var N = n;
            var K = kdim;
            var EM = m;
            var valid = m;
            var strideAm = 16;
            var strideAk = 1;
            var strideB1e = 256;
            var strideB1k = 1;
            var strideB1n = 16;
            var strideB2e = 256;
            var strideB2k = 1;
            var strideB2n = 16;
            var strideCm = 16;
            var strideCn = 1;
            var strideAsm = 1;
            var strideAsk = 1;
            var strideB1se = 1;
            var strideB1sk = 0;
            var strideB1sn = 0;
            var strideB2se = 1;
            var strideB2sk = 0;
            var strideB2sn = 0;
            var groupN = 16;
            var groupK = 16;
            var topK = 1;

            if (fc1)
            {
                var args = stackalloc IntPtr[]
                {
                    (IntPtr)(&pa), (IntPtr)(&pb1), (IntPtr)(&pb2), (IntPtr)(&pc),
                    (IntPtr)(&psa), (IntPtr)(&psb1), (IntPtr)(&psb2), (IntPtr)(&pw),
                    (IntPtr)(&ps), (IntPtr)(&pe), (IntPtr)(&pp), (IntPtr)(&M),
                    (IntPtr)(&N), (IntPtr)(&K), (IntPtr)(&EM), (IntPtr)(&valid),
                    (IntPtr)(&strideAm), (IntPtr)(&strideAk), (IntPtr)(&strideB1e),
                    (IntPtr)(&strideB1k), (IntPtr)(&strideB1n), (IntPtr)(&strideB2e),
                    (IntPtr)(&strideB2k), (IntPtr)(&strideB2n), (IntPtr)(&strideCm),
                    (IntPtr)(&strideCn), (IntPtr)(&strideAsm), (IntPtr)(&strideAsk),
                    (IntPtr)(&strideB1se), (IntPtr)(&strideB1sk), (IntPtr)(&strideB1sn),
                    (IntPtr)(&strideB2se), (IntPtr)(&strideB2sk), (IntPtr)(&strideB2sn),
                    (IntPtr)(&groupN), (IntPtr)(&groupK), (IntPtr)(&topK)
                };
                kernel.Launch(Config, new(1), runtime.Stream, new(args, 37));
            }
            else
            {
                var args = stackalloc IntPtr[]
                {
                    (IntPtr)(&pa), (IntPtr)(&pb1), (IntPtr)(&pc), (IntPtr)(&psa),
                    (IntPtr)(&psb1), (IntPtr)(&pw), (IntPtr)(&ps), (IntPtr)(&pe),
                    (IntPtr)(&pp), (IntPtr)(&M), (IntPtr)(&N), (IntPtr)(&K),
                    (IntPtr)(&EM), (IntPtr)(&valid), (IntPtr)(&strideAm), (IntPtr)(&strideAk),
                    (IntPtr)(&strideB1e), (IntPtr)(&strideB1k), (IntPtr)(&strideB1n),
                    (IntPtr)(&strideCm), (IntPtr)(&strideCn), (IntPtr)(&strideAsm),
                    (IntPtr)(&strideAsk), (IntPtr)(&strideB1se), (IntPtr)(&strideB1sk),
                    (IntPtr)(&strideB1sn), (IntPtr)(&groupN), (IntPtr)(&groupK),
                    (IntPtr)(&topK)
                };
                kernel.Launch(Config, new(1), runtime.Stream, new(args, 29));
            }
        }

        var timing = TileGymKernel.Measure(runtime, Launch);
        TileGymKernel.Validate(c.CopyToHost(), new float[c.Length], name);
        TileGymKernel.Report(
            report,
            "moe-fp8",
            name,
            $"M={m},N={n},K={kdim},E=1",
            "E4M3,unit-scales,zero-input",
            a.ByteLength + b1.ByteLength + (b2?.ByteLength ?? 0) + c.ByteLength,
            timing);
    }

    static float[] Values(int count, float scale)
    {
        var values = new float[count];
        for (var i = 0; i < count; i++)
        {
            values[i] = (i % 17 - 8) * scale;
        }
        return values;
    }

    const string Fc1Signature = "const __nv_fp8_e4m3*, const __nv_fp8_e4m3*, const __nv_fp8_e4m3*, " +
        "float*, const float*, const float*, const float*, const float*, const int*, const int*, " +
        "const int*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, " +
        "int, int, int, int, int, int, int, int, int, int, int";

    const string Fc2Signature = "const __nv_fp8_e4m3*, const __nv_fp8_e4m3*, float*, const float*, " +
        "const float*, const float*, const int*, const int*, const int*, int, int, int, int, int, " +
        "int, int, int, int, int, int, int, int, int, int, int, int, int, int, int";
}
