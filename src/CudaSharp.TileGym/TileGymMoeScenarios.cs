using System.Collections.Generic;
using CudaSharp.TileGym;
using static CudaSharp.nvcuda;

namespace CudaSharp.Tester;

static class TileGymMoeScenarios
{
    const string Header = "moe_align_block.cuh";
    static readonly TileCppGrid Single = new(1);

    public static IEnumerable<TileGymBenchmark> Create(TileGymRuntime runtime, TileGymOptions options)
    {
        yield return Alignment();
        yield return Generic();
        yield return Fp8(fc1: true);
        yield return Fp8(fc1: false);
    }

    static TileGymBenchmark Alignment()
    {
        const int experts = 4, numel = 8, tokensPerThread = 2;
        const string shape = "tokens=8,experts=4";
        var grid = new TileCppGrid(experts);
        var s1 = TileGymKernel.Fixed(Header, "moe_align_block_size_stage1", "int, 1, 4",
            "const int*, int*, int, int", grid, "BLOCK_SIZE=1");
        var s2 = TileGymKernel.Fixed(Header, "moe_align_block_size_stage2", "int, 4, 4, 4",
            "int*", grid, "PADDED_EXPERTS=4");
        var s3 = TileGymKernel.Fixed(Header, "moe_align_block_size_stage3", "int, 4, 4, 4",
            "int*, int*, const int*, int*", Single, "BLOCK_SIZE=4");
        var s4 = TileGymKernel.Fixed(Header, "moe_align_block_size_stage4", "int, 4, 4",
            "const int*, int*, int*, int*, const int*, int, int", grid, "BLOCK_SIZE=4");
        return new([s1, s2, s3, s4], (runtime, report) =>
        {
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
            void Stage1(CUfunction function, TileCppGrid grid)
            {
                counts.Clear();
                cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                    top.Pointer, counts.Pointer, numel, tokensPerThread).Ok();
            }
            void Stage2(CUfunction function, TileCppGrid grid) =>
                cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream, counts.Pointer).Ok();
            void Stage3(CUfunction function, TileCppGrid grid) =>
                cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                    total.Pointer, max.Pointer, counts.Pointer, cumsum.Pointer).Ok();
            void Stage4(CUfunction function, TileCppGrid grid) =>
                cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                    top.Pointer, sorted.Pointer, expertIds.Pointer, counts.Pointer, cumsum.Pointer,
                    numel, tokensPerThread).Ok();
            void Validate()
            {
                var cs = cumsum.CopyToHost();
                if (!cs.AsSpan().SequenceEqual([0, 4, 8, 12, 16]))
                {
                    throw new InvalidOperationException("MoE alignment cumsum validation failed.");
                }
                var totalHost = total.CopyToHost();
                var maxHost = max.CopyToHost();
                if (totalHost[0] != 16 || maxHost[0] != 3)
                {
                    throw new InvalidOperationException("MoE alignment totals validation failed.");
                }
                var actual = sorted.CopyToHost();
                int[] expected = [1, 6, 8, 8, 3, 7, 8, 8, 0, 2, 5, 8, 4, 8, 8, 8];
                if (!actual.AsSpan().SequenceEqual(expected))
                {
                    throw new InvalidOperationException("MoE sorted-token validation failed.");
                }
            }

            var run1 = runtime.Run(s1, Stage1);
            runtime.Launch(s1, Stage1);
            runtime.Synchronize();
            var run2 = runtime.Run(s2, Stage2);
            runtime.Launch(s1, Stage1);
            runtime.Launch(s2, Stage2);
            runtime.Synchronize();
            var run3 = runtime.Run(s3, Stage3);
            runtime.Launch(s1, Stage1);
            runtime.Launch(s2, Stage2);
            runtime.Launch(s3, Stage3);
            runtime.Synchronize();
            var run4 = runtime.Run(s4, Stage4, Validate, options: new(SingleLaunch: true));
            report.Add("moe-align", shape, top.ByteLength + counts.ByteLength, run1);
            report.Add("moe-align", shape, counts.ByteLength, run2);
            report.Add("moe-align", shape, counts.ByteLength + cumsum.ByteLength, run3);
            report.Add("moe-align", shape, top.ByteLength + sorted.ByteLength + expertIds.ByteLength, run4);
        });
    }

    static unsafe TileGymBenchmark Generic()
    {
        const int m = 16, n = 16, kdim = 16;
        const string name = "fused_moe_kernel";
        var kernel = TileGymKernel.Fixed("moe.cuh", name,
            "float, float, 16, 16, 16, 1, false, false, 16, 16, 16, 16, 1, 16, 1, 256, 16, 1, 16, 1, 0, 0, 0, 0, 0, 16",
            "const float*, const float*, float*, const float*, const float*, const float*, const int*, const int*, " +
            "const int*, int", Single);
        return new(kernel, (runtime, report) =>
        {
            using var a = runtime.Allocate<float>(m * kdim);
            using var b = runtime.Allocate<float>(n * kdim);
            using var c = runtime.Allocate<float>(m * n);
            using var routing = new Routing(runtime, m);
            var ha = Values(a.Length, .03f);
            var hb = Values(b.Length, .02f);
            a.CopyFrom(ha);
            b.CopyFrom(hb);
            void Launch(CUfunction function, TileCppGrid grid)
            {
                var pa = a.Pointer;
                var pb = b.Pointer;
                var pc = c.Pointer;
                var nil = IntPtr.Zero;
                var pw = routing.Weights.Pointer;
                var ps = routing.Sorted.Pointer;
                var pe = routing.Experts.Pointer;
                var pp = routing.Padded.Pointer;
                var valid = m;
                var args = stackalloc void*[] { &pa, &pb, &pc, &nil, &nil, &pw, &ps, &pe, &pp, &valid };
                cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream, args, null).Ok();
            }
            var expected = new float[m * n];
            for (var row = 0; row < m; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    for (var x = 0; x < kdim; x++)
                    {
                        expected[row * n + col] += ha[row * kdim + x] * hb[col * kdim + x];
                    }
                }
            }
            var run = runtime.Run(kernel, Launch, () => c.Validate(expected, name, 2e-3f, 2e-3f));
            report.Add("moe", $"M={m},N={n},K={kdim},E=1", a.ByteLength + b.ByteLength + c.ByteLength, run);
        });
    }

    static unsafe TileGymBenchmark Fp8(bool fc1)
    {
        const int m = 16, n = 16, kdim = 16;
        var name = fc1 ? "fused_moe_fc1_layer_kernel" : "fused_moe_fc2_layer_kernel";
        var kernel = TileGymKernel.Fixed("moe.cuh", name, "float, __nv_fp8_e4m3, 16, 16, 16, 1, false",
            fc1 ? Fc1Signature : Fc2Signature, Single, "E4M3,unit-scales,zero-input");
        return new(kernel, (runtime, report) =>
        {
            using var a = runtime.Allocate<byte>(m * kdim);
            using var b1 = runtime.Allocate<byte>(n * kdim);
            using var b2 = fc1 ? runtime.Allocate<byte>(n * kdim) : null;
            using var c = runtime.Allocate<float>(m * n);
            using var scaleA = runtime.Allocate<float>(m);
            using var scaleB1 = runtime.Allocate<float>(1);
            using var scaleB2 = fc1 ? runtime.Allocate<float>(1) : null;
            using var routing = new Routing(runtime, m);
            var ones = new float[m];
            Array.Fill(ones, 1f);
            scaleA.CopyFrom(ones);
            scaleB1.CopyFrom([1f]);
            scaleB2?.CopyFrom([1f]);
            void Launch(CUfunction function, TileCppGrid grid)
            {
                var pa = a.Pointer;
                var pb1 = b1.Pointer;
                var pb2 = b2?.Pointer ?? default;
                var pc = c.Pointer;
                var psa = scaleA.Pointer;
                var psb1 = scaleB1.Pointer;
                var psb2 = scaleB2?.Pointer ?? default;
                var pw = routing.Weights.Pointer;
                var ps = routing.Sorted.Pointer;
                var pe = routing.Experts.Pointer;
                var pp = routing.Padded.Pointer;
                var M = m;
                var N = n;
                var K = kdim;
                var EM = m;
                var valid = m;
                var strideAm = 16;
                var strideAk = 1;
                var strideBe = 256;
                var strideBk = 1;
                var strideBn = 16;
                var strideCm = 16;
                var strideCn = 1;
                var strideAsm = 1;
                var strideAsk = 1;
                var strideBse = 1;
                var strideBsk = 0;
                var strideBsn = 0;
                var groupN = 16;
                var groupK = 16;
                var topK = 1;
                if (fc1)
                {
                    var args = stackalloc void*[]
                    {
                        &pa, &pb1, &pb2, &pc, &psa, &psb1, &psb2, &pw, &ps, &pe, &pp, &M, &N, &K, &EM, &valid,
                        &strideAm, &strideAk, &strideBe, &strideBk, &strideBn, &strideBe, &strideBk, &strideBn,
                        &strideCm, &strideCn, &strideAsm, &strideAsk, &strideBse, &strideBsk, &strideBsn,
                        &strideBse, &strideBsk, &strideBsn, &groupN, &groupK, &topK
                    };
                    cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream, args, null).Ok();
                }
                else
                {
                    var args = stackalloc void*[]
                    {
                        &pa, &pb1, &pc, &psa, &psb1, &pw, &ps, &pe, &pp, &M, &N, &K, &EM, &valid,
                        &strideAm, &strideAk, &strideBe, &strideBk, &strideBn, &strideCm, &strideCn,
                        &strideAsm, &strideAsk, &strideBse, &strideBsk, &strideBsn, &groupN, &groupK, &topK
                    };
                    cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream, args, null).Ok();
                }
            }
            var expected = new float[c.Length];
            var run = runtime.Run(kernel, Launch, () => c.Validate(expected, name));
            report.Add("moe-fp8", $"M={m},N={n},K={kdim},E=1",
                a.ByteLength + b1.ByteLength + (b2?.ByteLength ?? 0) + c.ByteLength, run);
        });
    }

    /// <summary>Identity routing of all rows to a single expert.</summary>
    sealed class Routing : IDisposable
    {
        public Routing(TileGymRuntime runtime, int rows)
        {
            Weights = runtime.Allocate<float>(rows);
            Sorted = runtime.Allocate<int>(rows);
            Experts = runtime.Allocate<int>(1);
            Padded = runtime.Allocate<int>(1);
            var weights = new float[rows];
            Array.Fill(weights, 1f);
            Weights.CopyFrom(weights);
            var sorted = new int[rows];
            for (var i = 0; i < rows; i++)
            {
                sorted[i] = i;
            }
            Sorted.CopyFrom(sorted);
            Experts.CopyFrom([0]);
            Padded.CopyFrom([rows]);
        }

        public CudaBuffer<float> Weights { get; }
        public CudaBuffer<int> Sorted { get; }
        public CudaBuffer<int> Experts { get; }
        public CudaBuffer<int> Padded { get; }

        public void Dispose()
        {
            Weights.Dispose();
            Sorted.Dispose();
            Experts.Dispose();
            Padded.Dispose();
        }
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
