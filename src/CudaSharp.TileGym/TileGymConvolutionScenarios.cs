using System.Collections.Generic;
using CudaSharp.TileGym;
using static CudaSharp.nvcuda;

namespace CudaSharp.Tester;

enum TileGymConvolutionKind { Forward2D, Forward3D, Transpose2D, Transpose3D }

sealed record TileGymConvolutionProblem(
    TileGymConvolutionKind Kind, int N, int Ci, int Co, int D, int H, int W,
    int Kd, int Kh, int Kw, int Sd, int Sh, int Sw, int Pd, int Ph, int Pw,
    int Dd, int Dh, int Dw, int Groups, int OutputPaddingD = 0,
    int OutputPaddingH = 0, int OutputPaddingW = 0)
{
    public bool Transposed => Kind is TileGymConvolutionKind.Transpose2D or TileGymConvolutionKind.Transpose3D;
    public bool ThreeDimensional => Kind is TileGymConvolutionKind.Forward3D or TileGymConvolutionKind.Transpose3D;
    public int Od => OutputSize(D, Kd, Sd, Pd, Dd, OutputPaddingD);
    public int Oh => OutputSize(H, Kh, Sh, Ph, Dh, OutputPaddingH);
    public int Ow => OutputSize(W, Kw, Sw, Pw, Dw, OutputPaddingW);
    public int InputLength => N * Ci * D * H * W;
    public int WeightLength => Co * (Ci / Groups) * Kd * Kh * Kw;
    public int OutputLength => N * Co * Od * Oh * Ow;

    int OutputSize(int input, int kernel, int stride, int padding, int dilation, int outputPadding)
        => Transposed
            ? (input - 1) * stride - 2 * padding + dilation * (kernel - 1) + outputPadding + 1
            : (input + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;

    public string Header => Kind switch
    {
        TileGymConvolutionKind.Forward2D => "conv2d_with_bias_dilation_groups.cuh",
        TileGymConvolutionKind.Forward3D => "conv3d_with_bias_dilation_groups.cuh",
        TileGymConvolutionKind.Transpose2D => "conv_transpose_2d.cuh",
        _ => "conv_transpose_3d.cuh"
    };

    public string Kernel => Kind switch
    {
        TileGymConvolutionKind.Forward2D => "conv2d_implicit_gemm_kernel",
        TileGymConvolutionKind.Forward3D => "conv3d_implicit_gemm_kernel",
        TileGymConvolutionKind.Transpose2D => "conv_transpose_2d_implicit_gemm_kernel",
        _ => "conv_transpose_3d_implicit_gemm_kernel"
    };

    public string TemplateArguments(int block)
    {
        var dims = ThreeDimensional
            ? $"{N}, {Ci}, {Co}, {D}, {H}, {W}, {Od}, {Oh}, {Ow}, {Kd}, {Kh}, {Kw}, {Sd}, {Sh}, {Sw}, {Pd}, {Ph}, {Pw}, {Dd}, {Dh}, {Dw}"
            : $"{N}, {Ci}, {Co}, {H}, {W}, {Oh}, {Ow}, {Kh}, {Kw}, {Sh}, {Sw}, {Ph}, {Pw}, {Dh}, {Dw}";
        return $"{dims}, {Groups}, {block}";
    }

    public TileCppGrid Grid(int block) => new(checked((uint)((OutputLength + block - 1) / block)));

    public TileCppGrid MmaGrid() => new(
        checked((uint)((N * Od * Oh * Ow + 31) / 32)),
        checked((uint)(Groups * ((Co / Groups + MmaTileN - 1) / MmaTileN))));

    public int MmaTileN => Co / Groups <= 32 ? 32 : 64;

    public float[] Reference(float[] input, float[] weights, float[] convBias, float[] modelBias)
    {
        var output = new float[OutputLength];
        var ciGroup = Ci / Groups;
        var coGroup = Co / Groups;
        for (var n = 0; n < N; n++)
            for (var co = 0; co < Co; co++)
                for (var od = 0; od < Od; od++)
                    for (var oh = 0; oh < Oh; oh++)
                        for (var ow = 0; ow < Ow; ow++)
                        {
                            float sum = 0;
                            for (var c = 0; c < ciGroup; c++)
                                for (var kd = 0; kd < Kd; kd++)
                                    for (var kh = 0; kh < Kh; kh++)
                                        for (var kw = 0; kw < Kw; kw++)
                                        {
                                            var z = od * Sd - Pd + kd * Dd;
                                            var y = oh * Sh - Ph + kh * Dh;
                                            var x = ow * Sw - Pw + kw * Dw;
                                            if (Transposed)
                                            {
                                                z = od + Pd - kd * Dd;
                                                y = oh + Ph - kh * Dh;
                                                x = ow + Pw - kw * Dw;
                                                if (z % Sd != 0 || y % Sh != 0 || x % Sw != 0)
                                                {
                                                    continue;
                                                }
                                                z /= Sd;
                                                y /= Sh;
                                                x /= Sw;
                                            }
                                            if (z < 0 || z >= D || y < 0 || y >= H || x < 0 || x >= W)
                                            {
                                                continue;
                                            }
                                            var ic = (co / coGroup) * ciGroup + c;
                                            var weightIndex = ((((co * ciGroup + c) * Kd + kd) * Kh + kh) * Kw + kw);
                                            sum += input[((((n * Ci + ic) * D + z) * H + y) * W + x)] * weights[weightIndex];
                                        }
                            if (Kind is TileGymConvolutionKind.Forward2D or TileGymConvolutionKind.Transpose2D)
                            {
                                sum += convBias[co];
                            }
                            if (Kind is TileGymConvolutionKind.Forward2D or TileGymConvolutionKind.Forward3D)
                            {
                                sum = Math.Max(sum, 0);
                            }
                            if (Kind is TileGymConvolutionKind.Forward2D or TileGymConvolutionKind.Transpose2D)
                            {
                                sum += modelBias[co];
                            }
                            output[((((n * Co + co) * Od + od) * Oh + oh) * Ow + ow)] = sum;
                        }
        return output;
    }
}

static class TileGymConvolutionScenarios
{
    internal static TileGymConvolutionProblem[] Problems =>
    [
        new(TileGymConvolutionKind.Forward2D, 1, 4, 6, 1, 5, 6, 1, 3, 2, 1, 2, 1, 0, 1, 1, 1, 2, 1, 2),
        new(TileGymConvolutionKind.Forward3D, 1, 4, 4, 4, 5, 4, 2, 2, 2, 2, 1, 2, 1, 0, 1, 1, 2, 1, 2),
        new(TileGymConvolutionKind.Transpose2D, 1, 4, 6, 1, 4, 5, 1, 3, 2, 1, 2, 2, 0, 1, 1, 1, 2, 1, 2, 0, 1, 1),
        new(TileGymConvolutionKind.Transpose3D, 1, 4, 4, 3, 4, 3, 2, 2, 2, 2, 1, 2, 1, 0, 1, 1, 2, 1, 2, 1, 0, 1)
    ];

    internal static TileGymConvolutionProblem MmaBoundaryProblem =>
        new(TileGymConvolutionKind.Forward2D, 1, 16, 132, 1, 4, 4, 1, 3, 3,
            1, 1, 1, 0, 1, 1, 1, 1, 1, 2);

    public static IEnumerable<TileGymBenchmark> Create(TileGymRuntime runtime, TileGymOptions options)
    {
        foreach (var problem in Problems)
        {
            yield return Direct(problem);
            yield return Mma(problem, bfloat16: false);
            yield return Mma(problem, bfloat16: true);
        }
        yield return Mma(MmaBoundaryProblem, bfloat16: false);
        yield return Mma(MmaBoundaryProblem, bfloat16: true);
    }

    static float[] Values(int length, float scale)
    {
        var values = new float[length];
        for (var i = 0; i < values.Length; i++)
        {
            values[i] = ((i * 17) % 23 - 11) * scale;
        }
        return values;
    }

    internal static ushort ToBfloat16(float value)
    {
        var bits = BitConverter.SingleToUInt32Bits(value);
        if ((bits & 0x7f800000u) == 0x7f800000u)
        {
            return (ushort)((bits >> 16) | ((bits & 0x007fffffu) == 0 ? 0u : 0x40u));
        }
        return unchecked((ushort)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16));
    }

    internal static float FromBfloat16(ushort bits)
        => BitConverter.UInt32BitsToSingle((uint)bits << 16);

    internal static void ValidateMmaOutput(ReadOnlySpan<ushort> actual, ReadOnlySpan<float> expected,
        bool bfloat16, string name)
    {
        if (actual.Length != expected.Length)
        {
            throw new InvalidOperationException($"{name} validation length mismatch.");
        }
        for (var i = 0; i < actual.Length; i++)
        {
            var value = bfloat16 ? FromBfloat16(actual[i]) : (float)BitConverter.UInt16BitsToHalf(actual[i]);
            var reference = expected[i];
            if (!float.IsFinite(value) || !float.IsFinite(reference))
            {
                throw new InvalidOperationException($"{name} validation failed at {i}: {value} != {reference}.");
            }
            var referenceBits = bfloat16 ? ToBfloat16(reference) : BitConverter.HalfToUInt16Bits((Half)reference);
            var orderedActual = (actual[i] & 0x8000) != 0 ? 0x8000 - (actual[i] & 0x7fff) : 0x8000 + actual[i];
            var orderedExpected = (referenceBits & 0x8000) != 0
                ? 0x8000 - (referenceBits & 0x7fff) : 0x8000 + referenceBits;
            if (Math.Abs(orderedActual - orderedExpected) > 1)
            {
                throw new InvalidOperationException($"{name} validation failed at {i}: {value} != {reference} (more than one output ULP).");
            }
        }
    }

    static TileGymBenchmark Mma(TileGymConvolutionProblem problem, bool bfloat16)
    {
        var type = bfloat16 ? "__nv_bfloat16" : "__half";
        var name = problem.Kind switch
        {
            TileGymConvolutionKind.Forward2D => "conv2d_mma_kernel",
            TileGymConvolutionKind.Forward3D => "conv3d_mma_kernel",
            TileGymConvolutionKind.Transpose2D => "conv_transpose_2d_mma_kernel",
            _ => "conv_transpose_3d_mma_kernel"
        };
        var hasBias = !problem.ThreeDimensional;
        var signature = hasBias
            ? $"const {type}*, const {type}*, const {type}*, const {type}*, {type}*"
            : $"const {type}*, const {type}*, {type}*";
        var dimensions = problem.TemplateArguments(128);
        var blockSeparator = dimensions.LastIndexOf(", ", StringComparison.Ordinal);
        var grid = problem.MmaGrid();
        var kernel = TileGymKernel.Fixed(problem.Header, name, $"{type}, {dimensions[..blockSeparator]}", signature,
            grid, $"{(bfloat16 ? "bf16" : "fp16")},MMA=32x{problem.MmaTileN}x64");
        return new(kernel, (runtime, report) =>
        {
            using var input = runtime.Allocate<ushort>(problem.InputLength);
            using var weights = runtime.Allocate<ushort>(problem.WeightLength);
            using var bias = runtime.Allocate<ushort>(problem.Co);
            using var modelBias = runtime.Allocate<ushort>(problem.Co);
            using var output = runtime.Allocate<ushort>(problem.OutputLength);
            float[] Pack(CudaBuffer<ushort> buffer, float scale)
            {
                var values = Values(buffer.Length, scale);
                var packed = new ushort[values.Length];
                var rounded = new float[values.Length];
                for (var i = 0; i < values.Length; i++)
                {
                    packed[i] = bfloat16 ? ToBfloat16(values[i]) : BitConverter.HalfToUInt16Bits((Half)values[i]);
                    rounded[i] = bfloat16 ? FromBfloat16(packed[i]) : (float)BitConverter.UInt16BitsToHalf(packed[i]);
                }
                buffer.CopyFrom(packed);
                return rounded;
            }
            var hostInput = Pack(input, .05f);
            var hostWeights = Pack(weights, .04f);
            var hostBias = Pack(bias, .03f);
            var hostModelBias = Pack(modelBias, .02f);
            var expected = problem.Reference(hostInput, hostWeights, hostBias, hostModelBias);
            void Launch(CUfunction function, TileCppGrid grid) =>
                LaunchConvolution(runtime, function, grid, hasBias, input.Pointer, weights.Pointer,
                    bias.Pointer, modelBias.Pointer, output.Pointer);
            void Validate()
            {
                var actual = output.CopyToHost();
                ValidateMmaOutput(actual, expected, bfloat16, name);
            }
            var run = runtime.Run(kernel, Launch, Validate);
            report.Add("convolution", $"N={problem.N},Ci={problem.Ci},Co={problem.Co}," +
                $"out={problem.Od}x{problem.Oh}x{problem.Ow},groups={problem.Groups}",
                input.ByteLength + weights.ByteLength + output.ByteLength, run);
        });
    }

    static TileGymBenchmark Direct(TileGymConvolutionProblem problem)
    {
        const int block = 128;
        var hasBias = !problem.ThreeDimensional;
        var signature = hasBias
            ? "const float*, const float*, const float*, const float*, float*"
            : "const float*, const float*, float*";
        var templates = problem.TemplateArguments(block);
        var grid = problem.Grid(block);
        var kernel = TileGymKernel.Fixed(problem.Header, problem.Kernel, templates, signature, grid, $"BLOCK={block}");
        return new(kernel, (runtime, report) =>
        {
            using var input = runtime.Allocate<float>(problem.InputLength);
            using var weights = runtime.Allocate<float>(problem.WeightLength);
            using var bias = runtime.Allocate<float>(problem.Co);
            using var modelBias = runtime.Allocate<float>(problem.Co);
            using var output = runtime.Allocate<float>(problem.OutputLength);
            var hostInput = Values(input.Length, .05f);
            var hostWeights = Values(weights.Length, .04f);
            var hostBias = Values(bias.Length, .03f);
            var hostModelBias = Values(modelBias.Length, .02f);
            input.CopyFrom(hostInput);
            weights.CopyFrom(hostWeights);
            bias.CopyFrom(hostBias);
            modelBias.CopyFrom(hostModelBias);
            var expected = problem.Reference(hostInput, hostWeights, hostBias, hostModelBias);
            void Launch(CUfunction function, TileCppGrid grid) =>
                LaunchConvolution(runtime, function, grid, hasBias, input.Pointer, weights.Pointer,
                    bias.Pointer, modelBias.Pointer, output.Pointer);
            var run = runtime.Run(kernel, Launch, () => output.Validate(expected, problem.Kernel, 1e-4f, 1e-4f));
            report.Add("convolution", $"N={problem.N},Ci={problem.Ci},Co={problem.Co}," +
                $"in={problem.D}x{problem.H}x{problem.W},out={problem.Od}x{problem.Oh}x{problem.Ow}," +
                $"groups={problem.Groups}", input.ByteLength + weights.ByteLength + output.ByteLength, run);
        });
    }

    static void LaunchConvolution(TileGymRuntime runtime, CUfunction function, TileCppGrid grid, bool hasBias,
        CUdeviceptr input, CUdeviceptr weights, CUdeviceptr bias, CUdeviceptr modelBias, CUdeviceptr output)
    {
        if (hasBias)
        {
            cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                input, weights, bias, modelBias, output).Ok();
        }
        else
        {
            cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                input, weights, output).Ok();
        }
    }
}
