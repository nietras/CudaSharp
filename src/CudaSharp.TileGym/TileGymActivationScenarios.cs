using System.Collections.Generic;
using CudaSharp.TileGym;
using static CudaSharp.nvcuda;

namespace CudaSharp.Tester;

enum TileGymReluOperation
{
    Relu,
    Elu,
    LeakyRelu,
    Selu,
    Celu,
    Rrelu
}

static class TileGymActivationScenarios
{
    const float Alpha = .5f, Lower = .125f, Upper = 1f / 3;

    public static IEnumerable<TileGymBenchmark> Create(TileGymRuntime runtime, TileGymOptions options)
    {
        var count = Math.Max(1024, options.Elements);
        foreach (var operation in Enum.GetValues<TileGymReluOperation>())
        {
            yield return Relu(count, operation);
            yield return ReluBackward(count, operation);
        }
        yield return GeluActivation(count, backward: false);
        yield return GeluActivation(count, backward: true);
        var gegluRows = Math.Max(1, count / 512);
        yield return Geglu(gegluRows, 256, backward: false);
        yield return Geglu(gegluRows, 256, backward: true);
        var rows = Math.Max(1, count / 1024);
        yield return SiluAndMul(rows, 1024, backward: false);
        yield return SiluAndMul(rows, 1024, backward: true);
        yield return SiluAndMulRowWise(rows, 1024);
        yield return Swiglu(rows, 1024, backward: false);
        yield return Swiglu(rows, 1024, backward: true);
        yield return SwigluPersistent(rows, 1024);
    }

    static TileGymBenchmark Relu(int count, TileGymReluOperation operation)
    {
        const string name = "relu_activation_fwd_kernel";
        var candidates = TileGymElementwiseCandidates.For(count);
        var kernel = TileGymKernel.Tuned("activation/relu.cuh", name,
            "const float*, float*, int, float, float, float, bool",
            new TileGymElementwiseProblem(count, (int)operation), candidates);
        return new(kernel, (runtime, report) =>
        {
            using var input = runtime.Allocate<float>(count);
            using var output = runtime.Allocate<float>(count);
            var host = Values(count);
            input.CopyFrom(host);
            var expected = Array.ConvertAll(host, x => ReluReference(x, operation, Alpha, Lower, Upper, false));
            void Launch(CUfunction function, TileCppGrid grid)
            {
                byte training = 0;
                cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                    input.Pointer, output.Pointer, count, Alpha, Lower, Upper, training).Ok();
            }
            var run = runtime.Run(kernel, Launch, () => output.Validate(expected, name, 2e-4f, 2e-4f));
            report.Add("activation", $"{count},OP={(int)operation} ({operation})",
                input.ByteLength + output.ByteLength, run);
        });
    }

    static TileGymBenchmark ReluBackward(int count, TileGymReluOperation operation)
    {
        const string name = "relu_activation_bwd_kernel";
        var candidates = TileGymElementwiseCandidates.For(count);
        var kernel = TileGymKernel.Tuned("activation/relu.cuh", name,
            "const float*, const float*, float*, int, float, float, float, bool",
            new TileGymElementwiseProblem(count, (int)operation), candidates);
        return new(kernel, (runtime, report) =>
        {
            using var dy = runtime.Allocate<float>(count);
            using var x = runtime.Allocate<float>(count);
            using var dx = runtime.Allocate<float>(count);
            var hx = Values(count);
            var hdy = Ones(count);
            x.CopyFrom(hx);
            dy.CopyFrom(hdy);
            var expected = Array.ConvertAll(hx, value => ReluReference(value, operation, Alpha, Lower, Upper, true));
            void Launch(CUfunction function, TileCppGrid grid)
            {
                byte training = 0;
                cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                    dy.Pointer, x.Pointer, dx.Pointer, count, Alpha, Lower, Upper, training).Ok();
            }
            var run = runtime.Run(kernel, Launch, () => dx.Validate(expected, name, 2e-4f, 2e-4f));
            report.Add("activation", $"{count},OP={(int)operation} ({operation})",
                dy.ByteLength + x.ByteLength + dx.ByteLength, run);
        });
    }

    static TileGymBenchmark GeluActivation(int count, bool backward)
    {
        var name = backward ? "gelu_bwd_kernel" : "gelu_fwd_kernel";
        var signature = backward ? "const float*, const float*, float*, int" : "const float*, float*, int";
        var candidates = TileGymElementwiseCandidates.For(count);
        var kernel = TileGymKernel.Tuned("activation/gelu.cuh", name, signature,
            new TileGymElementwiseProblem(count, 0), candidates);
        return new(kernel, (runtime, report) =>
        {
            using var x = runtime.Allocate<float>(count);
            using var output = runtime.Allocate<float>(count);
            using var dy = backward ? runtime.Allocate<float>(count) : null;
            var hx = Values(count);
            x.CopyFrom(hx);
            if (dy is not null)
            {
                var hdy = Ones(count);
                dy.CopyFrom(hdy);
            }
            void Launch(CUfunction function, TileCppGrid grid)
            {
                if (backward)
                {
                    cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                        dy!.Pointer, x.Pointer, output.Pointer, count).Ok();
                }
                else
                {
                    cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                        x.Pointer, output.Pointer, count).Ok();
                }
            }
            var expected = Array.ConvertAll(hx, value => backward ? GeluDerivative(value) : Gelu(value));
            var run = runtime.Run(kernel, Launch, () => output.Validate(expected, name, 4e-4f, 4e-4f));
            report.Add("activation", $"{count}", x.ByteLength + output.ByteLength + (dy?.ByteLength ?? 0), run);
        });
    }

    static TileGymBenchmark Geglu(int rows, int hidden, bool backward)
    {
        var name = backward ? "geglu_bwd_kernel" : "geglu_fwd_kernel";
        var signature = backward ? "float*, const float*, const float*, int, int, int, int" :
            "const float*, float*, int, int, int, int";
        var problem = new TileGymElementwiseProblem(rows * hidden, 0);
        var candidates = TileGymElementwiseCandidates.For(problem.Count);
        var kernel = TileGymKernel.Tuned("activation/geglu.cuh", name, signature, problem, candidates);
        return new(kernel, (runtime, report) =>
        {
            using var x = runtime.Allocate<float>(rows * hidden * 2);
            using var output = runtime.Allocate<float>(backward ? rows * hidden * 2 : rows * hidden);
            using var dy = backward ? runtime.Allocate<float>(rows * hidden) : null;
            var hx = Values(x.Length);
            x.CopyFrom(hx);
            if (dy is not null)
            {
                var hdy = Ones(dy.Length);
                dy.CopyFrom(hdy);
            }
            void Launch(CUfunction function, TileCppGrid grid)
            {
                var elements = rows * hidden;
                if (backward)
                {
                    cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                        output.Pointer, dy!.Pointer, x.Pointer, hidden, hidden * 2, hidden, elements).Ok();
                }
                else
                {
                    cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                        x.Pointer, output.Pointer, hidden, hidden * 2, hidden, elements).Ok();
                }
            }
            var expected = new float[output.Length];
            for (var row = 0; row < rows; row++)
            {
                for (var col = 0; col < hidden; col++)
                {
                    var a = hx[row * hidden * 2 + col];
                    var b = hx[row * hidden * 2 + hidden + col];
                    if (backward)
                    {
                        expected[row * hidden * 2 + col] = Gelu(b);
                        expected[row * hidden * 2 + hidden + col] = a * GeluDerivative(b);
                    }
                    else
                    {
                        expected[row * hidden + col] = a * Gelu(b);
                    }
                }
            }
            var run = runtime.Run(kernel, Launch, () => output.Validate(expected, name, 5e-4f, 5e-4f));
            report.Add("activation", $"{rows}x{hidden * 2}",
                x.ByteLength + output.ByteLength + (dy?.ByteLength ?? 0), run);
        });
    }

    static TileGymBenchmark SiluAndMul(int rows, int hidden, bool backward)
    {
        var name = backward ? "silu_and_mul_backward_kernel" : "silu_and_mul_kernel";
        var signature = backward ? "const float*, const float*, float*, int, int" : "const float*, float*, int, int";
        var kernel = TileGymKernel.Fixed("silu_and_mul.cuh", name, $"float, {hidden}", signature,
            new TileCppGrid((uint)rows), $"float,BLOCK_SIZE={hidden}");
        return new(kernel, (runtime, report) =>
        {
            using var input = runtime.Allocate<float>(rows * hidden * 2);
            using var output = runtime.Allocate<float>(backward ? input.Length : rows * hidden);
            using var grad = backward ? runtime.Allocate<float>(rows * hidden) : null;
            var host = Values(input.Length);
            input.CopyFrom(host);
            if (grad is not null)
            {
                var hgrad = Ones(grad.Length);
                grad.CopyFrom(hgrad);
            }
            void Launch(CUfunction function, TileCppGrid grid)
            {
                if (backward)
                {
                    cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                        grad!.Pointer, input.Pointer, output.Pointer, hidden * 2, hidden).Ok();
                }
                else
                {
                    cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                        input.Pointer, output.Pointer, hidden * 2, hidden).Ok();
                }
            }
            var expected = SiluReference(host, rows, hidden, backward);
            var run = runtime.Run(kernel, Launch, () => output.Validate(expected, name, 4e-4f, 4e-4f));
            report.Add("fused", $"{rows}x{hidden * 2}",
                input.ByteLength + output.ByteLength + (grad?.ByteLength ?? 0), run);
        });
    }

    static TileGymBenchmark SiluAndMulRowWise(int rows, int hidden)
    {
        const string name = "silu_and_mul_kernel_row_wise";
        var n = hidden * 2;
        var kernel = TileGymKernel.Fixed("silu_and_mul.cuh", name,
            $"float, {n}, {hidden}, {hidden}, {n}, {hidden}", "float*, float*",
            new TileCppGrid((uint)rows), $"float,N={n},HIDDEN_SIZE={hidden}");
        return new(kernel, (runtime, report) =>
        {
            using var input = runtime.Allocate<float>(rows * n);
            using var output = runtime.Allocate<float>(rows * hidden);
            var host = Values(input.Length);
            input.CopyFrom(host);
            void Launch(CUfunction function, TileCppGrid grid) =>
                cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                    input.Pointer, output.Pointer).Ok();
            var expected = SiluReference(host, rows, hidden, false);
            var run = runtime.Run(kernel, Launch, () => output.Validate(expected, name, 4e-4f, 4e-4f));
            report.Add("fused", $"{rows}x{n}", input.ByteLength + output.ByteLength, run);
        });
    }

    static TileGymBenchmark Swiglu(int rows, int columns, bool backward)
    {
        var name = backward ? "swiglu_backward_kernel" : "swiglu_forward_kernel_gather";
        var signature = backward ? "const float*, const float*, const float*, float*, float*, int, int" :
            "const float*, const float*, float*, int, int";
        var kernel = TileGymKernel.Fixed("swiglu.cuh", name, $"float, {columns}", signature,
            new TileCppGrid((uint)rows), $"float,BLOCK_SIZE={columns}");
        return new(kernel, (runtime, report) =>
        {
            using var a = runtime.Allocate<float>(rows * columns);
            using var b = runtime.Allocate<float>(rows * columns);
            using var c = runtime.Allocate<float>(rows * columns);
            using var second = backward ? runtime.Allocate<float>(rows * columns) : null;
            using var dc = backward ? runtime.Allocate<float>(rows * columns) : null;
            var ha = Values(a.Length);
            var hb = Array.ConvertAll(ha, static x => x * .75f + .25f);
            a.CopyFrom(ha);
            b.CopyFrom(hb);
            if (dc is not null)
            {
                var hdc = Ones(dc.Length);
                dc.CopyFrom(hdc);
            }
            void Launch(CUfunction function, TileCppGrid grid)
            {
                if (backward)
                {
                    cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                        dc!.Pointer, a.Pointer, b.Pointer, c.Pointer, second!.Pointer, columns, columns).Ok();
                }
                else
                {
                    cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                        a.Pointer, b.Pointer, c.Pointer, columns, columns).Ok();
                }
            }
            var expected = new float[c.Length];
            var expectedSecond = new float[c.Length];
            for (var i = 0; i < expected.Length; i++)
            {
                var sig = Sigmoid(ha[i]);
                var silu = ha[i] * sig;
                expected[i] = backward ? hb[i] * (silu * (1 - sig) + sig) : silu * hb[i];
                expectedSecond[i] = silu;
            }
            void Validate()
            {
                c.Validate(expected, name, 4e-4f, 4e-4f);
                second?.Validate(expectedSecond, name, 4e-4f, 4e-4f);
            }
            var run = runtime.Run(kernel, Launch, Validate);
            report.Add("fused", $"{rows}x{columns}", a.ByteLength + b.ByteLength + c.ByteLength +
                (second?.ByteLength ?? 0) + (dc?.ByteLength ?? 0), run);
        });
    }

    static TileGymBenchmark SwigluPersistent(int rows, int columns)
    {
        const string name = "swiglu_forward_kernel_pv";
        var kernel = TileGymKernel.Fixed("swiglu.cuh", name, $"float, {rows}, {columns}, {columns}, 4",
            "float*, float*, float*", new TileCppGrid((uint)rows), $"float,BLOCK_SIZE={columns},OCCUPANCY=4");
        return new(kernel, (runtime, report) =>
        {
            using var a = runtime.Allocate<float>(rows * columns);
            using var b = runtime.Allocate<float>(rows * columns);
            using var c = runtime.Allocate<float>(rows * columns);
            var ha = Values(a.Length);
            var hb = Array.ConvertAll(ha, static x => x * .75f + .25f);
            a.CopyFrom(ha);
            b.CopyFrom(hb);
            void Launch(CUfunction function, TileCppGrid grid) =>
                cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                    a.Pointer, b.Pointer, c.Pointer).Ok();
            var expected = new float[c.Length];
            for (var i = 0; i < expected.Length; i++)
            {
                expected[i] = ha[i] * Sigmoid(ha[i]) * hb[i];
            }
            var run = runtime.Run(kernel, Launch, () => c.Validate(expected, name, 4e-4f, 4e-4f));
            report.Add("fused", $"{rows}x{columns}", a.ByteLength + b.ByteLength + c.ByteLength, run);
        });
    }

    static float[] Values(int count)
    {
        var values = new float[count];
        for (var i = 0; i < count; i++)
        {
            values[i] = (i % 257 - 128) / 64f;
        }
        return values;
    }

    static float[] Ones(int count)
    {
        var values = new float[count];
        Array.Fill(values, 1f);
        return values;
    }
    internal static float ReluReference(float x, TileGymReluOperation operation, float alpha, float lower,
        float upper, bool backward)
    {
        const float seluScale = 1.0507009873554805f;
        const float seluAlpha = 1.6732632423543772f;
        if (backward)
        {
            return operation switch
            {
                TileGymReluOperation.Relu => x > 0 ? 1f : 0f,
                TileGymReluOperation.Elu => x > 0 ? 1f : alpha * MathF.Exp(x),
                TileGymReluOperation.LeakyRelu => x > 0 ? 1f : alpha,
                TileGymReluOperation.Selu => seluScale * (x > 0 ? 1f : seluAlpha * MathF.Exp(x)),
                TileGymReluOperation.Celu => x > 0 ? 1f : MathF.Exp(x / alpha),
                TileGymReluOperation.Rrelu => x > 0 ? 1f : (lower + upper) * .5f,
                _ => throw new ArgumentOutOfRangeException(nameof(operation))
            };
        }
        return operation switch
        {
            TileGymReluOperation.Relu => Math.Max(x, 0),
            TileGymReluOperation.Elu => x > 0 ? x : alpha * (MathF.Exp(x) - 1f),
            TileGymReluOperation.LeakyRelu => x > 0 ? x : alpha * x,
            TileGymReluOperation.Selu => seluScale * (x > 0 ? x : seluAlpha * (MathF.Exp(x) - 1f)),
            TileGymReluOperation.Celu => x > 0 ? x : alpha * (MathF.Exp(x / alpha) - 1f),
            TileGymReluOperation.Rrelu => x > 0 ? x : (lower + upper) * .5f * x,
            _ => throw new ArgumentOutOfRangeException(nameof(operation))
        };
    }
    static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
    static float Gelu(float x) => .5f * x * (1f + MathF.Tanh(.7978845608028654f * (x + .044715f * x * x * x)));
    static float GeluDerivative(float x) =>
        .5f * (1f + MathF.Tanh(.7978845608028654f * (x + .044715f * x * x * x))) +
        x * .3989422804014327f * MathF.Exp(-.5f * x * x);
    static float[] SiluReference(float[] input, int rows, int hidden, bool backward)
    {
        var result = new float[backward ? input.Length : rows * hidden];
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < hidden; c++)
            {
                var a = input[r * hidden * 2 + c];
                var b = input[r * hidden * 2 + hidden + c];
                var sig = Sigmoid(a);
                var silu = a * sig;
                if (backward)
                {
                    result[r * hidden * 2 + c] = b * (sig + silu * (1 - sig));
                    result[r * hidden * 2 + hidden + c] = silu;
                }
                else
                {
                    result[r * hidden + c] = silu * b;
                }
            }
        }
        return result;
    }
}
