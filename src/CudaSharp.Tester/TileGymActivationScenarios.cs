using System;
using CudaSharp.Tile;

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
    static readonly TileCppConfig Config = new([]);

    public static void RunAll(TileGymRuntime runtime, TileGymReport report, int elementCount = 1 << 20)
    {
        var count = Math.Max(1024, elementCount);
        foreach (var operation in Enum.GetValues<TileGymReluOperation>())
        {
            RunRelu(runtime, report, count, operation);
            RunReluBackward(runtime, report, count, operation);
        }
        RunGelu(runtime, report, count, backward: false);
        RunGelu(runtime, report, count, backward: true);
        RunGeglu(runtime, report, Math.Max(1, count / 512), 256, backward: false);
        RunGeglu(runtime, report, Math.Max(1, count / 512), 256, backward: true);
        RunSiluAndMul(runtime, report, Math.Max(1, count / 1024), 1024, backward: false);
        RunSiluAndMul(runtime, report, Math.Max(1, count / 1024), 1024, backward: true);
        RunSiluAndMulRowWise(runtime, report, Math.Max(1, count / 1024), 1024);
        RunSwiglu(runtime, report, Math.Max(1, count / 1024), 1024, backward: false);
        RunSwiglu(runtime, report, Math.Max(1, count / 1024), 1024, backward: true);
        RunSwigluPersistent(runtime, report, 1, 1024);
    }

    public static unsafe void RunRelu(TileGymRuntime runtime, TileGymReport report, int elementCount,
        TileGymReluOperation operation)
    {
        const string name = "relu_activation_fwd_kernel";
        var problem = new TileGymElementwiseProblem(elementCount, (int)operation);
        const float alpha = .5f, lower = .125f, upper = 1f / 3;
        using var input = runtime.Allocate<float>(elementCount);
        using var output = runtime.Allocate<float>(elementCount);
        var host = Values(elementCount);
        input.CopyFrom(host);
        void Launch(TileCppKernel kernel, TileCppConfig config, TileCppGrid grid)
        {
            var x = input.Pointer.Value;
            var y = output.Pointer.Value;
            var n = elementCount;
            var a = alpha;
            var lo = lower;
            var hi = upper;
            byte training = 0;
            var args = stackalloc IntPtr[] { (IntPtr)(&x), (IntPtr)(&y), (IntPtr)(&n), (IntPtr)(&a),
                (IntPtr)(&lo), (IntPtr)(&hi), (IntPtr)(&training) };
            kernel.Launch(config, grid, runtime.Stream, new(args, 7));
        }
        var expected = Array.ConvertAll(host, x => ReluReference(x, operation, alpha, lower, upper, false));
        var tuned = TileGymTuning.Tune(runtime, problem, TileGymElementwiseCandidates.For(elementCount),
            "activation/relu.cuh", name, "const float*, float*, int, float, float, float, bool",
            static (p, candidate) => p.TemplateArguments(candidate),
            static (p, candidate) => p.Grid(candidate), Launch,
            validate: _ => TileGymKernel.Validate(output.CopyToHost(), expected, name, 2e-4f, 2e-4f));
        void LaunchSelected() => Launch(tuned.Kernel, tuned.Candidate.CompilerConfig, tuned.Grid);
        var timing = TileGymKernel.Measure(runtime, LaunchSelected);
        TileGymKernel.Validate(output.CopyToHost(), expected, name, 2e-4f, 2e-4f);
        TileGymKernel.Report(report, "activation", name, $"{elementCount},OP={(int)operation} ({operation})",
            input.ByteLength + output.ByteLength, timing, tuned);
    }

    static unsafe void RunReluBackward(TileGymRuntime runtime, TileGymReport report, int count,
        TileGymReluOperation operation)
    {
        const string name = "relu_activation_bwd_kernel";
        var problem = new TileGymElementwiseProblem(count, (int)operation);
        const float alpha = .5f, lower = .125f, upper = 1f / 3;
        using var dy = runtime.Allocate<float>(count);
        using var x = runtime.Allocate<float>(count);
        using var dx = runtime.Allocate<float>(count);
        var hx = Values(count);
        var hdy = Ones(count);
        x.CopyFrom(hx);
        dy.CopyFrom(hdy);
        void Launch(TileCppKernel kernel, TileCppConfig config, TileCppGrid grid)
        {
            var pdy = dy.Pointer.Value;
            var px = x.Pointer.Value;
            var pdx = dx.Pointer.Value;
            var n = count;
            var a = alpha;
            var lo = lower;
            var hi = upper;
            byte training = 0;
            var args = stackalloc IntPtr[] { (IntPtr)(&pdy), (IntPtr)(&px), (IntPtr)(&pdx), (IntPtr)(&n),
                (IntPtr)(&a), (IntPtr)(&lo), (IntPtr)(&hi), (IntPtr)(&training) };
            kernel.Launch(config, grid, runtime.Stream, new(args, 8));
        }
        var expected = Array.ConvertAll(hx, value => ReluReference(value, operation, alpha, lower, upper, true));
        var tuned = TileGymTuning.Tune(runtime, problem, TileGymElementwiseCandidates.For(count),
            "activation/relu.cuh", name, "const float*, const float*, float*, int, float, float, float, bool",
            static (p, candidate) => p.TemplateArguments(candidate),
            static (p, candidate) => p.Grid(candidate), Launch,
            validate: _ => TileGymKernel.Validate(dx.CopyToHost(), expected, name, 2e-4f, 2e-4f));
        void LaunchSelected() => Launch(tuned.Kernel, tuned.Candidate.CompilerConfig, tuned.Grid);
        var timing = TileGymKernel.Measure(runtime, LaunchSelected);
        TileGymKernel.Validate(dx.CopyToHost(), expected, name, 2e-4f, 2e-4f);
        TileGymKernel.Report(report, "activation", name, $"{count},OP={(int)operation} ({operation})",
            dy.ByteLength + x.ByteLength + dx.ByteLength, timing, tuned);
    }

    static unsafe void RunGelu(TileGymRuntime runtime, TileGymReport report, int count, bool backward)
    {
        var name = backward ? "gelu_bwd_kernel" : "gelu_fwd_kernel";
        var signature = backward ? "const float*, const float*, float*, int" : "const float*, float*, int";
        var problem = new TileGymElementwiseProblem(count, 0);
        using var x = runtime.Allocate<float>(count);
        using var output = runtime.Allocate<float>(count);
        using var dy = backward ? runtime.Allocate<float>(count) : null;
        var hx = Values(count);
        x.CopyFrom(hx);
        dy?.CopyFrom(Ones(count));
        void Launch(TileCppKernel kernel, TileCppConfig config, TileCppGrid grid)
        {
            var px = x.Pointer.Value;
            var po = output.Pointer.Value;
            var n = count;
            if (backward)
            {
                var pdy = dy!.Pointer.Value;
                var args = stackalloc IntPtr[] { (IntPtr)(&pdy), (IntPtr)(&px), (IntPtr)(&po), (IntPtr)(&n) };
                kernel.Launch(config, grid, runtime.Stream, new(args, 4));
            }
            else
            {
                var args = stackalloc IntPtr[] { (IntPtr)(&px), (IntPtr)(&po), (IntPtr)(&n) };
                kernel.Launch(config, grid, runtime.Stream, new(args, 3));
            }
        }
        var expected = Array.ConvertAll(hx, value => backward ? GeluDerivative(value) : Gelu(value));
        var tuned = TileGymTuning.Tune(runtime, problem, TileGymElementwiseCandidates.For(count),
            "activation/gelu.cuh", name, signature,
            static (p, candidate) => p.TemplateArguments(candidate),
            static (p, candidate) => p.Grid(candidate), Launch,
            validate: _ => TileGymKernel.Validate(output.CopyToHost(), expected, name, 4e-4f, 4e-4f));
        void LaunchSelected() => Launch(tuned.Kernel, tuned.Candidate.CompilerConfig, tuned.Grid);
        var timing = TileGymKernel.Measure(runtime, LaunchSelected);
        TileGymKernel.Validate(output.CopyToHost(), expected, name, 4e-4f, 4e-4f);
        var bytes = x.ByteLength + output.ByteLength + (dy?.ByteLength ?? 0);
        TileGymKernel.Report(report, "activation", name, $"{count}", bytes, timing, tuned);
    }

    static unsafe void RunGeglu(TileGymRuntime runtime, TileGymReport report, int rows, int hidden, bool backward)
    {
        var name = backward ? "geglu_bwd_kernel" : "geglu_fwd_kernel";
        var signature = backward ? "float*, const float*, const float*, int, int, int, int" :
            "const float*, float*, int, int, int, int";
        var problem = new TileGymElementwiseProblem(rows * hidden, 0);
        using var x = runtime.Allocate<float>(rows * hidden * 2);
        using var output = runtime.Allocate<float>(backward ? rows * hidden * 2 : rows * hidden);
        using var dy = backward ? runtime.Allocate<float>(rows * hidden) : null;
        var hx = Values(x.Length);
        x.CopyFrom(hx);
        dy?.CopyFrom(Ones(dy.Length));
        void Launch(TileCppKernel kernel, TileCppConfig config, TileCppGrid grid)
        {
            var px = x.Pointer.Value;
            var po = output.Pointer.Value;
            var pdy = dy?.Pointer.Value ?? IntPtr.Zero;
            var n = hidden;
            var xs = hidden * 2;
            var ys = hidden;
            var elements = rows * hidden;
            if (backward)
            {
                var args = stackalloc IntPtr[] { (IntPtr)(&po), (IntPtr)(&pdy), (IntPtr)(&px), (IntPtr)(&n),
                    (IntPtr)(&xs), (IntPtr)(&ys), (IntPtr)(&elements) };
                kernel.Launch(config, grid, runtime.Stream, new(args, 7));
            }
            else
            {
                var args = stackalloc IntPtr[] { (IntPtr)(&px), (IntPtr)(&po), (IntPtr)(&n),
                    (IntPtr)(&xs), (IntPtr)(&ys), (IntPtr)(&elements) };
                kernel.Launch(config, grid, runtime.Stream, new(args, 6));
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
        var tuned = TileGymTuning.Tune(runtime, problem, TileGymElementwiseCandidates.For(problem.Count),
            "activation/geglu.cuh", name, signature,
            static (p, candidate) => p.TemplateArguments(candidate),
            static (p, candidate) => p.Grid(candidate), Launch,
            validate: _ => TileGymKernel.Validate(output.CopyToHost(), expected, name, 5e-4f, 5e-4f));
        void LaunchSelected() => Launch(tuned.Kernel, tuned.Candidate.CompilerConfig, tuned.Grid);
        var timing = TileGymKernel.Measure(runtime, LaunchSelected);
        TileGymKernel.Validate(output.CopyToHost(), expected, name, 5e-4f, 5e-4f);
        TileGymKernel.Report(report, "activation", name, $"{rows}x{hidden * 2}",
            x.ByteLength + output.ByteLength + (dy?.ByteLength ?? 0), timing, tuned);
    }

    static unsafe void RunSiluAndMul(TileGymRuntime runtime, TileGymReport report, int rows, int hidden, bool backward)
    {
        var name = backward ? "silu_and_mul_backward_kernel" : "silu_and_mul_kernel";
        var signature = backward ? "const float*, const float*, float*, int, int" : "const float*, float*, int, int";
        using var kernel = TileGymKernel.Create(runtime, "silu_and_mul.cuh", name, $"float, {hidden}", signature);
        using var input = runtime.Allocate<float>(rows * hidden * 2);
        using var output = runtime.Allocate<float>(backward ? input.Length : rows * hidden);
        using var grad = backward ? runtime.Allocate<float>(rows * hidden) : null;
        var host = Values(input.Length);
        input.CopyFrom(host);
        grad?.CopyFrom(Ones(grad.Length));
        void Launch()
        {
            var pi = input.Pointer.Value;
            var po = output.Pointer.Value;
            var pg = grad?.Pointer.Value ?? IntPtr.Zero;
            var stride = hidden * 2;
            var h = hidden;
            if (backward)
            {
                var args = stackalloc IntPtr[]
                {
                    (IntPtr)(&pg), (IntPtr)(&pi), (IntPtr)(&po), (IntPtr)(&stride), (IntPtr)(&h)
                };
                kernel.Launch(Config, new((uint)rows), runtime.Stream, new(args, 5));
            }
            else
            {
                var args = stackalloc IntPtr[]
                {
                    (IntPtr)(&pi), (IntPtr)(&po), (IntPtr)(&stride), (IntPtr)(&h)
                };
                kernel.Launch(Config, new((uint)rows), runtime.Stream, new(args, 4));
            }
        }
        var timing = TileGymKernel.Measure(runtime, Launch);
        var expected = SiluReference(host, rows, hidden, backward);
        TileGymKernel.Validate(output.CopyToHost(), expected, name, 4e-4f, 4e-4f);
        TileGymKernel.Report(report, "fused", name, $"{rows}x{hidden * 2}", $"float,BLOCK_SIZE={hidden}",
            input.ByteLength + output.ByteLength + (grad?.ByteLength ?? 0), timing);
    }

    static unsafe void RunSiluAndMulRowWise(TileGymRuntime runtime, TileGymReport report, int rows, int hidden)
    {
        var n = hidden * 2;
        using var kernel = TileGymKernel.Create(runtime, "silu_and_mul.cuh", "silu_and_mul_kernel_row_wise",
            $"float, {n}, {hidden}, {hidden}, {n}, {hidden}", "float*, float*");
        using var input = runtime.Allocate<float>(rows * n);
        using var output = runtime.Allocate<float>(rows * hidden);
        var host = Values(input.Length);
        input.CopyFrom(host);
        void Launch()
        {
            var pi = input.Pointer.Value;
            var po = output.Pointer.Value;
            var args = stackalloc IntPtr[] { (IntPtr)(&pi), (IntPtr)(&po) };
            kernel.Launch(Config, new((uint)rows), runtime.Stream, new(args, 2));
        }
        var timing = TileGymKernel.Measure(runtime, Launch);
        TileGymKernel.Validate(output.CopyToHost(), SiluReference(host, rows, hidden, false), "silu_and_mul_kernel_row_wise", 4e-4f, 4e-4f);
        TileGymKernel.Report(
            report,
            "fused",
            "silu_and_mul_kernel_row_wise",
            $"{rows}x{n}",
            $"float,N={n},HIDDEN_SIZE={hidden}",
            input.ByteLength + output.ByteLength,
            timing);
    }

    static unsafe void RunSwiglu(TileGymRuntime runtime, TileGymReport report, int rows, int columns, bool backward)
    {
        var name = backward ? "swiglu_backward_kernel" : "swiglu_forward_kernel_gather";
        var signature = backward ? "const float*, const float*, const float*, float*, float*, int, int" :
            "const float*, const float*, float*, int, int";
        using var kernel = TileGymKernel.Create(runtime, "swiglu.cuh", name, $"float, {columns}", signature);
        using var a = runtime.Allocate<float>(rows * columns);
        using var b = runtime.Allocate<float>(rows * columns);
        using var c = runtime.Allocate<float>(rows * columns);
        using var second = backward ? runtime.Allocate<float>(rows * columns) : null;
        using var dc = backward ? runtime.Allocate<float>(rows * columns) : null;
        var ha = Values(a.Length);
        var hb = Array.ConvertAll(ha, static x => x * .75f + .25f);
        a.CopyFrom(ha);
        b.CopyFrom(hb);
        dc?.CopyFrom(Ones(dc.Length));
        void Launch()
        {
            var pa = a.Pointer.Value;
            var pb = b.Pointer.Value;
            var pc = c.Pointer.Value;
            var ps = second?.Pointer.Value ?? IntPtr.Zero;
            var pdc = dc?.Pointer.Value ?? IntPtr.Zero;
            var stride = columns;
            var cols = columns;
            if (backward)
            {
                var args = stackalloc IntPtr[]
                {
                    (IntPtr)(&pdc), (IntPtr)(&pa), (IntPtr)(&pb), (IntPtr)(&pc),
                    (IntPtr)(&ps), (IntPtr)(&stride), (IntPtr)(&cols)
                };
                kernel.Launch(Config, new((uint)rows), runtime.Stream, new(args, 7));
            }
            else
            {
                var args = stackalloc IntPtr[]
                {
                    (IntPtr)(&pa), (IntPtr)(&pb), (IntPtr)(&pc), (IntPtr)(&cols),
                    (IntPtr)(&stride)
                };
                kernel.Launch(Config, new((uint)rows), runtime.Stream, new(args, 5));
            }
        }
        var timing = TileGymKernel.Measure(runtime, Launch);
        var expected = new float[c.Length];
        var expectedSecond = backward ? new float[c.Length] : null;
        for (var i = 0; i < expected.Length; i++)
        {
            var sig = Sigmoid(ha[i]);
            var silu = ha[i] * sig;
            expected[i] = backward ? hb[i] * (silu * (1 - sig) + sig) : silu * hb[i];
            if (backward)
            {
                expectedSecond![i] = silu;
            }
        }
        TileGymKernel.Validate(c.CopyToHost(), expected, name, 4e-4f, 4e-4f);
        if (backward)
        {
            TileGymKernel.Validate(second!.CopyToHost(), expectedSecond!, name, 4e-4f, 4e-4f);
        }
        TileGymKernel.Report(
            report, "fused", name, $"{rows}x{columns}", $"float,BLOCK_SIZE={columns}",
            a.ByteLength + b.ByteLength + c.ByteLength +
            (second?.ByteLength ?? 0) + (dc?.ByteLength ?? 0),
            timing);
    }

    static unsafe void RunSwigluPersistent(TileGymRuntime runtime, TileGymReport report, int rows, int columns)
    {
        using var kernel = TileGymKernel.Create(runtime, "swiglu.cuh", "swiglu_forward_kernel_pv",
            $"float, {rows}, {columns}, {columns}, 4", "float*, float*, float*");
        using var a = runtime.Allocate<float>(rows * columns);
        using var b = runtime.Allocate<float>(rows * columns);
        using var c = runtime.Allocate<float>(rows * columns);
        var ha = Values(a.Length);
        var hb = Array.ConvertAll(ha, static x => x * .75f + .25f);
        a.CopyFrom(ha);
        b.CopyFrom(hb);
        void Launch()
        {
            var pa = a.Pointer.Value;
            var pb = b.Pointer.Value;
            var pc = c.Pointer.Value;
            var args = stackalloc IntPtr[] { (IntPtr)(&pa), (IntPtr)(&pb), (IntPtr)(&pc) };
            kernel.Launch(Config, new((uint)rows, 1), runtime.Stream, new(args, 3));
        }
        var timing = TileGymKernel.Measure(runtime, Launch);
        var expected = new float[c.Length];
        for (var i = 0; i < expected.Length; i++)
        {
            expected[i] = ha[i] * Sigmoid(ha[i]) * hb[i];
        }
        TileGymKernel.Validate(c.CopyToHost(), expected, "swiglu_forward_kernel_pv", 4e-4f, 4e-4f);
        TileGymKernel.Report(
            report, "fused", "swiglu_forward_kernel_pv", $"{rows}x{columns}",
            $"float,BLOCK_SIZE={columns},OCCUPANCY=4",
            a.ByteLength + b.ByteLength + c.ByteLength, timing);
    }

    static TileCppGrid Grid(int count, int block) => new((uint)((count + block - 1) / block));
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
