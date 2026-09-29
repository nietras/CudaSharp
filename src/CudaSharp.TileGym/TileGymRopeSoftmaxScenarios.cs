using System.Collections.Generic;
using CudaSharp.TileGym;
using static CudaSharp.nvcuda;

namespace CudaSharp.Tester;

static class TileGymRopeSoftmaxScenarios
{
    public static IEnumerable<TileGymBenchmark> Create(TileGymRuntime runtime, TileGymOptions options)
    {
        yield return Rope(backward: false);
        yield return Rope(backward: true);
        yield return Softmax(online: false, backward: false);
        yield return Softmax(online: false, backward: true);
        yield return Softmax(online: true, backward: false);
        yield return Softmax(online: true, backward: true);
    }

    static TileGymBenchmark Rope(bool backward)
    {
        const int batch = 1;
        const int qHeads = 32;
        const int kHeads = 8;
        const int sequence = 16;
        const int head = 128;
        const int half = 64;
        var name = backward ? "rope_backward_kernel" : "rope_kernel";
        var templates = $"float, float, float, {batch}, {qHeads}, {kHeads}, {qHeads}, {kHeads}, {half}, {half}, " +
            $"{head}, 1, {sequence}";
        if (!backward)
        {
            templates += $", {qHeads * sequence * head}, {sequence * head}, {head}, " +
                $"{kHeads * sequence * head}, {sequence * head}, {head}";
        }
        var kernel = TileGymKernel.Fixed("rope.cuh", name, templates, "float*, float*, const float*, const float*",
            new TileCppGrid(batch * sequence));
        return new(kernel, (runtime, report) =>
        {
            using var q = runtime.Allocate<float>(batch * qHeads * sequence * head);
            using var k = runtime.Allocate<float>(batch * kHeads * sequence * head);
            using var cos = runtime.Allocate<float>(sequence * 2 * half);
            using var sin = runtime.Allocate<float>(sequence * 2 * half);
            var hq = Values(q.Length);
            var hk = Values(k.Length);
            var hc = new float[cos.Length];
            var hs = new float[sin.Length];
            for (var s = 0; s < sequence; s++)
            {
                for (var p = 0; p < 2; p++)
                {
                    for (var d = 0; d < half; d++)
                    {
                        var angle = (s + 1) * (d + 1) / 1024f;
                        hc[(s * 2 + p) * half + d] = MathF.Cos(angle);
                        hs[(s * 2 + p) * half + d] = MathF.Sin(angle);
                    }
                }
            }
            cos.CopyFrom(hc);
            sin.CopyFrom(hs);
            void Reset()
            {
                q.CopyFrom(hq);
                k.CopyFrom(hk);
            }
            void Launch(CUfunction function, TileCppGrid grid) =>
                cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                    q.Pointer, k.Pointer, cos.Pointer, sin.Pointer).Ok();
            var eq = Rotate(hq, hc, hs, qHeads, sequence, head, backward);
            var ek = Rotate(hk, hc, hs, kHeads, sequence, head, backward);
            void Validate()
            {
                q.Validate(eq, name, 5e-4f, 5e-4f);
                k.Validate(ek, name, 5e-4f, 5e-4f);
            }
            var run = runtime.Run(kernel, Launch, Validate, Reset);
            report.Add("rope", $"B={batch},QH={qHeads},KH={kHeads},S={sequence},D={head}",
                q.ByteLength + k.ByteLength + cos.ByteLength + sin.ByteLength, run);
        });
    }

    static TileGymBenchmark Softmax(bool online, bool backward)
    {
        var rows = online ? 1 << 14 : 1 << 16;
        var columns = online ? 1025 : 256;
        var name = (backward, online) switch
        {
            (true, true) => "online_softmax_kernel_backward",
            (true, false) => "softmax_kernel_backward",
            (false, true) => "online_softmax_kernel",
            (false, false) => "softmax_kernel",
        };
        var signature = backward
            ? "float*, const float*, const float*, int, int, int, int"
            : online
                ? "float*, const float*, int, int, int"
                : "float*, const float*, int, int, int, int, int";
        var problem = new TileGymSoftmaxProblem(rows, columns, online, backward);
        var candidates = TileGymSoftmaxCandidates.For(problem);
        var kernel = TileGymKernel.Tuned("softmax.cuh", name, signature, problem, candidates);
        return new(kernel, (runtime, report) =>
        {
            using var input = runtime.Allocate<float>(rows * columns);
            using var output = runtime.Allocate<float>(input.Length);
            using var dy = backward ? runtime.Allocate<float>(input.Length) : null;
            var hi = Values(input.Length);
            var probabilities = Softmax(hi, rows, columns);
            var expected = probabilities;
            if (dy is not null)
            {
                input.CopyFrom(probabilities);
                var hdy = Values(dy.Length);
                dy.CopyFrom(hdy);
                expected = new float[input.Length];
                for (var r = 0; r < rows; r++)
                {
                    var dot = 0f;
                    for (var c = 0; c < columns; c++)
                    {
                        dot += probabilities[r * columns + c] * hdy[r * columns + c];
                    }
                    for (var c = 0; c < columns; c++)
                    {
                        expected[r * columns + c] = probabilities[r * columns + c] * (hdy[r * columns + c] - dot);
                    }
                }
            }
            else
            {
                input.CopyFrom(hi);
            }
            void Launch(CUfunction function, TileCppGrid grid)
            {
                if (dy is not null)
                {
                    cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                        output.Pointer, input.Pointer, dy.Pointer, columns, columns, columns, columns).Ok();
                }
                else if (online)
                {
                    cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                        output.Pointer, input.Pointer, columns, columns, columns).Ok();
                }
                else
                {
                    cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, runtime.Stream,
                        output.Pointer, input.Pointer, columns, columns, rows, columns, rows).Ok();
                }
            }
            var run = runtime.Run(kernel, Launch, () => output.Validate(expected, name, 8e-4f, 8e-4f));
            report.Add("softmax", $"{rows}x{columns}",
                input.ByteLength + output.ByteLength + (dy?.ByteLength ?? 0), run);
        });
    }

    static float[] Rotate(float[] x, float[] cos, float[] sin, int heads, int sequence, int head, bool backward)
    {
        var y = (float[])x.Clone();
        var half = head / 2;
        for (var h = 0; h < heads; h++)
        {
            for (var s = 0; s < sequence; s++)
            {
                for (var d = 0; d < half; d++)
                {
                    var offset = (h * sequence + s) * head;
                    var a = x[offset + d];
                    var b = x[offset + half + d];
                    var c = cos[(s * 2) * half + d];
                    var sn = sin[(s * 2) * half + d];
                    y[offset + d] = backward ? a * c + b * sn : a * c - b * sn;
                    y[offset + half + d] = backward ? b * c - a * sn : b * c + a * sn;
                }
            }
        }
        return y;
    }

    static float[] Softmax(float[] x, int rows, int columns)
    {
        var y = new float[x.Length];
        for (var r = 0; r < rows; r++)
        {
            var max = float.NegativeInfinity;
            for (var c = 0; c < columns; c++)
            {
                max = Math.Max(max, x[r * columns + c]);
            }
            var sum = 0f;
            for (var c = 0; c < columns; c++)
            {
                var value = MathF.Exp(x[r * columns + c] - max);
                y[r * columns + c] = value;
                sum += value;
            }
            for (var c = 0; c < columns; c++)
            {
                y[r * columns + c] /= sum;
            }
        }
        return y;
    }

    static float[] Values(int count)
    {
        var a = new float[count];
        for (var i = 0; i < count; i++)
        {
            a[i] = (i % 127 - 63) / 32f;
        }
        return a;
    }
}
