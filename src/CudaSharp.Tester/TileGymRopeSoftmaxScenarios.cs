using System;
using CudaSharp.Tile;
using static CudaSharp.nvcuda;

namespace CudaSharp.Tester;

static class TileGymRopeSoftmaxScenarios
{
    static readonly TileCppConfig Config = new([]);

    public static void RunAll(TileGymRuntime runtime, TileGymReport report)
    {
        RunRope(runtime, report, false);
        RunRope(runtime, report, true);
        RunSoftmax(runtime, report, false, false);
        RunSoftmax(runtime, report, false, true);
        RunSoftmax(runtime, report, true, false);
        RunSoftmax(runtime, report, true, true);
    }

    static unsafe void RunRope(TileGymRuntime runtime, TileGymReport report, bool backward)
    {
        const int batch = 1;
        const int qHeads = 2;
        const int kHeads = 1;
        const int sequence = 4;
        const int head = 64;
        const int half = 32;
        var name = backward ? "rope_backward_kernel" : "rope_kernel";
        var templates = $"float, float, float, {batch}, {qHeads}, {kHeads}, {qHeads}, {kHeads}, {half}, {half}, {head}, 1, {sequence}";
        using var kernel = TileGymKernel.Create(runtime, "rope.cuh", name, templates, "float*, float*, const float*, const float*");
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
        q.CopyFrom(hq);
        k.CopyFrom(hk);
        cos.CopyFrom(hc);
        sin.CopyFrom(hs);

        void Launch()
        {
            var pq = q.Pointer.Value;
            var pk = k.Pointer.Value;
            var pc = cos.Pointer.Value;
            var ps = sin.Pointer.Value;
            var args = stackalloc IntPtr[] { (IntPtr)(&pq), (IntPtr)(&pk), (IntPtr)(&pc), (IntPtr)(&ps) };
            kernel.Launch(Config, new(batch * sequence), runtime.Stream, new(args, 4));
        }
        var timing = TileGymKernel.Measure(runtime, Launch);
        q.CopyFrom(hq);
        k.CopyFrom(hk);
        Launch();
        cuStreamSynchronize(runtime.Stream).Ok();
        var eq = Rotate(hq, hc, hs, qHeads, sequence, head, backward);
        var ek = Rotate(hk, hc, hs, kHeads, sequence, head, backward);
        TileGymKernel.Validate(q.CopyToHost(), eq, name, 5e-4f, 5e-4f);
        TileGymKernel.Validate(k.CopyToHost(), ek, name, 5e-4f, 5e-4f);
        TileGymKernel.Report(
            report, "rope", name,
            $"B={batch},QH={qHeads},KH={kHeads},S={sequence},D={head}", templates,
            q.ByteLength + k.ByteLength + cos.ByteLength + sin.ByteLength,
            timing);
    }

    static unsafe void RunSoftmax(TileGymRuntime runtime, TileGymReport report, bool online, bool backward)
    {
        const int rows = 4;
        var columns = online ? 1025 : 256;
        var block = online ? 256 : 256;
        var name = backward ? (online ? "online_softmax_kernel_backward" : "softmax_kernel_backward") : (online ? "online_softmax_kernel" : "softmax_kernel");
        var signature = backward
            ? "float*, const float*, const float*, int, int, int, int"
            : online
                ? "float*, const float*, int, int, int"
                : "float*, const float*, int, int, int, int, int";
        using var kernel = TileGymKernel.Create(
            runtime, "softmax.cuh", name, $"float, {block}", signature);
        using var input = runtime.Allocate<float>(rows * columns);
        using var output = runtime.Allocate<float>(input.Length);
        using var dy = backward ? runtime.Allocate<float>(input.Length) : null;
        var hi = Values(input.Length);
        var probabilities = Softmax(hi, rows, columns);
        if (backward)
        {
            input.CopyFrom(probabilities);
            var hdy = Values(dy!.Length);
            dy.CopyFrom(hdy);
        }
        else
        {
            input.CopyFrom(hi);
        }

        void Launch()
        {
            var po = output.Pointer.Value;
            var pi = input.Pointer.Value;
            var pdy = dy?.Pointer.Value ?? IntPtr.Zero;
            var stride = columns;
            var nrows = rows;
            var ncols = columns;
            var programs = rows;
            if (backward)
            {
                var args = stackalloc IntPtr[]
                {
                    (IntPtr)(&po), (IntPtr)(&pi), (IntPtr)(&pdy), (IntPtr)(&stride),
                    (IntPtr)(&stride), (IntPtr)(&stride), (IntPtr)(&ncols)
                };
                kernel.Launch(Config, new(rows), runtime.Stream, new(args, 7));
            }
            else if (online)
            {
                var args = stackalloc IntPtr[] { (IntPtr)(&po), (IntPtr)(&pi), (IntPtr)(&stride), (IntPtr)(&stride), (IntPtr)(&ncols) };
                kernel.Launch(Config, new(rows), runtime.Stream, new(args, 5));
            }
            else
            {
                var args = stackalloc IntPtr[]
                {
                    (IntPtr)(&po), (IntPtr)(&pi), (IntPtr)(&stride), (IntPtr)(&stride),
                    (IntPtr)(&nrows), (IntPtr)(&ncols), (IntPtr)(&programs)
                };
                kernel.Launch(Config, new(rows), runtime.Stream, new(args, 7));
            }
        }
        var timing = TileGymKernel.Measure(runtime, Launch);
        float[] expected;
        if (backward)
        {
            var hdy = dy!.CopyToHost();
            expected = new float[input.Length];
            for (var r = 0; r < rows; r++)
            {
                var dot = 0f;
                for (var c = 0; c < columns; c++) dot += probabilities[r * columns + c] * hdy[r * columns + c];
                for (var c = 0; c < columns; c++) expected[r * columns + c] = probabilities[r * columns + c] * (hdy[r * columns + c] - dot);
            }
        }
        else
        {
            expected = probabilities;
        }
        TileGymKernel.Validate(output.CopyToHost(), expected, name, 8e-4f, 8e-4f);
        TileGymKernel.Report(
            report, "softmax", name, $"{rows}x{columns}", $"float,BLOCK_SIZE={block}",
            input.ByteLength + output.ByteLength + (dy?.ByteLength ?? 0), timing);
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
            for (var c = 0; c < columns; c++) max = Math.Max(max, x[r * columns + c]);
            var sum = 0f;
            for (var c = 0; c < columns; c++)
            {
                var value = MathF.Exp(x[r * columns + c] - max);
                y[r * columns + c] = value;
                sum += value;
            }
            for (var c = 0; c < columns; c++) y[r * columns + c] /= sum;
        }
        return y;
    }

    static float[] Values(int count)
    {
        var a = new float[count];
        for (var i = 0; i < count; i++) a[i] = (i % 127 - 63) / 32f;
        return a;
    }
}
