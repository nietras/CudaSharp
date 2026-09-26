using System;
using CudaSharp.Tile;
using static CudaSharp.nvcuda;

namespace CudaSharp.Tester;

sealed class TileGymResettableTimer : ITileCppTimer
{
    readonly CudaEventTileCppTimer _stateless = new();

    public Action? Prepare { get; set; }
    public bool IsMeasuring { get; private set; }

    public void Synchronize(CUstream stream) => _stateless.Synchronize(stream);

    public float Measure(Action launch, CUstream stream, TileCppTimingOptions options)
    {
        ArgumentNullException.ThrowIfNull(launch);
        ArgumentNullException.ThrowIfNull(options);
        if (Prepare is null)
        {
            return _stateless.Measure(launch, stream, options);
        }
        var prepare = Prepare;
        Synchronize(stream);
        cuEventCreate(out var start, 0).Ok();
        cuEventCreate(out var end, 0).Ok();
        try
        {
            IsMeasuring = true;
            prepare();
            launch();
            Synchronize(stream);
            var samples = new float[64];
            var count = 0;
            var total = 0f;
            do
            {
                prepare();
                cuEventRecord(start, stream).Ok();
                launch();
                cuEventRecord(end, stream).Ok();
                cuEventSynchronize(end).Ok();
                cuEventElapsedTime(out var milliseconds, start, end).Ok();
                samples[count++] = milliseconds;
                total += milliseconds;
            }
            while (count < samples.Length && (count < 10 || total < options.MeasurementMilliseconds));
            Array.Sort(samples, 0, count);
            return samples[count / 2];
        }
        finally
        {
            IsMeasuring = false;
            cuEventDestroy(start).Ok();
            cuEventDestroy(end).Ok();
        }
    }
}
