using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Text;

namespace CudaSharp.TileGym;

/// <summary>Buffers nested compile-stage timings until the outermost operation finishes.</summary>
sealed class TileCppCompilationTiming : IDisposable
{
    [ThreadStatic]
    static TileCppCompilationTiming? s_current;

    readonly TileCppCompilationTiming? _parent = s_current;
    readonly List<Stage> _stages;
    readonly string _operation;
    readonly long _start = Stopwatch.GetTimestamp();
    readonly int _threadId = Environment.CurrentManagedThreadId;
    string? _stage;
    long _stageStart;

    public TileCppCompilationTiming(string operation)
    {
        _operation = operation;
        _stages = _parent?._stages ?? [];
        s_current = this;
    }

    public void Step(string stage)
    {
        var end = Stopwatch.GetTimestamp();
        FinishStage(end);
        _stage = stage;
        _stageStart = Stopwatch.GetTimestamp();
    }

    void FinishStage(long end)
    {
        if (_stage is not null)
        {
            _stages.Add(new Stage(_operation, _stage, _stageStart, end));
        }
    }

    public void Dispose()
    {
        var end = Stopwatch.GetTimestamp();
        FinishStage(end);
        s_current = _parent;
        if (_parent is not null)
        {
            return;
        }

        // Writing while a parent stage is being timed would make Trace's own lock look like a compile bottleneck.
        var total = Stopwatch.GetElapsedTime(_start, end).TotalMilliseconds;
        var output = new StringBuilder();
        output.Append(CultureInfo.InvariantCulture,
            $"Compile timing: {_operation}; thread={_threadId}; total={total:F3} ms");
        foreach (var stage in _stages)
        {
            var offset = Stopwatch.GetElapsedTime(_start, stage.Start).TotalMilliseconds;
            var elapsed = Stopwatch.GetElapsedTime(stage.Start, stage.End).TotalMilliseconds;
            output.Append(CultureInfo.InvariantCulture,
                $"\n  +{offset:F3} ms {stage.Operation}.{stage.Name}: {elapsed:F3} ms");
        }
        var message = output.ToString();
        Trace.WriteLine(message);
    }

    readonly record struct Stage(string Operation, string Name, long Start, long End)
    {
    }
}
