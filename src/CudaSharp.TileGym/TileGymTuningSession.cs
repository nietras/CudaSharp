using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using CudaSharp.Tile;
using static CudaSharp.nvcuda;

namespace CudaSharp.Tester;

sealed record TileGymTunedResult(
    TileGymCandidate Candidate, TileCppKernel Kernel, TileCppGrid Grid, double TuneMilliseconds,
    float KernelMilliseconds, bool CacheHit, int CandidateCount);

sealed class TileGymTuningSession<TProblem> : IDisposable where TProblem : notnull
{
    readonly Dictionary<TileCppConfig, TileGymCandidate> _candidates;
    readonly Dictionary<(TProblem Problem, TileCppConfig Config), TileCppKernel> _kernels = [];
    readonly HashSet<TProblem> _tuned = [];
    readonly HashSet<(TProblem Problem, TileCppConfig Config)> _validated = [];
    readonly TileCppAutotuner _autotuner;
    readonly Func<TProblem, TileGymCandidate, TileCppKernel> _createKernel;
    readonly Func<TProblem, TileGymCandidate, TileCppGrid> _getGrid;
    readonly ITileCppTimer _timer;
    readonly object _gate = new();
    bool _disposed;

    public TileGymTuningSession(IEnumerable<TileGymCandidate> candidates,
        Func<TProblem, TileGymCandidate, TileCppKernel> createKernel,
        Func<TProblem, TileGymCandidate, TileCppGrid> getGrid, ITileCppTimer? timer = null)
    {
        ArgumentNullException.ThrowIfNull(candidates);
        ArgumentNullException.ThrowIfNull(createKernel);
        ArgumentNullException.ThrowIfNull(getGrid);
        var items = candidates.ToArray();
        _candidates = new Dictionary<TileCppConfig, TileGymCandidate>(items.Length);
        foreach (var candidate in items)
        {
            ArgumentNullException.ThrowIfNull(candidate);
            if (!_candidates.TryAdd(candidate.CompilerConfig, candidate))
            {
                throw new ArgumentException("Candidates must have distinct compiler configurations.", nameof(candidates));
            }
        }
        _timer = timer ?? new TileGymResettableTimer();
        _autotuner = new TileCppAutotuner(new TileCppSearchSpace(items.Select(static c => c.CompilerConfig)), _timer);
        _createKernel = createKernel;
        _getGrid = getGrid;
    }

    public TileGymTunedResult Tune(TProblem problem, CUstream stream,
        Action<TileCppKernel, TileCppConfig, TileCppGrid> launch,
        TileCppTimingOptions? timingOptions = null, Action<string>? log = null,
        Action<TileGymCandidate>? validate = null, Action? prepare = null)
    {
        ArgumentNullException.ThrowIfNull(launch);
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (prepare is not null && _timer is not TileGymResettableTimer)
            {
                throw new InvalidOperationException("Reset-aware tuning requires a TileGymResettableTimer.");
            }
            var resettableTimer = _timer as TileGymResettableTimer;
            if (resettableTimer is not null)
            {
                resettableTimer.Prepare = prepare;
            }
            try
            {
                var cacheHit = _tuned.Contains(problem);
                var stopwatch = Stopwatch.StartNew();
                var result = _autotuner.Tune(stream, problem,
                    config =>
                    {
                        var candidate = _candidates[config];
                        var kernelKey = (problem, config);
                        if (!_kernels.TryGetValue(kernelKey, out var kernel))
                        {
                            kernel = _createKernel(problem, candidate);
                            _kernels.Add(kernelKey, kernel);
                        }
                        if (resettableTimer?.IsMeasuring != true)
                        {
                            prepare?.Invoke();
                        }
                        launch(kernel, config, _getGrid(problem, candidate));
                        if (validate is not null && !_validated.Contains(kernelKey))
                        {
                            _timer.Synchronize(stream);
                            validate(candidate);
                            _validated.Add(kernelKey);
                        }
                    },
                    (_, config) => _getGrid(problem, _candidates[config]),
                    timingOptions: timingOptions, log: log);
                stopwatch.Stop();
                _tuned.Add(problem);
                return new(_candidates[result.Config], _kernels[(problem, result.Config)], result.Grid,
                    cacheHit ? 0 : stopwatch.Elapsed.TotalMilliseconds, result.Milliseconds, cacheHit, _candidates.Count);
            }
            finally
            {
                if (resettableTimer is not null)
                {
                    resettableTimer.Prepare = null;
                }
            }
        }
    }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed)
            {
                return;
            }
            foreach (var kernel in _kernels.Values)
            {
                kernel.Dispose();
            }
            _kernels.Clear();
            _disposed = true;
        }
    }
}
