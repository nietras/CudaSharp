using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using CudaSharp.TileGym;
using static CudaSharp.nvcuda;

namespace CudaSharp.Tester;

sealed record TileGymTunedResult(
    TileGymCandidate Candidate, TileCppKernel Kernel, TileCppGrid Grid, double TuneMilliseconds,
    float KernelMilliseconds, bool CacheHit, int CandidateCount,
    double? CompileMilliseconds = null, double? LoadMilliseconds = null, double? FirstLaunchMilliseconds = null);

sealed class TileGymTuningSession<TProblem> : IDisposable where TProblem : notnull
{
    readonly Dictionary<TileCppConfig, TileGymCandidate> _candidates;
    readonly Dictionary<(TProblem Problem, TileCppConfig Config), TileCppKernel> _kernels = [];
    readonly HashSet<TProblem> _tuned = [];
    readonly HashSet<(TProblem Problem, TileCppConfig Config)> _validated = [];
    readonly HashSet<(TProblem Problem, TileCppConfig Config)> _warmed = [];
    readonly TileCppAutotuner _autotuner;
    readonly Func<TProblem, TileGymCandidate, TileCppKernel> _createKernel;
    readonly Func<TProblem, TileGymCandidate, TileCppGrid> _getGrid;
    readonly ITileCppTimer _timer;
    readonly Action<TileCppKernel, TileCppConfig> _compile;
    readonly Action<TileCppKernel, TileCppConfig> _load;
    readonly object _gate = new();
    bool _disposed;

    public TileGymTuningSession(IEnumerable<TileGymCandidate> candidates,
        Func<TProblem, TileGymCandidate, TileCppKernel> createKernel,
        Func<TProblem, TileGymCandidate, TileCppGrid> getGrid, ITileCppTimer? timer = null,
        Action<TileCppKernel, TileCppConfig>? compile = null,
        Action<TileCppKernel, TileCppConfig>? load = null)
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
        _compile = compile ?? ((kernel, config) => kernel.Compile(config));
        _load = load ?? ((kernel, config) => kernel.LoadFunction(config));
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
                double? compileMilliseconds = null;
                double? loadMilliseconds = null;
                double? firstLaunchMilliseconds = null;
                var loadTimes = new Dictionary<TileCppConfig, double>();
                var firstLaunchTimes = new Dictionary<TileCppConfig, double>();
                var failures = new Dictionary<TileCppConfig, Exception>();
                var toCompile = new List<KeyValuePair<(TProblem Problem, TileCppConfig Config), TileCppKernel>>();
                if (!cacheHit)
                {
                    foreach (var candidate in _candidates.Values)
                    {
                        var kernelKey = (problem, candidate.CompilerConfig);
                        if (_kernels.TryGetValue(kernelKey, out var existing))
                        {
                            toCompile.Add(new(kernelKey, existing));
                            continue;
                        }
                        try
                        {
                            var kernel = _createKernel(problem, candidate);
                            _kernels.Add(kernelKey, kernel);
                            toCompile.Add(new(kernelKey, kernel));
                        }
                        catch (Exception ex)
                        {
                            failures.Add(candidate.CompilerConfig, ex);
                        }
                    }
                    var compileWatch = Stopwatch.StartNew();
                    var compileFailures = new System.Collections.Concurrent.ConcurrentDictionary<TileCppConfig, Exception>();
                    Parallel.ForEach(toCompile, pair =>
                    {
                        try
                        {
                            _compile(pair.Value, pair.Key.Config);
                        }
                        catch (Exception ex)
                        {
                            compileFailures.TryAdd(pair.Key.Config, ex);
                        }
                    });
                    compileWatch.Stop();
                    compileMilliseconds = compileWatch.Elapsed.TotalMilliseconds;
                    foreach (var failure in compileFailures)
                    {
                        failures.Add(failure.Key, failure.Value);
                    }
                }
                var result = _autotuner.Tune(stream, problem,
                    config =>
                    {
                        var candidate = _candidates[config];
                        var kernelKey = (problem, config);
                        if (failures.TryGetValue(config, out var error))
                        {
                            throw new InvalidOperationException($"Compilation failed for {config}.", error);
                        }
                        var kernel = _kernels[kernelKey];
                        var first = !cacheHit && !_warmed.Contains(kernelKey);
                        if (first)
                        {
                            var phaseWatch = Stopwatch.StartNew();
                            _load(kernel, config);
                            phaseWatch.Stop();
                            loadTimes[config] = phaseWatch.Elapsed.TotalMilliseconds;
                        }
                        if (resettableTimer?.IsMeasuring != true)
                        {
                            prepare?.Invoke();
                        }
                        var launchWatch = first ? Stopwatch.StartNew() : null;
                        launch(kernel, config, _getGrid(problem, candidate));
                        if (launchWatch is not null)
                        {
                            _timer.Synchronize(stream);
                            launchWatch.Stop();
                            firstLaunchTimes[config] = launchWatch.Elapsed.TotalMilliseconds;
                            _warmed.Add(kernelKey);
                        }
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
                if (!cacheHit)
                {
                    loadTimes.TryGetValue(result.Config, out var load);
                    firstLaunchTimes.TryGetValue(result.Config, out var firstLaunch);
                    loadMilliseconds = load;
                    firstLaunchMilliseconds = firstLaunch;
                }
                return new(_candidates[result.Config], _kernels[(problem, result.Config)], result.Grid,
                    cacheHit ? 0 : stopwatch.Elapsed.TotalMilliseconds, result.Milliseconds, cacheHit, _candidates.Count,
                    compileMilliseconds, loadMilliseconds, firstLaunchMilliseconds);
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
