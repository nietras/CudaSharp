using System.Collections.Generic;
using System.Linq;
using System.Threading;
using static CudaSharp.nvcuda;

namespace CudaSharp.Tile;

/// <summary>Specifies CUDA Tile C++ benchmark time budgets.</summary>
/// <seealso href="https://docs.nvidia.com/cuda/cutile-python/performance.html" />
public sealed record TileCppTimingOptions
{
    /// <summary>Creates benchmark time budgets.</summary>
    /// <param name="warmupMilliseconds">Approximate warmup duration.</param>
    /// <param name="measurementMilliseconds">Approximate measurement duration.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/cutile-python/performance.html" />
    public TileCppTimingOptions(float warmupMilliseconds = 25, float measurementMilliseconds = 100)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(warmupMilliseconds);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(measurementMilliseconds);
        WarmupMilliseconds = warmupMilliseconds;
        MeasurementMilliseconds = measurementMilliseconds;
    }

    /// <summary>Gets the approximate warmup duration.</summary>
    public float WarmupMilliseconds { get; }

    /// <summary>Gets the approximate measurement duration.</summary>
    public float MeasurementMilliseconds { get; }
}

/// <summary>Measures repeated CUDA Tile C++ kernel launches.</summary>
/// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html" />
public interface ITileCppTimer
{
    /// <summary>Waits for preceding work on a CUDA stream to complete.</summary>
    /// <param name="stream">CUDA stream to synchronize.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html" />
    void Synchronize(CUstream stream);

    /// <summary>Measures a kernel-launch callback on a CUDA stream.</summary>
    /// <param name="launch">Callback that enqueues one kernel launch.</param>
    /// <param name="stream">CUDA stream used by the callback.</param>
    /// <param name="options">Warmup and measurement budgets.</param>
    /// <returns>Representative execution time in milliseconds.</returns>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html" />
    float Measure(Action launch, CUstream stream, TileCppTimingOptions options);
}

/// <summary>Measures CUDA Tile C++ launches with per-invocation CUDA event pairs.</summary>
/// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html" />
public sealed class CudaEventTileCppTimer : ITileCppTimer
{
    /// <inheritdoc />
    public void Synchronize(CUstream stream) => cuStreamSynchronize(stream).Ok();

    /// <inheritdoc />
    public float Measure(Action launch, CUstream stream, TileCppTimingOptions options)
    {
        ArgumentNullException.ThrowIfNull(launch);
        ArgumentNullException.ThrowIfNull(options);
        Synchronize(stream);

        launch();
        Synchronize(stream);
        var estimate = MeasureBatch(launch, stream, 5) / 5;
        var warmupCount = Math.Max(1, (int)(options.WarmupMilliseconds / Math.Max(estimate, 0.001f)));
        var repeatCount = Math.Max(10, (int)(options.MeasurementMilliseconds / Math.Max(estimate, 0.001f)));

        for (var i = 0; i < warmupCount; i++)
            launch();
        Synchronize(stream);

        var starts = new CUevent[repeatCount];
        var ends = new CUevent[repeatCount];
        try
        {
            for (var i = 0; i < repeatCount; i++)
            {
                cuEventCreate(out starts[i], 0).Ok();
                cuEventCreate(out ends[i], 0).Ok();
                cuEventRecord(starts[i], stream).Ok();
                launch();
                cuEventRecord(ends[i], stream).Ok();
            }

            cuEventSynchronize(ends[^1]).Ok();
            var times = GC.AllocateUninitializedArray<float>(repeatCount);
            for (var i = 0; i < repeatCount; i++)
                cuEventElapsedTime(out times[i], starts[i], ends[i]).Ok();
            Array.Sort(times);

            var trim = times.Length / 10;
            var first = trim;
            var length = times.Length - 2 * trim;
            if (length <= 0)
            {
                first = 0;
                length = times.Length;
            }
            return times[first + length / 2];
        }
        finally
        {
            DestroyEvents(starts);
            DestroyEvents(ends);
        }
    }

    static float MeasureBatch(Action launch, CUstream stream, int count)
    {
        cuEventCreate(out var start, 0).Ok();
        cuEventCreate(out var end, 0).Ok();
        try
        {
            cuEventRecord(start, stream).Ok();
            for (var i = 0; i < count; i++)
                launch();
            cuEventRecord(end, stream).Ok();
            cuEventSynchronize(end).Ok();
            cuEventElapsedTime(out var milliseconds, start, end).Ok();
            return milliseconds;
        }
        finally
        {
            cuEventDestroy(start).Ok();
            cuEventDestroy(end).Ok();
        }
    }

    static void DestroyEvents(CUevent[] events)
    {
        foreach (var cudaEvent in events)
        {
            if (cudaEvent.Value != IntPtr.Zero)
                cuEventDestroy(cudaEvent).Ok();
        }
    }
}

/// <summary>Searches and caches the fastest CUDA Tile C++ kernel configuration for each problem key.</summary>
/// <remarks>
/// Tuning first launches candidate configurations to populate their compile cache, then times only successfully
/// compiled candidates. The fastest configuration is cached and launched once more for the caller.
/// </remarks>
/// <seealso href="https://docs.nvidia.com/cuda/cutile-python/performance.html" />
public sealed class TileCppAutotuner
{
    static readonly IReadOnlyDictionary<string, object?> EmptyArguments = new Dictionary<string, object?>();
    readonly Dictionary<object, TileCppTunedResult> _cache = [];
    readonly Lock _lock = new();
    readonly TileCppSearchSpace _searchSpace;
    readonly ITileCppTimer _timer;

    /// <summary>Creates a CUDA Tile C++ autotuner.</summary>
    /// <param name="searchSpace">Configurations and optional problem-dependent filter.</param>
    /// <param name="timer">Optional launch timer; CUDA events are used by default.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/cutile-python/performance.html" />
    public TileCppAutotuner(TileCppSearchSpace searchSpace, ITileCppTimer? timer = null)
    {
        ArgumentNullException.ThrowIfNull(searchSpace);
        _searchSpace = searchSpace;
        _timer = timer ?? new CudaEventTileCppTimer();
    }

    /// <summary>Clears all cached tuning results.</summary>
    /// <seealso href="https://docs.nvidia.com/cuda/cutile-python/performance.html" />
    public void ClearCache()
    {
        lock (_lock)
            _cache.Clear();
    }

    /// <summary>Removes a cached tuning result for one problem key.</summary>
    /// <param name="key">Problem-specific cache key.</param>
    /// <returns>Whether a cached result was removed.</returns>
    /// <seealso href="https://docs.nvidia.com/cuda/cutile-python/performance.html" />
    public bool ClearCache(object key)
    {
        ArgumentNullException.ThrowIfNull(key);
        lock (_lock)
            return _cache.Remove(key);
    }

    /// <summary>Tunes, caches, and launches a CUDA Tile C++ kernel configuration.</summary>
    /// <param name="stream">CUDA stream used for compilation warmup, timing, and the final launch.</param>
    /// <param name="key">Problem-specific cache key.</param>
    /// <param name="launch">Callback that compiles if needed and launches one configuration.</param>
    /// <param name="getGrid">Computes the launch grid for a configuration.</param>
    /// <param name="namedArguments">Problem arguments available to filtering and grid computation.</param>
    /// <param name="maxIterations">Maximum number of valid configurations to benchmark.</param>
    /// <param name="seed">Optional deterministic sampling seed.</param>
    /// <param name="forceRetune">Whether to ignore and replace a cached winner.</param>
    /// <param name="timingOptions">Optional warmup and measurement budgets.</param>
    /// <param name="log">Optional diagnostic callback.</param>
    /// <returns>The selected configuration, grid, and measured execution time.</returns>
    /// <seealso href="https://docs.nvidia.com/cuda/cutile-python/performance.html" />
    public TileCppTunedResult Tune(CUstream stream, object key,
        Action<TileCppConfig> launch,
        Func<IReadOnlyDictionary<string, object?>, TileCppConfig, TileCppGrid> getGrid,
        IReadOnlyDictionary<string, object?>? namedArguments = null,
        int maxIterations = 60, int? seed = null, bool forceRetune = false,
        TileCppTimingOptions? timingOptions = null, Action<string>? log = null)
    {
        ArgumentNullException.ThrowIfNull(key);
        ArgumentNullException.ThrowIfNull(launch);
        ArgumentNullException.ThrowIfNull(getGrid);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxIterations);
        namedArguments ??= EmptyArguments;
        timingOptions ??= new TileCppTimingOptions();

        var arguments = namedArguments ?? EmptyArguments;
        TileCppTunedResult result;
        lock (_lock)
        {
            if (!forceRetune && _cache.TryGetValue(key, out var cached))
            {
                result = cached;
                log?.Invoke($"CUDA Tile C++ autotune cache hit: {result.Config}");
            }
            else
            {
                result = TuneCore(stream, launch, getGrid, arguments,
                    maxIterations, seed, timingOptions, log);
                _cache[key] = result;
            }
        }

        launch(result.Config);
        return result;
    }

    TileCppTunedResult TuneCore(CUstream stream,
        Action<TileCppConfig> launch,
        Func<IReadOnlyDictionary<string, object?>, TileCppConfig, TileCppGrid> getGrid,
        IReadOnlyDictionary<string, object?> namedArguments,
        int maxIterations, int? seed, TileCppTimingOptions timingOptions, Action<string>? log)
    {
        var indices = Enumerable.Range(0, _searchSpace.Count).ToArray();
        Shuffle(indices, seed is null ? Random.Shared : new Random(seed.Value));

        var candidates = new List<Candidate>(Math.Min(maxIterations, indices.Length));
        foreach (var index in indices)
        {
            if (candidates.Count >= maxIterations)
                break;

            var config = _searchSpace[index];
            if (!_searchSpace.IsMatch(namedArguments, config))
                continue;

            try
            {
                var grid = getGrid(namedArguments, config);
                launch(config);
                candidates.Add(new Candidate(config, grid));
            }
            catch (Exception ex)
            {
                log?.Invoke($"CUDA Tile C++ configuration rejected during precompile: {config}; {ex.Message}");
            }
        }

        if (candidates.Count == 0)
            throw new InvalidOperationException("No valid CUDA Tile C++ configuration was found.");
        _timer.Synchronize(stream);

        TileCppTunedResult? best = null;
        foreach (var candidate in candidates)
        {
            try
            {
                var milliseconds = _timer.Measure(() => launch(candidate.Config), stream, timingOptions);
                if (best is null || milliseconds < best.Milliseconds)
                {
                    best = new TileCppTunedResult(candidate.Config, candidate.Grid, milliseconds);
                    log?.Invoke($"New CUDA Tile C++ best: {candidate.Config}; {milliseconds:F3} ms");
                }
                else
                {
                    log?.Invoke($"CUDA Tile C++ candidate: {candidate.Config}; {milliseconds:F3} ms");
                }
            }
            catch (Exception ex)
            {
                log?.Invoke($"CUDA Tile C++ configuration rejected during timing: {candidate.Config}; {ex.Message}");
            }
        }

        return best ?? throw new InvalidOperationException("No CUDA Tile C++ configuration completed timing.");
    }

    static void Shuffle(Span<int> values, Random random)
    {
        for (var i = values.Length - 1; i > 0; i--)
        {
            var other = random.Next(i + 1);
            (values[i], values[other]) = (values[other], values[i]);
        }
    }

    readonly record struct Candidate(TileCppConfig Config, TileCppGrid Grid);
}
