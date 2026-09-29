using System.Collections.Generic;
using System.Diagnostics;
using CudaSharp.TileGym;
using static CudaSharp.nvcuda;

namespace CudaSharp.Tester;

/// <summary>The variant chosen for a kernel and what it cost to choose it.</summary>
sealed record TileGymSelection(TileGymVariant Variant, TileGymCompiledKernel Compiled,
    double FirstLaunchMilliseconds, double TuneMilliseconds, int Candidates, IReadOnlyList<string> Rejections);

/// <summary>
/// Selects the fastest variant among already compiled and loaded kernels. Each variant is launched once and
/// validated; when more than one remains, each is timed and the fastest is selected. Compilation and module
/// loading are never part of tuning.
/// </summary>
sealed class TileGymTuner
{
    readonly ITileCppTimer _timer;
    readonly CUstream _stream;
    readonly Func<TileGymKernelSpec, TileGymCompiledKernel> _resolve;
    readonly TileCppTimingOptions _timingOptions;

    public TileGymTuner(ITileCppTimer timer, CUstream stream,
        Func<TileGymKernelSpec, TileGymCompiledKernel> resolve, TileCppTimingOptions? timingOptions = null)
    {
        ArgumentNullException.ThrowIfNull(timer);
        ArgumentNullException.ThrowIfNull(resolve);
        _timer = timer;
        _stream = stream;
        _resolve = resolve;
        _timingOptions = timingOptions ?? new TileCppTimingOptions();
    }

    public TileGymSelection Select(string name, IReadOnlyList<TileGymVariant> variants, TileGymLaunch launch,
        Action? validate = null, Action? reset = null)
    {
        ArgumentNullException.ThrowIfNull(variants);
        ArgumentNullException.ThrowIfNull(launch);
        var watch = Stopwatch.StartNew();
        var rejections = new List<string>();
        Exception? firstError = null;
        var valid = new List<(TileGymVariant Variant, TileGymCompiledKernel Compiled, double FirstLaunch)>();
        foreach (var variant in variants)
        {
            var compiled = _resolve(variant.Spec);
            try
            {
                if (compiled.Error is { } error)
                {
                    throw new InvalidOperationException($"Compilation or loading failed for {variant.Spec}.", error);
                }
                reset?.Invoke();
                var first = Stopwatch.StartNew();
                launch(compiled.Function, variant.Grid);
                _timer.Synchronize(_stream);
                first.Stop();
                validate?.Invoke();
                valid.Add((variant, compiled, first.Elapsed.TotalMilliseconds));
            }
            catch (Exception ex)
            {
                firstError ??= ex;
                rejections.Add($"{variant.Label}: {(ex.InnerException ?? ex).Message}");
            }
        }
        if (valid.Count == 0)
        {
            throw new InvalidOperationException($"No valid variant of {name} is available.", firstError);
        }
        if (variants.Count == 1)
        {
            var only = valid[0];
            return new(only.Variant, only.Compiled, only.FirstLaunch, 0, 1, rejections);
        }

        var best = valid[0];
        var bestMilliseconds = float.PositiveInfinity;
        foreach (var candidate in valid)
        {
            var function = candidate.Compiled.Function;
            var grid = candidate.Variant.Grid;
            try
            {
                var milliseconds = _timer.Measure(() => launch(function, grid), _stream, _timingOptions);
                if (milliseconds < bestMilliseconds)
                {
                    best = candidate;
                    bestMilliseconds = milliseconds;
                }
            }
            catch (Exception ex)
            {
                rejections.Add($"{candidate.Variant.Label}: {ex.Message}");
            }
        }
        if (float.IsPositiveInfinity(bestMilliseconds))
        {
            throw new InvalidOperationException($"No variant of {name} completed timing.");
        }
        watch.Stop();
        return new(best.Variant, best.Compiled, best.FirstLaunch, watch.Elapsed.TotalMilliseconds,
            variants.Count, rejections);
    }
}
