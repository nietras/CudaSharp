using System;
using System.Collections.Generic;
using System.Linq;
using CudaSharp.Tile;

namespace CudaSharp.Tester;

static class TileGymTuning
{
    public static TileGymTunedResult Tune<TProblem>(
        TileGymRuntime runtime, TProblem problem, IReadOnlyList<TileGymCandidate> candidates,
        string header, string kernelName, string signature,
        Func<TProblem, TileGymCandidate, string> templateArguments,
        Func<TProblem, TileGymCandidate, TileCppGrid> grid,
        Action<TileCppKernel, TileCppConfig, TileCppGrid> launch,
        Action<TileGymCandidate>? validate = null, Action? prepare = null,
        TileCppTimingOptions? timingOptions = null, Action<string>? log = null)
        where TProblem : notnull
    {
        ArgumentNullException.ThrowIfNull(runtime);
        ArgumentNullException.ThrowIfNull(problem);
        ArgumentNullException.ThrowIfNull(candidates);
        ArgumentException.ThrowIfNullOrWhiteSpace(header);
        ArgumentException.ThrowIfNullOrWhiteSpace(kernelName);
        ArgumentException.ThrowIfNullOrWhiteSpace(signature);
        ArgumentNullException.ThrowIfNull(templateArguments);
        ArgumentNullException.ThrowIfNull(grid);
        ArgumentNullException.ThrowIfNull(launch);

        var selectedCandidates = runtime.EnableAutotuning ? candidates : candidates.Take(1).ToArray();
        var searchSpace = string.Join(";", selectedCandidates.Select(static candidate =>
            $"{candidate}|{candidate.CompilerConfig}"));
        var key = (typeof(TProblem), problem, header, kernelName, signature, searchSpace);
        var session = runtime.GetOrCreateTuningSession(key, () =>
            new TileGymTuningSession<TProblem>(selectedCandidates,
                (p, candidate) => TileGymKernel.Create(runtime.Compiler, header, kernelName,
                    templateArguments(p, candidate), signature), grid));
        return session.Tune(problem, runtime.Stream, launch, timingOptions, log, validate, prepare);
    }
}
