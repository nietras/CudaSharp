using System;
using System.Collections.Generic;
using CudaSharp.Tile;

namespace CudaSharp.Tester;

readonly record struct TileGymPersistentLayerNormProblem(int Rows, int Columns, int SmCount)
{
    public string TemplateArguments(TileGymCandidate candidate)
    {
        ArgumentNullException.ThrowIfNull(candidate);
        var grid = Grid(candidate).X;
        return $"float, float, float, {candidate["BlockN"]}, {Columns}, false, true, true, " +
            $"{Rows}, {Columns}, {grid}, 0.00001f";
    }

    public TileCppGrid Grid(TileGymCandidate candidate)
    {
        ArgumentNullException.ThrowIfNull(candidate);
        var blockN = candidate.GetInt32("BlockN");
        if (Rows <= 0 || Columns <= 0 || SmCount <= 0 || blockN <= 0 ||
            blockN > Rows || (long)blockN * Columns > 256 * 8 * 32)
        {
            throw new ArgumentOutOfRangeException(nameof(candidate));
        }
        return new TileCppGrid(checked((uint)Math.Min(SmCount, 1 + (Rows - 1) / blockN)));
    }
}

static class TileGymPersistentLayerNormCandidates
{
    public static IReadOnlyList<TileGymCandidate> For(TileGymPersistentLayerNormProblem problem)
    {
        var candidates = new List<TileGymCandidate>();
        foreach (var blockN in new[] { 2, 4, 8, 16, 32 })
        {
            if (blockN <= problem.Rows && (long)blockN * problem.Columns <= 256 * 8 * 32)
            {
                candidates.Add(new TileGymCandidate([TileGymHyperparameter.Integer("BlockN", blockN)]));
            }
        }
        return candidates;
    }
}
