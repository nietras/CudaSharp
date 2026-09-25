using System;
using System.Collections.Generic;
using CudaSharp.Tile;

namespace CudaSharp.Tester;

readonly record struct TileGymPersistentRmsNormProblem(int Rows, int Columns, int SmCount)
{
    public string TemplateArguments(TileGymCandidate candidate)
    {
        ArgumentNullException.ThrowIfNull(candidate);
        return $"float, float, {candidate["TileM"]}, {Columns}, {candidate["Occupancy"]}, " +
            $"{Rows}, {Columns}, {Grid(candidate).X}, 0.00001f, 0.0f";
    }

    public TileCppGrid Grid(TileGymCandidate candidate)
    {
        ArgumentNullException.ThrowIfNull(candidate);
        var tileM = candidate.GetInt32("TileM");
        if (Rows <= 0 || SmCount <= 0 || tileM <= 0 || tileM > Rows)
        {
            throw new ArgumentOutOfRangeException(nameof(candidate));
        }
        return new TileCppGrid(checked((uint)Math.Min(SmCount, 1 + (Rows - 1) / tileM)));
    }
}

static class TileGymPersistentRmsNormCandidates
{
    public static IReadOnlyList<TileGymCandidate> For(TileGymPersistentRmsNormProblem problem)
    {
        var candidates = new List<TileGymCandidate>();
        foreach (var tileM in new[] { 2, 4, 8 })
        {
            if (tileM > problem.Rows || (long)tileM * problem.Columns > 256 * 8 * 32)
            {
                continue;
            }
            foreach (var occupancy in new[] { 1, 2 })
            {
                candidates.Add(new TileGymCandidate(
                [
                    TileGymHyperparameter.Integer("TileM", tileM),
                    TileGymHyperparameter.Integer("Occupancy", occupancy)
                ]));
            }
        }
        return candidates;
    }
}
