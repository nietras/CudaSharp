using System.Collections.Generic;
using CudaSharp.TileGym;

namespace CudaSharp.Tester;

readonly record struct TileGymPersistentLayerNormProblem(int Rows, int Columns, int SmCount) : ITileGymProblem
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
        var programs = Math.Min(SmCount, 1 + (Rows - 1) / blockN);
        return new TileCppGrid(checked((uint)programs));
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
                var blockNParameter = TileGymHyperparameter.Integer("BlockN", blockN);
                var candidate = new TileGymCandidate([blockNParameter]);
                candidates.Add(candidate);
            }
        }
        return candidates;
    }
}
