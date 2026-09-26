using System;
using CudaSharp.Tile;

namespace CudaSharp.Tester;

readonly record struct TileGymDropoutProblem(int Count, float Probability, uint Seed)
{
    public string TemplateArguments(TileGymCandidate candidate) =>
        FormattableString.Invariant($"float, {candidate["BlockSize"]}, {Count}, {Probability:R}f, {Seed}u");

    public TileCppGrid Grid(TileGymCandidate candidate)
    {
        var block = candidate.GetInt32("BlockSize");
        if (Count <= 0 || block <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(candidate));
        }
        return new TileCppGrid(checked((uint)(1 + (Count - 1) / block)));
    }
}

static class TileGymDropoutCandidates
{
    public static TileGymCandidate[] For()
        =>
        [
            new([TileGymHyperparameter.Integer("BlockSize", 256)]),
            new([TileGymHyperparameter.Integer("BlockSize", 512)]),
            new([TileGymHyperparameter.Integer("BlockSize", 1024)])
        ];
}
