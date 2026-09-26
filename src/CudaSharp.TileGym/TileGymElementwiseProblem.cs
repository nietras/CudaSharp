using System;
using CudaSharp.Tile;

namespace CudaSharp.Tester;

readonly record struct TileGymElementwiseProblem(int Count, int Operation)
{
    public string TemplateArguments(TileGymCandidate candidate)
        => $"float, {candidate["BlockSize"]}, {Operation}";

    public TileCppGrid Grid(TileGymCandidate candidate)
    {
        var blockSize = candidate.GetInt32("BlockSize");
        if (Count <= 0 || blockSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(candidate));
        }
        return new TileCppGrid(checked((uint)(1 + (Count - 1) / blockSize)));
    }
}

static class TileGymElementwiseCandidates
{
    public static TileGymCandidate[] For(int count)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(count);
        return
        [
            Create(256),
            Create(512),
            Create(1024)
        ];
    }

    static TileGymCandidate Create(int blockSize)
        => new([TileGymHyperparameter.Integer("BlockSize", blockSize)]);
}
