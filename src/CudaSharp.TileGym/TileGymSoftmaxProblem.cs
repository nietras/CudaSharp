using System;
using System.Collections.Generic;
using CudaSharp.Tile;

namespace CudaSharp.Tester;

readonly record struct TileGymSoftmaxProblem(int Rows, int Columns, bool Online, bool Backward)
{
    public string TemplateArguments(TileGymCandidate candidate) =>
        $"float, {candidate["BlockSize"]}{(!Online && !Backward ? ", 0" : "")}";

    public TileCppGrid Grid(TileGymCandidate candidate)
    {
        if (Rows <= 0 || Columns <= 0 || candidate.GetInt32("BlockSize") <= 0 ||
            !Online && candidate.GetInt32("BlockSize") < Columns)
        {
            throw new ArgumentOutOfRangeException(nameof(candidate));
        }
        return new TileCppGrid(checked((uint)Rows));
    }
}

static class TileGymSoftmaxCandidates
{
    public static IReadOnlyList<TileGymCandidate> For(TileGymSoftmaxProblem problem)
    {
        var candidates = new List<TileGymCandidate>();
        foreach (var block in new[] { 128, 256, 512 })
        {
            if (problem.Online || block >= problem.Columns)
            {
                candidates.Add(new TileGymCandidate([TileGymHyperparameter.Integer("BlockSize", block)]));
            }
        }
        return candidates;
    }
}
