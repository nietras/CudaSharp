using System;
using System.Collections.Generic;
using CudaSharp.Tile;

namespace CudaSharp.Tester;

enum TileGymAttentionKind
{
    Forward,
    GemmaForward,
    BackwardPreprocess,
    BackwardMain
}

readonly record struct TileGymAttentionProblem(
    TileGymAttentionKind Kind, int Sequence, int Dimension, bool Causal, int Architecture)
{
    public string TemplateArguments(TileGymCandidate candidate)
    {
        ArgumentNullException.ThrowIfNull(candidate);
        var blockM = candidate["BlockM"];
        var blockN = candidate["BlockN"];
        var occupancy = candidate["Occupancy"];
        return Kind switch
        {
            TileGymAttentionKind.Forward =>
                $"float, 1, 1, 1, {Sequence}, {Sequence}, {blockM}, {blockN}, {Dimension}, " +
                $"{Bool(Causal)}, true, {occupancy}, {candidate["NumCtas"]}",
            TileGymAttentionKind.GemmaForward =>
                $"float, 1, 1, 1, {Sequence}, {Sequence}, {blockM}, {blockN}, {Dimension}, " +
                $"{Bool(Causal)}, 0, false, {occupancy}",
            TileGymAttentionKind.BackwardPreprocess =>
                $"float, 1, 1, {Sequence}, {blockM}, {Dimension}, {occupancy}",
            TileGymAttentionKind.BackwardMain =>
                $"float, 1, 1, {Sequence}, {Sequence}, {blockM}, {blockN}, {Dimension}, " +
                $"{Bool(Causal)}, {occupancy}",
            _ => throw new ArgumentOutOfRangeException(nameof(Kind))
        };
    }

    public TileCppGrid Grid(TileGymCandidate candidate)
    {
        ArgumentNullException.ThrowIfNull(candidate);
        var block = Kind == TileGymAttentionKind.BackwardMain
            ? candidate.GetInt32("BlockN") : candidate.GetInt32("BlockM");
        if (Sequence <= 0 || block <= 0 ||
            Kind == TileGymAttentionKind.Forward &&
            ((Sequence + block - 1) / block) % candidate.GetInt32("NumCtas") != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(candidate), "Invalid attention launch configuration.");
        }
        return new TileCppGrid(checked((uint)(1 + (Sequence - 1) / block)));
    }

    static string Bool(bool value) => value ? "true" : "false";
}

static class TileGymAttentionCandidates
{
    public static IReadOnlyList<TileGymCandidate> For(TileGymAttentionProblem problem)
    {
        if (problem.Sequence < 64 ||
            problem.Kind != TileGymAttentionKind.Forward && problem.Sequence % 64 != 0)
        {
            throw new NotSupportedException("No safe attention tile is available for this sequence length.");
        }
        var candidates = new List<TileGymCandidate>();
        if (problem.Kind == TileGymAttentionKind.GemmaForward)
        {
            if (problem.Architecture < 90)
            {
                candidates.Add(Create(64, 64, 2));
                candidates.Add(Create(128, 64, 2));
            }
            else
            {
                candidates.Add(Create(256, 128, 1));
                candidates.Add(Create(128, 128, 2));
            }
        }
        else if (problem.Kind == TileGymAttentionKind.Forward)
        {
            if (problem.Architecture is 120 or 121)
            {
                candidates.Add(Create(64, 64, 2));
            }
            else if (problem.Architecture < 90)
            {
                candidates.Add(Create(64, 64, 2));
                candidates.Add(Create(128, 64, 2));
            }
            else if (problem.Architecture == 90)
            {
                foreach (var blockM in new[] { 64, 128 })
                {
                    foreach (var blockN in new[] { 64, 128 })
                    {
                        candidates.Add(Create(blockM, blockN, 2));
                    }
                }
            }
            else
            {
                candidates.Add(Create(256, 128, 1));
                candidates.Add(Create(128, 128, 2));
                candidates.Add(Create(256, 128, 2));
                candidates.Add(Create(256, 128, 2, numCtas: 2));
            }
        }
        else
        {
            candidates.Add(Create(64, 64, 2));
        }

        candidates.RemoveAll(candidate => candidate.GetInt32("BlockM") > problem.Sequence ||
            candidate.GetInt32("BlockN") > problem.Sequence ||
            problem.Causal && candidate.GetInt32("BlockM") % candidate.GetInt32("BlockN") != 0 ||
            problem.Kind == TileGymAttentionKind.Forward &&
            ((problem.Sequence + candidate.GetInt32("BlockM") - 1) / candidate.GetInt32("BlockM")) %
            candidate.GetInt32("NumCtas") != 0);
        if (candidates.Count == 0)
        {
            candidates.Add(Create(64, 64, 2));
        }
        if (!candidates.Exists(candidate => candidate.GetInt32("BlockM") == 64 &&
            candidate.GetInt32("BlockN") == 64 && candidate.GetInt32("Occupancy") == 1))
        {
            candidates.Add(Create(64, 64, 1));
        }
        return candidates;
    }

    static TileGymCandidate Create(int blockM, int blockN, int occupancy, int numCtas = 1)
        => new(
        [
            TileGymHyperparameter.Integer("BlockM", blockM),
            TileGymHyperparameter.Integer("BlockN", blockN),
            TileGymHyperparameter.Integer("Occupancy", occupancy),
            TileGymHyperparameter.Integer("NumCtas", numCtas)
        ]);
}
