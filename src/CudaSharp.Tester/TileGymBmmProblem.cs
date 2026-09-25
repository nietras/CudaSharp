using System;
using System.Collections.Generic;
using CudaSharp.Tile;

namespace CudaSharp.Tester;

readonly record struct TileGymBmmProblem(
    int Batch, int M, int N, int K, string ElementType, bool TransposeA, bool TransposeB,
    bool Persistent, int Architecture, int SmCount)
{
    public string TemplateArguments(TileGymCandidate candidate)
    {
        ArgumentNullException.ThrowIfNull(candidate);
        var common = $"{ElementType}, {candidate["TileM"]}, {candidate["TileN"]}, " +
            $"{candidate["TileK"]}, {candidate["GroupM"]}, " +
            $"{Bool(TransposeA)}, {Bool(TransposeB)}";
        return Persistent
            ? $"{common}, {Batch}, {M}, {N}, {K}, {candidate["NumCtas"]}, {candidate["Occupancy"]}"
            : common;
    }

    public TileCppGrid Grid(TileGymCandidate candidate)
    {
        ArgumentNullException.ThrowIfNull(candidate);
        var tileM = candidate.GetInt32("TileM");
        var tileN = candidate.GetInt32("TileN");
        var numCtas = candidate.GetInt32("NumCtas");
        var occupancy = candidate.GetInt32("Occupancy");
        if (Batch <= 0 || M <= 0 || N <= 0 || K <= 0 || tileM <= 0 || tileN <= 0 ||
            candidate.GetInt32("TileK") <= 0 || numCtas <= 0 || occupancy <= 0 ||
            Persistent && SmCount < numCtas)
        {
            throw new ArgumentOutOfRangeException(nameof(candidate), "Invalid BMM dimensions or tile configuration.");
        }
        var tiles = checked((long)CeilDiv(M, tileM) * CeilDiv(N, tileN));
        if (!Persistent)
        {
            return new TileCppGrid(checked((uint)tiles), checked((uint)Batch));
        }
        var programs = checked(Math.Min(SmCount / numCtas, tiles * Batch) * occupancy);
        return new TileCppGrid(checked((uint)programs));
    }

    static int CeilDiv(int value, int divisor) => 1 + (value - 1) / divisor;
    static string Bool(bool value) => value ? "true" : "false";
}

static class TileGymBmmCandidates
{
    public static IReadOnlyList<TileGymCandidate> For(TileGymBmmProblem problem)
    {
        if (!problem.Persistent)
        {
            return
            [
                TileGymMatmulCandidates.Create(64, 64, 32),
                TileGymMatmulCandidates.Create(64, 64, 64),
                TileGymMatmulCandidates.Create(128, 64, 32)
            ];
        }

        var configs = new List<TileGymCandidate>();
        if (problem.Architecture is 120 or 121)
        {
            foreach (var tileM in new[] { 64, 128 })
            {
                foreach (var tileN in new[] { 64, 128 })
                {
                    foreach (var tileK in new[] { 32, 64 })
                    {
                        foreach (var occupancy in new[] { 1, 2, 4 })
                        {
                            configs.Add(TileGymMatmulCandidates.Create(tileM, tileN, tileK,
                                occupancy: occupancy));
                        }
                    }
                }
            }
        }
        else if (problem.Architecture < 90)
        {
            foreach (var tileM in new[] { 64, 128 })
            {
                foreach (var tileN in new[] { 64, 128 })
                {
                    foreach (var tileK in new[] { 32, 64, 128 })
                    {
                        foreach (var occupancy in new[] { 1, 2 })
                        {
                            configs.Add(TileGymMatmulCandidates.Create(tileM, tileN, tileK,
                                occupancy: occupancy));
                        }
                    }
                }
            }
        }
        else if (problem.Architecture == 90)
        {
            foreach (var tileM in new[] { 64, 128, 256 })
            {
                foreach (var tileN in new[] { 64, 128, 256 })
                {
                    foreach (var occupancy in new[] { 1, 2 })
                    {
                        configs.Add(TileGymMatmulCandidates.Create(tileM, tileN, 64,
                            numCtas: 2, occupancy: occupancy));
                    }
                }
            }
        }
        else
        {
            foreach (var tileM in new[] { 128, 256 })
            {
                configs.Add(TileGymMatmulCandidates.Create(tileM, 256, 64, numCtas: 2));
            }
        }
        return configs.FindAll(candidate => candidate.GetInt32("NumCtas") <= problem.SmCount);
    }
}
