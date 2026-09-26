using System;
using System.Collections.Generic;
using System.Linq;

namespace CudaSharp.Tester;

static class TileGymMatmulCandidates
{
    public static TileGymCandidate Create(int tileM, int tileN, int tileK,
        int groupM = 8, int numCtas = 1, int occupancy = 1)
    {
        if (tileM <= 0 || tileN <= 0 || tileK <= 0 || groupM <= 0 ||
            numCtas <= 0 || occupancy <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(tileM), "Tile and scheduling parameters must be positive.");
        }
        return new TileGymCandidate(
        [
            TileGymHyperparameter.Integer("TileM", tileM),
            TileGymHyperparameter.Integer("TileN", tileN),
            TileGymHyperparameter.Integer("TileK", tileK),
            TileGymHyperparameter.Integer("GroupM", groupM),
            TileGymHyperparameter.Integer("NumCtas", numCtas),
            TileGymHyperparameter.Integer("Occupancy", occupancy)
        ]);
    }

    public static IReadOnlyList<TileGymCandidate> For(TileGymMatmulProblem problem)
    {
        var candidates = problem.Persistent ? Persistent(problem.Architecture) : Standard(problem.Architecture);
        return candidates.Where(candidate => candidate.GetInt32("NumCtas") <= problem.SmCount).ToArray();
    }

    public static IReadOnlyList<TileGymCandidate> Select(TileGymMatmulProblem problem,
        int? tileM, int? tileN, int? tileK, int? occupancy)
    {
        if (tileM is null && tileN is null && tileK is null && occupancy is null)
        {
            return For(problem);
        }
        return [Create(tileM ?? (problem.M == 64 ? 64 : 128), tileN ?? 64,
            tileK ?? 64, occupancy: occupancy ?? 1)];
    }

    static IReadOnlyList<TileGymCandidate> Standard(int arch)
    {
        if (arch is 120 or 121)
        {
            return [Create(128, 64, 64), Create(128, 64, 32, occupancy: 2)];
        }
        if (arch < 90)
        {
            var configs = new List<TileGymCandidate>();
            foreach (var tileM in new[] { 64, 128 })
            {
                foreach (var tileN in new[] { 64, 128 })
                {
                    foreach (var tileK in new[] { 32, 64, 128 })
                    {
                        foreach (var occupancy in new[] { 1, 2 })
                        {
                            configs.Add(Create(tileM, tileN, tileK, occupancy: occupancy));
                        }
                    }
                }
            }
            return configs;
        }
        return
        [
            Create(128, 128, 32),
            Create(256, 256, 64, numCtas: 2),
            Create(256, 256, 64, numCtas: 4),
            Create(512, 256, 64, numCtas: 2)
        ];
    }

    static IReadOnlyList<TileGymCandidate> Persistent(int arch)
    {
        if (arch is 120 or 121)
        {
            return
            [
                Create(64, 64, 64, occupancy: 2),
                Create(64, 64, 64, occupancy: 4),
                Create(64, 64, 64),
                Create(128, 64, 64, occupancy: 2),
                Create(128, 64, 64),
                Create(128, 64, 64, occupancy: 4),
                Create(256, 256, 64)
            ];
        }
        if (arch < 90)
        {
            return
            [
                Create(64, 64, 32, occupancy: 2),
                Create(64, 128, 32, occupancy: 2),
                Create(128, 64, 32, occupancy: 2),
                Create(128, 128, 32),
                Create(128, 128, 32, occupancy: 2)
            ];
        }
        return
        [
            Create(128, 512, 64, numCtas: 4),
            Create(256, 256, 64, numCtas: 2),
            Create(256, 256, 64),
            Create(256, 256, 128, numCtas: 2),
            Create(128, 128, 64)
        ];
    }
}
