using System;
using CudaSharp.Tile;

namespace CudaSharp.Tester;

readonly record struct TileGymMatmulProblem(
    int M, int N, int K, string ElementType, bool TransposeA, bool TransposeB,
    bool Persistent, int Architecture, int SmCount)
{
    public string KernelName => Persistent ? "static_persistent_matmul_kernel" : "matmul_kernel";
    public string Header => Persistent ? "persistent_matmul.cuh" : "matmul.cuh";

    public string TemplateArguments(TileGymCandidate candidate)
    {
        ArgumentNullException.ThrowIfNull(candidate);
        var tileK = candidate.GetInt32("TileK");
        var common = $"{ElementType}, {M}, {N}, {K}, {candidate["TileM"]}, {candidate["TileN"]}, " +
            $"{candidate["TileK"]}, {candidate["GroupM"]}";
        var kTiles = Persistent ? "" : $", {CeilDiv(K, tileK)}";
        return $"{common}{kTiles}, {Bool(TransposeA)}, {Bool(TransposeB)}, " +
            $"{candidate["NumCtas"]}, {candidate["Occupancy"]}";
    }

    public TileCppGrid Grid(TileGymCandidate candidate)
    {
        ArgumentNullException.ThrowIfNull(candidate);
        var tileM = candidate.GetInt32("TileM");
        var tileN = candidate.GetInt32("TileN");
        var numCtas = candidate.GetInt32("NumCtas");
        var occupancy = candidate.GetInt32("Occupancy");
        if (M <= 0 || N <= 0 || K <= 0 || tileM <= 0 || tileN <= 0 ||
            candidate.GetInt32("TileK") <= 0 || numCtas <= 0 || occupancy <= 0 ||
            Persistent && SmCount < numCtas)
        {
            throw new ArgumentOutOfRangeException(nameof(candidate), "Invalid matmul dimensions, tile size or grid parameters.");
        }
        var tiles = checked((long)CeilDiv(M, tileM) * CeilDiv(N, tileN));
        var blocks = Persistent ? checked(Math.Min(SmCount / numCtas, tiles) * occupancy) : tiles;
        return new TileCppGrid(checked((uint)blocks));
    }

    static int CeilDiv(int value, int divisor) => 1 + (value - 1) / divisor;
    static string Bool(bool value) => value ? "true" : "false";
}
