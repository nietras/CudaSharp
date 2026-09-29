namespace CudaSharp.Tester;

sealed record TileGymMatmulOptions(int Size = 64, int? Warmup = null, int? Iterations = null,
    int? TileM = null, int? TileN = null, int? TileK = null, int? Occupancy = null, bool SkipValidation = false)
{
    public bool IsDefault => this == new TileGymMatmulOptions();
}

sealed record TileGymOptions(int Elements = 1 << 20, TileGymMatmulOptions? Matmul = null)
{
    public TileGymMatmulOptions Matmul { get; init; } = Matmul ?? new();
}
