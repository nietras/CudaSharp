using System.Collections.Generic;
using System.Linq;

namespace CudaSharp.Tester;

delegate IEnumerable<TileGymBenchmark> TileGymBenchmarkFactory(TileGymRuntime runtime, TileGymOptions options);

sealed record TileGymScenario(string Family, TileGymBenchmarkFactory Create);

static class TileGymCatalog
{
    public static IReadOnlyList<TileGymScenario> Scenarios { get; } =
    [
        new("activation", TileGymActivationScenarios.Create),
        new("normalization", TileGymNormalizationScenarios.Create),
        new("rope-softmax", TileGymRopeSoftmaxScenarios.Create),
        new("attention-decode", TileGymAttentionScenarios.Create),
        new("mla-splitk", TileGymMlaScenarios.Create),
        new("recurrent-dropout", TileGymRecurrentScenarios.Create),
        new("moe-alignment", TileGymMoeScenarios.Create),
        new("matmul-bmm", TileGymMatrixScenarios.Create),
        new("convolution", TileGymConvolutionScenarios.Create),
    ];

    public static IEnumerable<TileGymScenario> Select(string? filter)
    {
        if (string.IsNullOrWhiteSpace(filter))
        {
            return Scenarios;
        }

        var terms = filter.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

        return Scenarios.Where(s => terms.Any(t => s.Family.Contains(t, StringComparison.OrdinalIgnoreCase)));
    }
}
