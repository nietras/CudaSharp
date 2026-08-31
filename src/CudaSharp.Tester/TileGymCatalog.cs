using System;
using System.Collections.Generic;
using System.Linq;

namespace CudaSharp.Tester;

sealed record TileGymScenario(string Family, Action<TileGymRuntime, TileGymReport> Run);

static class TileGymCatalog
{
    public static IReadOnlyList<TileGymScenario> Scenarios { get; } =
    [
        new TileGymScenario("activation", static (runtime, report) => TileGymActivationScenarios.RunAll(runtime, report, 1 << 20)),
        new TileGymScenario("normalization", TileGymNormalizationScenarios.RunAll),
        new TileGymScenario("rope-softmax", TileGymRopeSoftmaxScenarios.RunAll),
        new TileGymScenario("attention-decode", TileGymAttentionScenarios.RunAll),
        new TileGymScenario("mla-splitk", TileGymMlaScenarios.RunAll),
        new TileGymScenario("recurrent-dropout", TileGymRecurrentScenarios.RunAll),
        new TileGymScenario("moe-alignment", TileGymMoeScenarios.RunAll),
        new TileGymScenario("matmul-bmm", TileGymMatrixScenarios.RunAll),
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
