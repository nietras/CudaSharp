using System.IO;
using System.Linq;
using CudaSharp.Tester;

var device = 0;
var options = new TileGymOptions();
var matmul = options.Matmul;
var output = Path.Combine(AppContext.BaseDirectory, "tilegym-results");
string? filter = null;
var list = false;
var tuningChecks = false;
var enableAutotuning = true;
// The driver serializes cuModuleLoadData (TileIR JIT); concurrent loads only add contention and memory.
var loadParallelism = 1;
for (var i = 0; i < args.Length; i++)
{
    var hasValue = i + 1 < args.Length;
    switch (args[i])
    {
        case "--device" when hasValue: device = int.Parse(args[++i]); break;
        case "--elements" when hasValue: options = options with { Elements = int.Parse(args[++i]) }; break;
        case "--matmul-size" when hasValue: matmul = matmul with { Size = int.Parse(args[++i]) }; break;
        case "--matmul-warmup" when hasValue: matmul = matmul with { Warmup = int.Parse(args[++i]) }; break;
        case "--matmul-iterations" when hasValue: matmul = matmul with { Iterations = int.Parse(args[++i]) }; break;
        case "--matmul-tile-m" when hasValue: matmul = matmul with { TileM = int.Parse(args[++i]) }; break;
        case "--matmul-tile-n" when hasValue: matmul = matmul with { TileN = int.Parse(args[++i]) }; break;
        case "--matmul-tile-k" when hasValue: matmul = matmul with { TileK = int.Parse(args[++i]) }; break;
        case "--matmul-occupancy" when hasValue: matmul = matmul with { Occupancy = int.Parse(args[++i]) }; break;
        case "--skip-matmul-validation": matmul = matmul with { SkipValidation = true }; break;
        case "--output" when hasValue: output = args[++i]; break;
        case "--filter" when hasValue: filter = args[++i]; break;
        case "--list": list = true; break;
        case "--tuning-checks": tuningChecks = true; break;
        case "--no-autotune": enableAutotuning = false; break;
        case "--load-parallelism" when hasValue: loadParallelism = int.Parse(args[++i]); break;
    }
}
options = options with { Matmul = matmul };

if (tuningChecks)
{
    TileGymTuningChecks.Run();
    Console.WriteLine("TileGym tuning checks passed.");
    return;
}

if (list)
{
    foreach (var scenario in TileGymCatalog.Scenarios)
    {
        Console.WriteLine(scenario.Family);
    }

    return;
}

using var runtime = new TileGymRuntime(device) { EnableAutotuning = enableAutotuning };
Console.WriteLine($"CUDA Tile C++ SM {runtime.Architecture}");
var benchmarks = TileGymCatalog.Select(filter).SelectMany(s => s.Create(runtime, options)).ToArray();
if (benchmarks.Length == 0)
{
    throw new ArgumentException($"No TileGym scenario family matches '{filter}'.", nameof(filter));
}

var precompile = runtime.Precompile(benchmarks, loadParallelism);
precompile.WriteTo(Console.Out);
precompile.Write(output);
var report = new TileGymReport();
foreach (var benchmark in benchmarks)
{
    benchmark.Run(runtime, report);
}

report.Write(output);
Console.WriteLine(report.ToMarkdown());
Console.WriteLine($"Reports: {output}");
