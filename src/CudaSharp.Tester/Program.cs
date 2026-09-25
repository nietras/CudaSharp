using System;
using System.IO;
using CudaSharp.Tester;

var device = 0;
var elements = 1 << 20;
var matmulSize = 64;
int? matmulWarmup = null;
int? matmulIterations = null;
int? matmulTileM = null;
int? matmulTileN = null;
int? matmulTileK = null;
int? matmulOccupancy = null;
var skipMatmulValidation = false;
var output = Path.Combine(AppContext.BaseDirectory, "tilegym-results");
string? filter = null;
var list = false;
var tuningChecks = false;
var enableAutotuning = true;
for (var i = 0; i < args.Length; i++)
{
    if (args[i] == "--device" && i + 1 < args.Length)
    {
        device = int.Parse(args[++i]);
    }
    else if (args[i] == "--elements" && i + 1 < args.Length)
    {
        elements = int.Parse(args[++i]);
    }
    else if (args[i] == "--matmul-size" && i + 1 < args.Length)
    {
        matmulSize = int.Parse(args[++i]);
    }
    else if (args[i] == "--matmul-warmup" && i + 1 < args.Length)
    {
        matmulWarmup = int.Parse(args[++i]);
    }
    else if (args[i] == "--matmul-iterations" && i + 1 < args.Length)
    {
        matmulIterations = int.Parse(args[++i]);
    }
    else if (args[i] == "--matmul-tile-m" && i + 1 < args.Length)
    {
        matmulTileM = int.Parse(args[++i]);
    }
    else if (args[i] == "--matmul-tile-n" && i + 1 < args.Length)
    {
        matmulTileN = int.Parse(args[++i]);
    }
    else if (args[i] == "--matmul-tile-k" && i + 1 < args.Length)
    {
        matmulTileK = int.Parse(args[++i]);
    }
    else if (args[i] == "--matmul-occupancy" && i + 1 < args.Length)
    {
        matmulOccupancy = int.Parse(args[++i]);
    }
    else if (args[i] == "--skip-matmul-validation")
    {
        skipMatmulValidation = true;
    }
    else if (args[i] == "--output" && i + 1 < args.Length)
    {
        output = args[++i];
    }
    else if (args[i] == "--filter" && i + 1 < args.Length)
    {
        filter = args[++i];
    }
    else if (args[i] == "--list")
    {
        list = true;
    }
    else if (args[i] == "--tuning-checks")
    {
        tuningChecks = true;
    }
    else if (args[i] == "--no-autotune")
    {
        enableAutotuning = false;
    }
}

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

using var runtime = new TileGymRuntime(device);
runtime.EnableAutotuning = enableAutotuning;
var report = new TileGymReport();
Console.WriteLine($"CUDA Tile C++ SM {runtime.Architecture}");
var scenarios = TileGymCatalog.Select(filter);
var selected = false;
foreach (var scenario in scenarios)
{
    selected = true;
    if (scenario.Family == "activation")
    {
        TileGymActivationScenarios.RunAll(runtime, report, elements);
    }
    else
    {
        if (scenario.Family == "matmul-bmm")
        {
            if (matmulSize == 64 && matmulWarmup is null && matmulIterations is null &&
                matmulTileM is null && matmulTileN is null && matmulTileK is null &&
                matmulOccupancy is null && !skipMatmulValidation)
            {
                scenario.Run(runtime, report);
            }
            else
            {
                TileGymMatrixScenarios.RunMatmulOnly(runtime, report, matmulSize, matmulWarmup, matmulIterations,
                    skipMatmulValidation, matmulTileM, matmulTileN, matmulTileK, matmulOccupancy);
            }
        }
        else
        {
            scenario.Run(runtime, report);
        }
    }
}

if (!selected)
{
    throw new ArgumentException($"No TileGym scenario family matches '{filter}'.", nameof(filter));
}

report.Write(output);
Console.WriteLine(report.ToMarkdown());
Console.WriteLine($"Reports: {output}");
