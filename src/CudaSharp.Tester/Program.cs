using System;
using System.IO;
using CudaSharp.Tester;

var device = 0;
var elements = 1 << 20;
var output = Path.Combine(AppContext.BaseDirectory, "tilegym-results");
string? filter = null;
var list = false;
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
        scenario.Run(runtime, report);
    }
}

if (!selected)
{
    throw new ArgumentException($"No TileGym scenario family matches '{filter}'.", nameof(filter));
}

report.Write(output);
Console.WriteLine(report.ToMarkdown());
Console.WriteLine($"Reports: {output}");
