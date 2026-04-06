#if DEBUG
#define USEMANUALCONFIG
#endif
// Type 'Program' can be sealed because it has no subtypes in its containing assembly and is not externally visible
#pragma warning disable CA1852
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;
using BenchmarkDotNet.Columns;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Exporters;
using BenchmarkDotNet.Loggers;
using BenchmarkDotNet.Reports;
using BenchmarkDotNet.Running;
using CudaSharp;
using CudaSharp.ComparisonBenchmarks;
using Perfolizer.Helpers;
#if USEMANUALCONFIG
using BenchmarkDotNet.Jobs;
using Perfolizer.Horology;
#endif

[assembly: System.Runtime.InteropServices.ComVisible(false)]

Action<string> log = t => { Console.WriteLine(t); Trace.WriteLine(t); };

log($"{Environment.Version} args: {args.Length} versions: {GetVersions()}");

if (TryRunSerialLaunchProfile(args))
{
    return;
}

// Use args as switch to run BDN or not e.g. BDN only run when using script
if (args.Length > 0)
{
    var exporter = new CustomMarkdownExporter();

    var baseConfig = ManualConfig.CreateEmpty()
        .AddColumnProvider(DefaultColumnProviders.Instance)
        .AddExporter(exporter)
        .AddLogger(ConsoleLogger.Default);

    var config =
#if USEMANUALCONFIG
        baseConfig
#else
        (Debugger.IsAttached ? new DebugInProcessConfig() : DefaultConfig.Instance)
#endif
        .WithSummaryStyle(SummaryStyle.Default.WithMaxParameterColumnWidth(120))
        .WithOption(ConfigOptions.JoinSummary, true)
#if USEMANUALCONFIG
        .AddJob(Job.InProcess.WithIterationTime(TimeInterval.FromMilliseconds(100)).WithMinIterationCount(2).WithMaxIterationCount(5))
#endif
        ;

    //BenchmarkSwitcher.FromAssembly(Assembly.GetExecutingAssembly()).Run(args, config);

    var nameToBenchTypesSet = new Dictionary<string, Type[]>()
    {
        { nameof(LaunchKernelBench), new[] { typeof(LaunchKernelBench), } },
        { nameof(SerialLaunchKernelBench), new[] { typeof(SerialLaunchKernelBench), } },
        { nameof(TestBench), new[] { typeof(TestBench), } },
    };
    foreach (var (name, benchTypes) in nameToBenchTypesSet)
    {
        var summaries = BenchmarkRunner.Run(benchTypes, config, args);
        foreach (var s in summaries)
        {
            var cpuInfo = s.HostEnvironmentInfo.Cpu.Value;
            var processorName = CpuBrandHelper.ToShortBrandName(cpuInfo);
            var processorNameInDirectory = processorName
                .Replace(" Processor", "").Replace(" CPU", "")
                .Replace(" ", ".").Replace("/", "").Replace("\\", "")
                .Replace(".Graphics", "");
            log(processorName);

            var sourceDirectory = GetSourceDirectory();
            var directory = $"{sourceDirectory}/../../benchmarks/{processorNameInDirectory}";
            if (!Directory.Exists(directory)) { Directory.CreateDirectory(directory); }
            var filePath = Path.Combine(directory, $"{name}.md");

            using var logger = new StreamLogger(filePath);
            exporter.ExportToLog(s, logger);

            var versions = GetVersions();
            File.WriteAllText(Path.Combine(directory, "Versions.txt"), versions);
        }
    }
}
else
{
}

static string GetVersions() =>
     $"CudaSharp {GetFileVersion(typeof(nvcuda).Assembly)}, " +
     $"System {GetFileVersion(typeof(System.Exception).Assembly)}";

static string GetFileVersion(Assembly assembly) =>
    FileVersionInfo.GetVersionInfo(assembly.Location).FileVersion!;

static string GetSourceDirectory([CallerFilePath] string filePath = "") => Path.GetDirectoryName(filePath)!;

static bool TryRunSerialLaunchProfile(string[] args)
{
    if (!args.Contains("--profile-serial-launch", StringComparer.OrdinalIgnoreCase))
    {
        return false;
    }

    var variant = GetOption(args, "--variant") ?? "raw";
    var warmupCount = GetIntOption(args, "--warmup", 10);
    var repetitionCount = GetIntOption(args, "--repetitions", 1000);
    var serialLaunchCount = GetIntOption(args, "--serial-launch-count", 256);

    var bench = new SerialLaunchKernelBench
    {
        SerialLaunchCount = serialLaunchCount,
    };

    var benchmarkName = variant.ToLowerInvariant() switch
    {
        "raw" => nameof(SerialLaunchKernelBench.cuLaunchKernel_Raw_SerialTripleBuffer_StreamSync),
        "graph" => nameof(SerialLaunchKernelBench.cuGraphLaunch_SerialTripleBuffer_StreamSync),
        "device-graph" => nameof(SerialLaunchKernelBench.cuGraphLaunch_DeviceLaunchCapableSerialTripleBuffer_StreamSync),
        "captured" => nameof(SerialLaunchKernelBench.cuGraphLaunch_CapturedSerialTripleBuffer_StreamSync),
        _ => throw new ArgumentOutOfRangeException(nameof(args),
            $"Unsupported serial profile variant '{variant}'. Use raw, graph, device-graph, or captured."),
    };

    Action run = benchmarkName switch
    {
        nameof(SerialLaunchKernelBench.cuLaunchKernel_Raw_SerialTripleBuffer_StreamSync) =>
            bench.cuLaunchKernel_Raw_SerialTripleBuffer_StreamSync,
        nameof(SerialLaunchKernelBench.cuGraphLaunch_SerialTripleBuffer_StreamSync) =>
            bench.cuGraphLaunch_SerialTripleBuffer_StreamSync,
        nameof(SerialLaunchKernelBench.cuGraphLaunch_DeviceLaunchCapableSerialTripleBuffer_StreamSync) =>
            bench.cuGraphLaunch_DeviceLaunchCapableSerialTripleBuffer_StreamSync,
        nameof(SerialLaunchKernelBench.cuGraphLaunch_CapturedSerialTripleBuffer_StreamSync) =>
            bench.cuGraphLaunch_CapturedSerialTripleBuffer_StreamSync,
        _ => throw new UnreachableException(),
    };

    Console.WriteLine(
        $"Profiling serial launch variant '{variant}' with SerialLaunchCount={serialLaunchCount}, warmup={warmupCount}, repetitions={repetitionCount}.");

    bench.Setup();
    try
    {
        for (var i = 0; i < warmupCount; i++)
        {
            run();
        }

        var stopwatch = Stopwatch.StartNew();
        for (var i = 0; i < repetitionCount; i++)
        {
            run();
        }

        stopwatch.Stop();
        Console.WriteLine(
            $"Completed {repetitionCount} profiled invocations in {stopwatch.Elapsed.TotalMilliseconds:F3} ms.");
    }
    finally
    {
        bench.Cleanup();
    }

    return true;
}

static string? GetOption(string[] args, string name)
{
    for (var i = 0; i < args.Length - 1; i++)
    {
        if (string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase))
        {
            return args[i + 1];
        }
    }

    return null;
}

static int GetIntOption(string[] args, string name, int defaultValue)
{
    var value = GetOption(args, name);
    return value is not null && int.TryParse(value, out var parsed)
        ? parsed
        : defaultValue;
}

class CustomMarkdownExporter : MarkdownExporter
{
    public CustomMarkdownExporter()
    {
        Dialect = "GitHub";
        UseCodeBlocks = true;
        CodeBlockStart = "```";
        StartOfGroupHighlightStrategy = MarkdownHighlightStrategy.None;
        ColumnsStartWithSeparator = true;
        EscapeHtml = true;
    }
}
