using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace CudaSharp.Tester;

sealed record TileGymResult(
    string Family,
    string Kernel,
    string Shape,
    string Configuration,
    string Status,
    double CompileMilliseconds,
    double TuneMilliseconds,
    double KernelMilliseconds,
    double Throughput,
    string ThroughputUnit,
    string? Diagnostic);

sealed class TileGymReport
{
    readonly List<TileGymResult> _results = [];

    public IReadOnlyList<TileGymResult> Results => _results;
    public void Add(TileGymResult result) => _results.Add(result);

    public void Write(string directory)
    {
        Directory.CreateDirectory(directory);
        File.WriteAllText(Path.Combine(directory, "tilegym-results.md"), ToMarkdown());
        File.WriteAllText(Path.Combine(directory, "tilegym-results.csv"), ToCsv());
    }

    public string ToMarkdown()
    {
        var text = new StringBuilder();
        text.AppendLine("# CudaSharp TileGym performance");
        text.AppendLine();
        text.AppendLine(
            "| Family | Kernel | Shape | Configuration | Status | Compile ms | Tune ms | Kernel ms | Throughput |");
        text.AppendLine("|---|---|---|---|---:|---:|---:|---:|---:|");
        foreach (var result in _results)
        {
            text.Append("| ")
                .Append(result.Family)
                .Append(" | ")
                .Append(result.Kernel)
                .Append(" | ")
                .Append(result.Shape)
                .Append(" | ")
                .Append(result.Configuration)
                .Append(" | ")
                .Append(result.Status)
                .Append(" | ")
                .Append(Format(result.CompileMilliseconds))
                .Append(" | ")
                .Append(Format(result.TuneMilliseconds))
                .Append(" | ")
                .Append(Format(result.KernelMilliseconds))
                .Append(" | ")
                .Append(Format(result.Throughput))
                .Append(' ')
                .Append(result.ThroughputUnit)
                .AppendLine(" |");
        }
        return text.ToString();
    }

    string ToCsv()
    {
        var text = new StringBuilder(
            "Family,Kernel,Shape,Configuration,Status,CompileMilliseconds,TuneMilliseconds,KernelMilliseconds,Throughput,ThroughputUnit,Diagnostic\n");
        foreach (var result in _results)
        {
            text.AppendLine(
                string.Join(',', new[]
                {
                    result.Family,
                    result.Kernel,
                    result.Shape,
                    result.Configuration,
                    result.Status,
                    Format(result.CompileMilliseconds),
                    Format(result.TuneMilliseconds),
                    Format(result.KernelMilliseconds),
                    Format(result.Throughput),
                    result.ThroughputUnit,
                    result.Diagnostic ?? string.Empty,
                }.Select(Escape)));
        }
        return text.ToString();
    }

    static string Format(double value) => value.ToString("0.######", CultureInfo.InvariantCulture);
    static string Escape(string value) => $"\"{value.Replace("\"", "\"\"")}\"";
}
