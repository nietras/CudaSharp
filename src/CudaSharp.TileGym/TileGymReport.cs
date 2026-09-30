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
    double? CompileMilliseconds,
    double TuneMilliseconds,
    double KernelMilliseconds,
    double Throughput,
    string ThroughputUnit,
    string? Diagnostic,
    double? HostMilliseconds = null,
    double? LoadMilliseconds = null,
    double? FirstLaunchMilliseconds = null,
    int Candidates = 1);

sealed class TileGymReport
{
    readonly List<TileGymResult> _results = [];

    public IReadOnlyList<TileGymResult> Results => _results;
    public void Add(TileGymResult result) => _results.Add(result);

    public void Add(string family, string shape, nuint bytes, TileGymRun run)
    {
        ArgumentNullException.ThrowIfNull(run);
        var selection = run.Selection;
        var searched = selection.Candidates > 1;
        string? diagnostic = null;
        if (searched || selection.Rejections.Count > 0)
        {
            diagnostic = $"{selection.Candidates} candidates, {selection.Rejections.Count} rejected.";
            if (selection.Rejections.Count > 0)
            {
                diagnostic += $" First rejection: {selection.Rejections[0]}";
            }
        }
        Add(new TileGymResult(family, run.Kernel.Name, shape, selection.Variant.Label,
            searched ? "Passed (searched)" : "Passed",
            selection.Compiled.CompileMilliseconds, selection.TuneMilliseconds, run.KernelMilliseconds,
            bytes / (run.KernelMilliseconds * 1_000_000.0), "GB/s", diagnostic, run.HostMilliseconds,
            selection.Compiled.LoadMilliseconds, selection.FirstLaunchMilliseconds, selection.Candidates));
    }

    public void Write(string directory)
    {
        Directory.CreateDirectory(directory);
        var markdownPath = Path.Combine(directory, "tilegym-results.md");
        var markdown = ToMarkdown();
        File.WriteAllText(markdownPath, markdown);
        var csvPath = Path.Combine(directory, "tilegym-results.csv");
        var csv = ToCsv();
        File.WriteAllText(csvPath, csv);
    }

    public string ToMarkdown()
    {
        var text = new StringBuilder();
        text.AppendLine("# CudaSharp TileGym performance");
        text.AppendLine();
        text.AppendLine(
            "| Family | Kernel | Shape | Configuration | Status | Compile ms | Load ms | First launch ms | Candidates | Tune ms | Host ms | Kernel ms | Throughput |");
        text.AppendLine("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
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
                .Append(Format(result.LoadMilliseconds))
                .Append(" | ")
                .Append(Format(result.FirstLaunchMilliseconds))
                .Append(" | ")
                .Append(result.Candidates)
                .Append(" | ")
                .Append(Format(result.TuneMilliseconds))
                .Append(" | ")
                .Append(Format(result.HostMilliseconds))
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

    public string ToCsv()
    {
        var text = new StringBuilder(
            "Family,Kernel,Shape,Configuration,Status,CompileMilliseconds,LoadMilliseconds," +
            "FirstLaunchMilliseconds,Candidates,TuneMilliseconds,HostMilliseconds,KernelMilliseconds,Throughput," +
            "ThroughputUnit,Diagnostic\n");
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
                    Format(result.LoadMilliseconds),
                    Format(result.FirstLaunchMilliseconds),
                    result.Candidates.ToString(CultureInfo.InvariantCulture),
                    Format(result.TuneMilliseconds),
                    Format(result.HostMilliseconds),
                    Format(result.KernelMilliseconds),
                    Format(result.Throughput),
                    result.ThroughputUnit,
                    result.Diagnostic ?? string.Empty,
                }.Select(Escape)));
        }
        return text.ToString();
    }

    static string Format(double value) => value.ToString("0.######", CultureInfo.InvariantCulture);
    static string Format(double? value) => value is { } number ? Format(number) : string.Empty;
    static string Escape(string value) => $"\"{value.Replace("\"", "\"\"")}\"";
}
