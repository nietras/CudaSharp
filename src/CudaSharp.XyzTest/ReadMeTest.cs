using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using PublicApiGenerator;
#pragma warning disable CA2007 // Consider calling ConfigureAwait on the awaited task
#pragma warning disable CA1822 // Member does not access instance data and can be marked as static

// Only parallelize on class level to avoid multiple writes to README file
[assembly: Parallelize(Workers = 1, Scope = ExecutionScope.ClassLevel)]

namespace CudaSharp.XyzTest;

[TestClass]
public partial class ReadMeTest
{
    static readonly string s_testSourceFilePath = SourceFile();
    static readonly string s_rootDirectory = Path.GetDirectoryName(s_testSourceFilePath) + @"../../../";
    static readonly string s_readmeFilePath = s_rootDirectory + @"README.md";

    [TestMethod]
    public void ReadMeTest_()
    {
        // Above example code is for demonstration purposes only.
        // Short names and repeated constants are only for demonstration.
    }

#if NET10_0
    // Only update README on latest .NET version to avoid multiple accesses
    [TestMethod]
#endif
    public void ReadMeTest_UpdateBenchmarksInMarkdown()
    {
        var readmeFilePath = s_readmeFilePath;

        var benchmarkFileNameToConfig = new Dictionary<string, (string Description, string ReadmeBefore, string ReadmeEnd, string SectionPrefix)>()
        {
            { "TestBench.md", new("TestBench Benchmark Results", "##### TestBench Benchmark Results", "## Example Catalogue", "###### ") },
        };

        var benchmarksDirectory = Path.Combine(s_rootDirectory, "benchmarks");
        var processorDirectories = Directory.EnumerateDirectories(benchmarksDirectory).ToArray();
        var processors = processorDirectories.Select(LastDirectoryName).ToArray();

        var readmeLines = File.ReadAllLines(readmeFilePath);

        foreach (var (fileName, config) in benchmarkFileNameToConfig)
        {
            var description = config.Description;
            var prefix = config.SectionPrefix;
            var readmeBefore = config.ReadmeBefore;
            var readmeEndLine = config.ReadmeEnd;
            var all = "";
            foreach (var processorDirectory in processorDirectories)
            {
                var contentsFilePath = Path.Combine(processorDirectory, fileName);
                if (File.Exists(contentsFilePath))
                {
                    var versionsFilePath = Path.Combine(processorDirectory, "Versions.txt");
                    var versions = File.ReadAllText(versionsFilePath);
                    var contents = File.ReadAllText(contentsFilePath);
                    var processor = LastDirectoryName(processorDirectory);

                    var section = $"{prefix}{processor} - {description} ({versions})";
                    var benchmarkTable = GetBenchmarkTable(contents);
                    var readmeContents = $"{section}{Environment.NewLine}{Environment.NewLine}{benchmarkTable}{Environment.NewLine}";
                    all += readmeContents;
                }
            }
            readmeLines = ReplaceReadmeLines(readmeLines, [all], readmeBefore, prefix, 0, readmeEndLine, 0);
        }

        var newReadme = string.Join(Environment.NewLine, readmeLines) + Environment.NewLine;
        File.WriteAllText(readmeFilePath, newReadme, System.Text.Encoding.UTF8);

        static string LastDirectoryName(string d) =>
            d.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar).Last();

        static string GetBenchmarkTable(string markdown) =>
            markdown.Substring(markdown.IndexOf('|'));
    }

#if NET10_0
    // Only update README on latest .NET version to avoid multiple accesses
    [TestMethod]
#endif
    public void ReadMeTest_UpdateExampleCodeInMarkdown()
    {
        var testSourceFilePath = s_testSourceFilePath;
        var readmeFilePath = s_readmeFilePath;
        var rootDirectory = s_rootDirectory;

        var readmeLines = File.ReadAllLines(readmeFilePath);

        // Update README examples
        var testSourceLines = File.ReadAllLines(testSourceFilePath);
        var testBlocksToUpdate = new (string StartLineContains, string ReadmeLineBeforeCodeBlock)[]
        {
            (nameof(ReadMeTest_) + "()", "## Example"),
            (nameof(ReadMeTest_) + "()", "### Example - Empty"),
        };
        readmeLines = UpdateReadme(testSourceLines, readmeLines, testBlocksToUpdate,
            sourceStartLineOffset: 2, "    }", sourceEndLineOffset: 0, sourceWhitespaceToRemove: 8);

        var newReadme = string.Join(Environment.NewLine, readmeLines) + Environment.NewLine;
        File.WriteAllText(readmeFilePath, newReadme, System.Text.Encoding.UTF8);
    }

    // Only update public API in README for latest .NET version to keep consistent
#if NET10_0
    [TestMethod]
#endif
    public void ReadMeTest_PublicApi()
    {
        var publicApi = typeof(nvcuda).Assembly.GeneratePublicApi();

        var readmeFilePath = s_readmeFilePath;
        var readmeLines = File.ReadAllLines(readmeFilePath);
        readmeLines = ReplaceReadmeLines(readmeLines, [publicApi],
            "## Public API Reference", "```csharp", 1, "```", 0);

        var newReadme = string.Join(Environment.NewLine, readmeLines) + Environment.NewLine;
        File.WriteAllText(readmeFilePath, newReadme, System.Text.Encoding.UTF8);
    }

    static string[] UpdateReadme(string[] sourceLines, string[] readmeLines,
        (string StartLineContains, string ReadmeLineBefore)[] blocksToUpdate,
        int sourceStartLineOffset, string sourceEndLineStartsWith, int sourceEndLineOffset, int sourceWhitespaceToRemove,
        string readmeStartLineStartsWith = "```csharp", int readmeStartLineOffset = 1,
        string readmeEndLineStartsWith = "```", int readmeEndLineOffset = 0)
    {
        foreach (var (startLineContains, readmeLineBeforeBlock) in blocksToUpdate)
        {
            var sourceExampleLines = SnipLines(sourceLines,
                startLineContains, sourceStartLineOffset,
                sourceEndLineStartsWith, sourceEndLineOffset,
                sourceWhitespaceToRemove);

            readmeLines = ReplaceReadmeLines(readmeLines, sourceExampleLines, readmeLineBeforeBlock,
                readmeStartLineStartsWith, readmeStartLineOffset, readmeEndLineStartsWith, readmeEndLineOffset);
        }

        return readmeLines;
    }

    static string[] ReplaceReadmeLines(string[] readmeLines, string[] newReadmeLines, string readmeLineBeforeBlock,
        string readmeStartLineStartsWith, int readmeStartLineOffset,
        string readmeEndLineStartsWith, int readmeEndLineOffset)
    {
        var readmeLineBeforeIndex = Array.FindIndex(readmeLines,
            l => l.StartsWith(readmeLineBeforeBlock, StringComparison.Ordinal)) + 1;
        if (readmeLineBeforeIndex == 0)
        { throw new ArgumentException($"README line '{readmeLineBeforeBlock}' not found."); }

        return ReplaceReadmeLines(readmeLines, newReadmeLines,
            readmeLineBeforeIndex, readmeStartLineStartsWith, readmeStartLineOffset, readmeEndLineStartsWith, readmeEndLineOffset);
    }

    static string[] ReplaceReadmeLines(string[] readmeLines, string[] newReadmeLines, int readmeLineBeforeIndex,
        string readmeStartLineStartsWith, int readmeStartLineOffset,
        string readmeEndLineStartsWith, int readmeEndLineOffset)
    {
        var readmeLinesSpan = readmeLines.AsSpan(readmeLineBeforeIndex);
        var readmeReplaceStartIndex = Array.FindIndex(readmeLines, readmeLineBeforeIndex,
            l => l.StartsWith(readmeStartLineStartsWith, StringComparison.Ordinal)) + readmeStartLineOffset;
        Debug.Assert(readmeReplaceStartIndex >= 0);
        var readmeReplaceEndIndex = Array.FindIndex(readmeLines, readmeReplaceStartIndex,
            l => l.StartsWith(readmeEndLineStartsWith, StringComparison.Ordinal)) + readmeEndLineOffset;

        readmeLines = readmeLines[..readmeReplaceStartIndex].AsEnumerable()
            .Concat(newReadmeLines)
            .Concat(readmeLines[readmeReplaceEndIndex..]).ToArray();
        return readmeLines;
    }

    static string[] SnipLines(string[] sourceLines,
        string startLineContains, int startLineOffset,
        string endLineStartsWith, int endLineOffset,
        int whitespaceToRemove = 8)
    {
        var sourceStartLine = Array.FindIndex(sourceLines,
            l => l.Contains(startLineContains, StringComparison.Ordinal));
        sourceStartLine += startLineOffset;
        var sourceEndLine = Array.FindIndex(sourceLines, sourceStartLine,
            l => l.StartsWith(endLineStartsWith, StringComparison.Ordinal));
        sourceEndLine += endLineOffset;
        var sourceExampleLines = sourceLines[sourceStartLine..sourceEndLine]
            .Select(l => l.Length > 0 ? l.Remove(0, whitespaceToRemove) : l).ToArray();
        return sourceExampleLines;
    }

    static string SourceFile([CallerFilePath] string sourceFilePath = "") => sourceFilePath;
}
