using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using PublicApiGenerator;
using static CudaSharp.nvcuda;
using static CudaSharp.nvrtc;
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

    public ReadMeTest()
    {
        try
        {
            cuInit().Ok();
        }
        catch (Exception ex)
        {
#pragma warning disable MSTEST0058 // Do not use asserts in catch blocks
            Assert.Inconclusive($"CUDA initialization failed: {ex.Message}");
#pragma warning restore MSTEST0058 // Do not use asserts in catch blocks
        }
    }

    [TestMethod]
    public unsafe void ReadMeTest_()
    {
        cuInit().Ok();
        cuDeviceGet(out var device, 0).Ok();
        cuCtxCreate(out var context, CUctx_flags.CU_CTX_SCHED_AUTO, device).Ok();

        var kernelSource =
            """
            extern "C" __global__ void saxpy(float a, float *x, float *y, float *out, size_t n)
            {
                size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
                if (tid < n) {
                    out[tid] = a * x[tid] + y[tid];
                }
            }
            """;
        nvrtcCreateProgram(out var prog, kernelSource, "saxpy.cu", 0, [], []).Ok();

        var compileResult = nvrtcCompileProgram(prog, 0, []);
        if (compileResult != nvrtcResult.NVRTC_SUCCESS)
        {
            nvrtcGetProgramLogSize(prog, out var logSize).Ok();
            var logBuffer = new byte[logSize];
            nvrtcGetProgramLog(prog, logBuffer).Ok();
            var log = Encoding.UTF8.GetString(logBuffer).TrimEnd('\0');
            Assert.Fail($"Compilation failed:\n{log}");
        }

        nvrtcGetPTXSize(prog, out var ptxSize).Ok();
        var ptxBuffer = new byte[ptxSize];
        nvrtcGetPTX(prog, ptxBuffer).Ok();

        nvrtcDestroyProgram(ref prog).Ok();

        cuModuleLoadData(out var module, ptxBuffer).Ok();
        cuModuleGetFunction(out var function, module, "saxpy").Ok();

        var n = 1024;
        var a = 2.5f;
        var bytes = (nuint)(n * sizeof(float));

        // Allocate Device Memory
        cuMemAlloc(out var d_x, bytes).Ok();
        cuMemAlloc(out var d_y, bytes).Ok();
        cuMemAlloc(out var d_out, bytes).Ok();

        // Allocate Host Memory (Pinned)
        cuMemHostAlloc(out var h_x_ptr, bytes, 0).Ok();
        cuMemHostAlloc(out var h_y_ptr, bytes, 0).Ok();
        cuMemHostAlloc(out var h_out_ptr, bytes, 0).Ok();

        // Initialize Host Data
        var h_x = new Span<float>((void*)h_x_ptr, n);
        var h_y = new Span<float>((void*)h_y_ptr, n);
        var h_out = new Span<float>((void*)h_out_ptr, n);

        for (var i = 0; i < n; i++)
        {
            h_x[i] = i;
            h_y[i] = i * 2;
        }

        cuMemcpyHtoD(d_x, h_x_ptr, bytes).Ok();
        cuMemcpyHtoD(d_y, h_y_ptr, bytes).Ok();

        // Kernel params
        void*[] args = [&a, &d_x, &d_y, &d_out, &n];
        var argsPtrs = new IntPtr[args.Length];
        for (var i = 0; i < args.Length; i++)
        {
            argsPtrs[i] = (IntPtr)args[i];
        }

        cuLaunchKernel(
            function,
            (uint)((n + 255) / 256), 1, 1, // Grid
            256, 1, 1, // Block
            0, new CUstream(IntPtr.Zero),
            new ReadOnlySpan<IntPtr>(argsPtrs),
            []
        ).Ok();

        cuCtxSynchronize().Ok();

        cuMemcpyDtoH(h_out_ptr, d_out, bytes).Ok();

        for (var i = 0; i < n; i++)
        {
            var actual = h_out[i];
            var expected = a * h_x[i] + h_y[i];
            Assert.AreEqual(expected, actual, 1e-5);
        }

        // Cleanup
        cuMemFreeHost(h_x_ptr);
        cuMemFreeHost(h_y_ptr);
        cuMemFreeHost(h_out_ptr);

        cuMemFree(d_x);
        cuMemFree(d_y);
        cuMemFree(d_out);
        cuCtxDestroy(context);

        // Above example code is for demonstration purposes only.
        // Short names and repeated constants are only for demonstration.
    }

    [TestMethod]
    public void ReadMeTest_cuCtx()
    {
        cuInit().Ok();
        cuDeviceGet(out var device, 0).Ok();
        cuCtxCreate(out var context, CUctx_flags.CU_CTX_SCHED_AUTO, device).Ok();
        cuCtxDestroy(context);
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
            (nameof(ReadMeTest_cuCtx) + "()", "### Example - cuCtx"),
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
            .Select(l => l.Length >= whitespaceToRemove ? l.Remove(0, whitespaceToRemove) : l).ToArray();
        return sourceExampleLines;
    }

    static string SourceFile([CallerFilePath] string sourceFilePath = "") => sourceFilePath;
}
