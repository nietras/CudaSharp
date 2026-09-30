using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using static CudaSharp.nvcuda;
using static CudaSharp.nvrtc;

namespace CudaSharp.TileGym.Test;

/// <summary>Tests batch NVRTC compilation and context-specific shared module ownership.</summary>
/// <seealso href="https://docs.nvidia.com/cuda/nvrtc/index.html#group__compilation" />
[TestClass]
public class TileCppBatchTest
{
    const string FirstExpression = "&relu_activation_fwd_kernel<float, 64, 0>";
    const string SecondExpression = "&relu_activation_fwd_kernel<float, 128, 0>";
    const string Source = """
        using int32_t = int;
        #include "relu.cuh"
        template __tile_global__ void relu_activation_fwd_kernel<float, 64, 0>(
            const float*, float*, int, float, float, float, bool);
        template __tile_global__ void relu_activation_fwd_kernel<float, 128, 0>(
            const float*, float*, int, float, float, float, bool);
        """;

    /// <summary>Validates batch inputs without loading NVRTC or the CUDA driver.</summary>
    /// <seealso href="https://docs.nvidia.com/cuda/nvrtc/index.html#group__compilation" />
    [TestMethod]
    public void TileCppBatchTest_ValidatesArgumentsWithoutNvrtc()
    {
        var compiler = new TileCppCompiler(120, installBundledHeaders: false);
        var config = new TileCppConfig([]);
        Assert.ThrowsExactly<ArgumentException>(() => compiler.CompileKernels("", "batch.cu", [], config));
        Assert.ThrowsExactly<ArgumentNullException>(() =>
            compiler.CompileKernels(null!, "batch.cu", [FirstExpression], config));
        Assert.ThrowsExactly<ArgumentException>(() => compiler.CompileKernels("", " ", [FirstExpression], config));
        Assert.ThrowsExactly<ArgumentNullException>(() =>
            compiler.CompileKernels("", "batch.cu", [FirstExpression], null!));
        Assert.ThrowsExactly<ArgumentNullException>(() => compiler.CompileKernels("", "batch.cu", [null!], config));
        Assert.ThrowsExactly<ArgumentException>(() => compiler.CompileKernels("", "batch.cu", [""], config));
        Assert.ThrowsExactly<ArgumentException>(() => compiler.CompileKernels("", "batch.cu", [" \t"], config));
        Assert.ThrowsExactly<ArgumentException>(() =>
            compiler.CompileKernels("", "batch.cu", [FirstExpression, "&kernel\0suffix"], config));
    }

    /// <summary>Checks CPU-only construction, input validation, and idempotent disposal.</summary>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html" />
    [TestMethod]
    public void TileCppBatchTest_ModuleConstructionAndDisposalWithoutDriver()
    {
        Assert.ThrowsExactly<ArgumentNullException>(() => new TileCppModule(null!));
        var emptyCompilation = new TileCppCompilationBatch([], new Dictionary<string, string>());
        Assert.ThrowsExactly<ArgumentException>(() => new TileCppModule(emptyCompilation));
        var compilation = new TileCppCompilationBatch([1], new Dictionary<string, string>
        {
            [FirstExpression] = "lowered_kernel",
        });
        var module = new TileCppModule(compilation);
        module.Dispose();
        module.Dispose();
        Assert.ThrowsExactly<ObjectDisposedException>(() => module.LoadFunctions());
    }

    /// <summary>Compiles two ReLU specializations once, deduplicates expressions, and propagates compiler errors.</summary>
    /// <seealso href="https://docs.nvidia.com/cuda/nvrtc/index.html#group__compilation" />
    [TestMethod]
    [DoNotParallelize]
    public void TileCppBatchTest_CUDA13_3CompilesDistinctSpecializationsOnce()
    {
        try
        {
            RequireNvrtc();
            var architectures = nvrtcGetSupportedArchs();
            var compiler = new TileCppCompiler(architectures[^1]);
            compiler.PrepareBundledHeaders();
            using var writer = new StringWriter(CultureInfo.InvariantCulture);
            using var listener = new TextWriterTraceListener(writer);
            Trace.Listeners.Add(listener);
            try
            {
                var compilation = CompileRelu(compiler);
                Assert.IsNotEmpty(compilation.TileIr);
                Assert.HasCount(2, compilation.EntryPoints);
                var first = compilation.EntryPoints[FirstExpression];
                var second = compilation.EntryPoints[SecondExpression];
                Assert.IsFalse(string.IsNullOrWhiteSpace(first));
                Assert.IsFalse(string.IsNullOrWhiteSpace(second));
                Assert.AreNotEqual(first, second);
                var trace = writer.ToString();
                var compileStages = Regex.Matches(trace, @"\.nvrtcCompileProgram\[program=0x");
                var registrations = Regex.Matches(trace, @"\.nvrtcAddNameExpression:");
                var outputs = Regex.Matches(trace, @"\.nvrtcGetTileIR:");
                var lowerings = Regex.Matches(trace, @"\.nvrtcGetLoweredNameString:");
                Assert.HasCount(1, compileStages);
                Assert.HasCount(2, registrations);
                Assert.HasCount(1, outputs);
                Assert.HasCount(2, lowerings);
                var config = new TileCppConfig([]);
                Assert.ThrowsExactly<CudaException<nvrtcResult>>(() => compiler.CompileKernels(
                    "template<int Variant> __tile_global__ void broken() { invalid CUDA source; }",
                    "batch_failure.cu", ["&broken<1>", "&broken<2>"], config));
            }
            finally
            {
                Trace.Listeners.Remove(listener);
            }
        }
        catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException)
        {
            Assert.Inconclusive($"CUDA Tile C++ NVRTC components are unavailable: {ex.Message}");
        }
    }

    /// <summary>Loads exact names, snapshots caller data, caches per context, and restores context on disposal.</summary>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html" />
    [TestMethod]
    public void TileCppBatchTest_CUDA13_3LoadsAndCachesFunctionsPerContext()
    {
        try
        {
            RequireNvrtc();
            CuInit.EnsureInit();
            cuDeviceGet(out var device, 0).Ok();
            cuCtxGetCurrent(out var previousContext).Ok();
            cuDevicePrimaryCtxRetain(out var context, device).Ok();
            try
            {
                cuCtxSetCurrent(context).Ok();
                var architecture = device.GetArchitecture();
                var compiler = new TileCppCompiler(architecture);
                var compilation = CompileRelu(compiler);
                var callerEntryPoints = new Dictionary<string, string>(compilation.EntryPoints, StringComparer.Ordinal);
                var callerTileIr = (byte[])compilation.TileIr.Clone();
                var callerBatch = new TileCppCompilationBatch(callerTileIr, callerEntryPoints);
                using var module = new TileCppModule(callerBatch);
                callerEntryPoints.Clear();
                Array.Clear(callerTileIr);
                var functions = module.LoadFunctions();
                Assert.HasCount(2, functions);
                Assert.AreNotEqual(functions[FirstExpression], functions[SecondExpression]);
                var cachedFunctions = module.LoadFunctions();
                Assert.AreSame(functions, cachedFunctions);
                foreach (var entry in compilation.EntryPoints)
                {
                    var function = functions[entry.Key];
                    Assert.AreNotEqual(default, function);
                    cuFuncGetName(out var namePointer, function).Ok();
                    var actualName = Marshal.PtrToStringUTF8(namePointer);
                    Assert.AreEqual(entry.Value, actualName);
                }

                cuCtxSetCurrent(default).Ok();
                Assert.ThrowsExactly<InvalidOperationException>(() => module.LoadFunctions());
                cuCtxCreate_v2(out var otherContext, default, device).Ok();
                try
                {
                    var otherFunctions = module.LoadFunctions();
                    Assert.HasCount(2, otherFunctions);
                    Assert.AreNotSame(functions, otherFunctions);
                    var cachedOtherFunctions = module.LoadFunctions();
                    Assert.AreSame(otherFunctions, cachedOtherFunctions);
                    cuCtxSetCurrent(default).Ok();
                    module.Dispose();
                    cuCtxGetCurrent(out var restoredContext).Ok();
                    Assert.AreEqual(default, restoredContext);
                    module.Dispose();
                    Assert.ThrowsExactly<ObjectDisposedException>(() => module.LoadFunctions());
                }
                finally
                {
                    module.Dispose();
                    cuCtxDestroy(otherContext).Ok();
                }
            }
            finally
            {
                cuCtxSetCurrent(previousContext).Ok();
                cuDevicePrimaryCtxRelease(device).Ok();
            }
        }
        catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException)
        {
            Assert.Inconclusive($"CUDA Tile C++ driver components are unavailable: {ex.Message}");
        }
    }

    /// <summary>Rejects incorrect lowered names instead of falling back to substring matching.</summary>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html" />
    [TestMethod]
    public void TileCppBatchTest_CUDA13_3RejectsMissingExactEntryPoint()
    {
        try
        {
            RequireNvrtc();
            CuInit.EnsureInit();
            cuDeviceGet(out var device, 0).Ok();
            cuCtxGetCurrent(out var previousContext).Ok();
            cuDevicePrimaryCtxRetain(out var context, device).Ok();
            try
            {
                cuCtxSetCurrent(context).Ok();
                var architecture = device.GetArchitecture();
                var compiler = new TileCppCompiler(architecture);
                var compilation = CompileRelu(compiler);
                var wrongEntryPoints = new Dictionary<string, string>(compilation.EntryPoints, StringComparer.Ordinal)
                {
                    [SecondExpression] = "relu_activation_fwd_kernel",
                };
                var wrongBatch = new TileCppCompilationBatch(compilation.TileIr, wrongEntryPoints);
                using var module = new TileCppModule(wrongBatch);
                Assert.ThrowsExactly<CudaException<CUresult>>(() => module.LoadFunctions());
                Assert.ThrowsExactly<CudaException<CUresult>>(() => module.LoadFunctions());
                module.Dispose();
                module.Dispose();
            }
            finally
            {
                cuCtxSetCurrent(previousContext).Ok();
                cuDevicePrimaryCtxRelease(device).Ok();
            }
        }
        catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException)
        {
            Assert.Inconclusive($"CUDA Tile C++ driver components are unavailable: {ex.Message}");
        }
    }

    static void RequireNvrtc()
    {
        nvrtcVersion(out var major, out var minor).Ok();
        if (major < 13 || major == 13 && minor < 3)
        {
            Assert.Inconclusive($"CUDA Tile C++ requires NVRTC 13.3 or later; found {major}.{minor}.");
        }
    }

    static TileCppCompilationBatch CompileRelu(TileCppCompiler compiler)
    {
        var headerPath = Path.Combine(AppContext.BaseDirectory, "src-tilecpp", "tilegym", "activation", "relu.cuh");
        var headerSource = File.ReadAllText(headerPath);
        var header = new TileCppHeader("relu.cuh", headerSource);
        var config = new TileCppConfig([]);
        return compiler.CompileKernels(Source, "relu_batch.cu",
            [FirstExpression, SecondExpression, FirstExpression], config, [header]);
    }
}
