using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using static CudaSharp.nvrtc;

namespace CudaSharp.Test;

[TestClass]
public class nvrtcTest
{
    const string KernelSource = """
        template<typename T>
        __global__ void increment(T* value)
        {
            *value += T{1};
        }
        """;

    public nvrtcTest()
    {
        try
        {
            nvrtcGetErrorString(nvrtcResult.NVRTC_SUCCESS);
        }
        catch (Exception ex)
        {
            Assert.Inconclusive($"CUDA initialization failed: {ex.Message}");
        }
    }

    [TestMethod]
    public void nvrtcTest_CUDA13_3BundledHeadersCanBeInstalledAndIncluded()
    {
        RequireNvrtcVersion(13, 3, "Bundled-header installation");

        var result = nvrtcGetBundledHeadersInfo(out var info, out var errorLog);
        AssertNvrtcSuccess(result, errorLog, "query bundled-header information");
        if (info.available == 0)
        {
            Assert.Inconclusive("This NVRTC installation does not provide bundled CUDA headers.");
        }

        Assert.IsGreaterThan((uint)0, info.numFiles);
        Assert.IsGreaterThan((nuint)0, info.compressedSize);
        Assert.IsGreaterThan((nuint)0, info.uncompressedSize);
        Assert.IsGreaterThanOrEqualTo(13, info.cudaVersionMajor);

        var installPath = Path.Combine(Path.GetTempPath(), $"CudaSharp-nvrtc-{Guid.NewGuid():N}");
        try
        {
            result = nvrtcInstallBundledHeaders(installPath, nvrtcInstallHeadersFlags.NVRTC_INSTALL_HEADERS_SKIP_IF_EXISTS, out errorLog);
            AssertNvrtcSuccess(result, errorLog, "install bundled headers");
            Assert.IsTrue(Directory.Exists(installPath));
            Assert.IsNotEmpty(Directory.EnumerateFiles(installPath, "*", SearchOption.AllDirectories));
            Assert.IsTrue(File.Exists(Path.Combine(installPath, "cuda_fp16.h")));
            Assert.IsTrue(Directory.Exists(Path.Combine(installPath, "cccl")));

            const string source = """
                #include <cuda_fp16.h>
                #include <cuda/std/type_traits>
                static_assert(cuda::std::is_same_v<int, int>);
                extern "C" __global__ void bundled_header_kernel(__half* value)
                {
                    *value = __float2half(1.0f);
                }
                """;

            nvrtcCreateProgram(out var program, source, "bundled_header_kernel.cu", 0, [], []).Ok();
            try
            {
                Compile(program,
                    $"--gpu-architecture=compute_{GetHighestArchitecture()}",
                    $"-I{installPath}",
                    $"-I{Path.Combine(installPath, "cccl")}");
                Assert.IsNotEmpty(nvrtcGetPTX(program));
            }
            finally
            {
                nvrtcDestroyProgram(ref program).Ok();
            }
        }
        finally
        {
            if (Directory.Exists(installPath))
            {
                result = nvrtcRemoveBundledHeaders(installPath, out errorLog);
                AssertNvrtcSuccess(result, errorLog, "remove bundled headers");
                Assert.IsFalse(Directory.Exists(installPath));
            }
        }
    }

    [TestMethod]
    public void nvrtcTest_nvrtcGetErrorString()
    {
        Assert.AreNotEqual(IntPtr.Zero, nvrtcGetErrorString(nvrtcResult.NVRTC_SUCCESS));
    }

    [TestMethod]
    public void nvrtcTest_nvrtcResult_ToStringFast()
    {
        Assert.EnumValuesToString<nvrtcResult>(r => r.ToStringFast());
        var unknown = (nvrtcResult)(int.MaxValue - 1);
        Assert.AreEqual("NVRTC_ERROR_UNKNOWN:2147483646", unknown.ToStringFast());
    }
    [TestMethod]
    public void nvrtcTest_nvrtcResult_nvrtcGetErrorStringString()
    {
        Assert.EnumValuesToString<nvrtcResult>(nvrtcGetErrorStringString);
        var unknown = (nvrtcResult)(int.MaxValue - 1);
        Assert.AreEqual("NVRTC_ERROR unknown", nvrtcGetErrorStringString(unknown));
    }
    [TestMethod]
    public void nvrtcTest_nvrtcResult_Ok()
    {
        Assert.EnumValuesOkThrows<nvrtcResult>(r => r == nvrtcResult.NVRTC_SUCCESS, r => r.Ok());
    }
    [TestMethod]
    public void nvrtcTest_nvrtcResult_IsOk()
    {
        Assert.IsTrue(nvrtcResult.NVRTC_SUCCESS.IsOk());
        Assert.IsFalse(nvrtcResult.NVRTC_ERROR_COMPILATION.IsOk());
        Assert.IsFalse(nvrtcResult.NVRTC_ERROR_INVALID_PROGRAM.IsOk());
    }
    [TestMethod]
    public void nvrtcTest_nvrtcResult_IsError()
    {
        Assert.IsFalse(nvrtcResult.NVRTC_SUCCESS.IsError());
        Assert.IsTrue(nvrtcResult.NVRTC_ERROR_COMPILATION.IsError());
        Assert.IsTrue(nvrtcResult.NVRTC_ERROR_INVALID_PROGRAM.IsError());
    }

    [TestMethod]
    public void nvrtcTest_VersionAndSupportedArchitectures()
    {
        nvrtcVersion(out var major, out var minor).Ok();
        var architectures = nvrtcGetSupportedArchs();

        Assert.IsGreaterThan(0, major);
        Assert.IsGreaterThanOrEqualTo(0, minor);
        Assert.IsNotEmpty(architectures);
        Assert.IsTrue(architectures.SequenceEqual(architectures.Order()));
        Assert.IsTrue(architectures.All(architecture => architecture > 0));
    }

    [TestMethod]
    public void nvrtcTest_CompilePTXAndLoweredName()
    {
        nvrtcCreateProgram(out var program, KernelSource, "increment.cu", 0, [], []).Ok();
        try
        {
            const string nameExpression = "&increment<int>";
            nvrtcAddNameExpression(program, nameExpression).Ok();
            Compile(program, $"--gpu-architecture=compute_{GetHighestArchitecture()}");

            var ptx = nvrtcGetPTX(program);
            var loweredName = nvrtcGetLoweredNameString(program, nameExpression);

            Assert.IsNotEmpty(ptx);
            Assert.IsTrue(Encoding.UTF8.GetString(ptx).Contains(".version"));
            Assert.IsFalse(string.IsNullOrWhiteSpace(loweredName));
        }
        finally
        {
            nvrtcDestroyProgram(ref program).Ok();
        }
    }

    [TestMethod]
    public void nvrtcTest_CompilationDiagnostics()
    {
        nvrtcCreateProgram(out var program, "__global__ void broken( {", "broken.cu", 0, [], []).Ok();
        try
        {
            var result = nvrtcCompileProgram(program, 0, []);
            Assert.AreEqual(nvrtcResult.NVRTC_ERROR_COMPILATION, result);

            var log = nvrtcGetProgramLogString(program);
            Assert.IsFalse(string.IsNullOrWhiteSpace(log));
            StringAssert.Contains(log, "error");
        }
        finally
        {
            nvrtcDestroyProgram(ref program).Ok();
        }
    }

    [TestMethod]
    public void nvrtcTest_CompileCUBIN()
    {
        nvrtcCreateProgram(out var program, KernelSource, "increment.cu", 0, [], []).Ok();
        try
        {
            Compile(program, $"--gpu-architecture=sm_{GetHighestArchitecture()}");
            Assert.IsNotEmpty(nvrtcGetCUBIN(program));
        }
        finally
        {
            nvrtcDestroyProgram(ref program).Ok();
        }
    }

    [TestMethod]
    public void nvrtcTest_CompileLTOIR()
    {
        nvrtcCreateProgram(out var program, KernelSource, "increment.cu", 0, [], []).Ok();
        try
        {
            Compile(program,
                $"--gpu-architecture=compute_{GetHighestArchitecture()}",
                "--relocatable-device-code=true",
                "-dlto");
            Assert.IsNotEmpty(nvrtcGetLTOIR(program));
        }
        finally
        {
            nvrtcDestroyProgram(ref program).Ok();
        }
    }

    [TestMethod]
    public void nvrtcTest_CUDA13_3TileCppCompilesToTileIR()
    {
        RequireNvrtcVersion(13, 3, "CUDA Tile C++");

        const string tileSource = """
            extern "C" __tile_global__ void tile_increment(int* value)
            {
                *value += 1;
            }
            """;

        nvrtcCreateProgram(out var program, tileSource, "tile_increment.cu", 0, [], []).Ok();
        try
        {
            Compile(program,
                $"--gpu-architecture=compute_{GetHighestArchitecture()}",
                "--std=c++20",
                "-enable-tile");

            Assert.IsNotEmpty(nvrtcGetTileIR(program));
        }
        finally
        {
            nvrtcDestroyProgram(ref program).Ok();
        }
    }

    [TestMethod]
    public void nvrtcTest_PCHHeapSize()
    {
        RequireNvrtcVersion(13, 0, "PCH APIs");

        nvrtcGetPCHHeapSize(out var size).Ok();
        Assert.IsGreaterThan((nuint)0, size);
        nvrtcSetPCHHeapSize(size).Ok();
    }

    static int GetHighestArchitecture() => nvrtcGetSupportedArchs()[^1];

    static void RequireNvrtcVersion(int requiredMajor, int requiredMinor, string feature)
    {
        nvrtcVersion(out var actualMajor, out var actualMinor).Ok();
        if (actualMajor < requiredMajor || actualMajor == requiredMajor && actualMinor < requiredMinor)
        {
            Assert.Inconclusive($"{feature} requires NVRTC {requiredMajor}.{requiredMinor} or later; found {actualMajor}.{actualMinor}.");
        }
    }

    static void Compile(nvrtcProgram program, params string[] options)
    {
        var result = nvrtcCompileProgram(program, options.Length, options);
        if (result != nvrtcResult.NVRTC_SUCCESS)
        {
            Assert.Fail($"NVRTC compilation failed with {result}:\n{nvrtcGetProgramLogString(program)}");
        }
    }

    static void AssertNvrtcSuccess(nvrtcResult result, IntPtr errorLog, string operation)
    {
        if (result == nvrtcResult.NVRTC_SUCCESS)
        {
            return;
        }

        var diagnostic = errorLog == IntPtr.Zero ? null : Marshal.PtrToStringUTF8(errorLog);
        Assert.Fail($"Failed to {operation} with {result}: {diagnostic}");
    }
}
