using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Text;
using static CudaSharp.nvJitLink;
using static CudaSharp.nvrtc;

namespace CudaSharp.Test;

[TestClass]
public class nvJitLinkTest
{
    const string KernelSource = """
        extern "C" __global__ void increment(int* value)
        {
            *value += 1;
        }
        """;

    public nvJitLinkTest()
    {
        try
        {
            nvJitLinkVersion(out _, out _).Ok();
        }
        catch (Exception ex)
        {
            Assert.Inconclusive($"nvJitLink initialization failed: {ex.Message}");
        }
    }

    [TestMethod]
    public void nvJitLinkTest_nvJitLinkResult_ToStringFast()
    {
        Assert.EnumValuesToString<nvJitLinkResult>(result => result.ToStringFast());
        var unknown = (nvJitLinkResult)(int.MaxValue - 1);
        Assert.AreEqual("NVJITLINK_ERROR_UNKNOWN:2147483646", unknown.ToStringFast());
    }

    [TestMethod]
    public void nvJitLinkTest_nvJitLinkResult_Ok()
    {
        Assert.EnumValuesOkThrows<nvJitLinkResult>(
            result => result == nvJitLinkResult.NVJITLINK_SUCCESS,
            result => result.Ok());
    }

    [TestMethod]
    public void nvJitLinkTest_nvJitLinkResult_IsOkAndIsError()
    {
        Assert.IsTrue(nvJitLinkResult.NVJITLINK_SUCCESS.IsOk());
        Assert.IsFalse(nvJitLinkResult.NVJITLINK_SUCCESS.IsError());
        Assert.IsFalse(nvJitLinkResult.NVJITLINK_ERROR_INVALID_INPUT.IsOk());
        Assert.IsTrue(nvJitLinkResult.NVJITLINK_ERROR_INVALID_INPUT.IsError());
    }

    [TestMethod]
    public void nvJitLinkTest_Version()
    {
        nvJitLinkVersion(out var major, out var minor).Ok();

        Assert.IsGreaterThan((uint)0, major);
        Assert.IsGreaterThanOrEqualTo((uint)0, minor);
    }

    [TestMethod]
    public void nvJitLinkTest_InvalidPtxProducesDiagnostic()
    {
        nvJitLinkCreate(out var handle, [$"-arch=sm_{GetHighestArchitecture()}"]).Ok();
        try
        {
            const string invalidPtx = """
                .version 8.0
                .target sm_80
                .address_size 64
                .visible .entry broken()
                {
                    invalid_instruction;
                    ret;
                }
                """;
            var result = nvJitLinkAddData(
                handle,
                nvJitLinkInputType.NVJITLINK_INPUT_PTX,
                Encoding.UTF8.GetBytes(invalidPtx),
                "invalid.ptx");

            Assert.AreEqual(nvJitLinkResult.NVJITLINK_ERROR_PTX_COMPILE, result);
            var errorLog = nvJitLinkGetErrorLogString(handle);
            Assert.IsFalse(string.IsNullOrWhiteSpace(errorLog));
            StringAssert.Contains(errorLog, "error");
        }
        finally
        {
            nvJitLinkDestroy(ref handle).Ok();
        }
    }

    [TestMethod]
    public void nvJitLinkTest_PtxLinksToCubin()
    {
        var ptx = CompilePtx();
        nvJitLinkCreate(out var handle, [$"-arch=sm_{GetHighestArchitecture()}"]).Ok();
        try
        {
            nvJitLinkAddData(handle, nvJitLinkInputType.NVJITLINK_INPUT_PTX, ptx, "increment.ptx").Ok();
            Complete(handle);

            var cubin = nvJitLinkGetLinkedCubin(handle);

            Assert.IsNotEmpty(cubin);
        }
        finally
        {
            nvJitLinkDestroy(ref handle).Ok();
        }
    }

    [TestMethod]
    public void nvJitLinkTest_LtoIrLinksToLtoIrAndCubin()
    {
        var ltoir = CompileLtoIr();
        nvJitLinkCreate(out var handle, ["-lto", $"-arch=sm_{GetHighestArchitecture()}"]).Ok();
        try
        {
            nvJitLinkAddData(handle, nvJitLinkInputType.NVJITLINK_INPUT_LTOIR, ltoir, "increment.ltoir").Ok();
            Complete(handle);

            var linkedLtoIr = nvJitLinkGetLinkedLTOIR(handle);
            var cubin = nvJitLinkGetLinkedCubin(handle);

            Assert.IsNotEmpty(linkedLtoIr);
            Assert.IsNotEmpty(cubin);
        }
        finally
        {
            nvJitLinkDestroy(ref handle).Ok();
        }
    }

    [TestMethod]
    public void nvJitLinkTest_LtoIrLinksToPtx()
    {
        var ltoir = CompileLtoIr();
        nvJitLinkCreate(out var handle, ["-lto", "-ptx", $"-arch=sm_{GetHighestArchitecture()}"]).Ok();
        try
        {
            nvJitLinkAddData(handle, nvJitLinkInputType.NVJITLINK_INPUT_LTOIR, ltoir, "increment.ltoir").Ok();
            Complete(handle);

            var linkedPtx = nvJitLinkGetLinkedPtxString(handle);

            Assert.IsFalse(string.IsNullOrWhiteSpace(linkedPtx));
            StringAssert.Contains(linkedPtx, ".version");
        }
        finally
        {
            nvJitLinkDestroy(ref handle).Ok();
        }
    }

    static byte[] CompilePtx()
    {
        nvrtcCreateProgram(out var program, KernelSource, "increment.cu", 0, [], []).Ok();
        try
        {
            Compile(program, $"--gpu-architecture=compute_{GetHighestArchitecture()}");
            return nvrtcGetPTX(program);
        }
        finally
        {
            nvrtcDestroyProgram(ref program).Ok();
        }
    }

    static byte[] CompileLtoIr()
    {
        nvrtcCreateProgram(out var program, KernelSource, "increment.cu", 0, [], []).Ok();
        try
        {
            Compile(program,
                $"--gpu-architecture=compute_{GetHighestArchitecture()}",
                "--relocatable-device-code=true",
                "-dlto");
            return nvrtcGetLTOIR(program);
        }
        finally
        {
            nvrtcDestroyProgram(ref program).Ok();
        }
    }

    static int GetHighestArchitecture() => nvrtcGetSupportedArchs()[^1];

    static void Compile(nvrtcProgram program, params string[] options)
    {
        var result = nvrtcCompileProgram(program, options.Length, options);
        if (result != nvrtcResult.NVRTC_SUCCESS)
        {
            Assert.Fail($"NVRTC compilation failed with {result}:\n{nvrtcGetProgramLogString(program)}");
        }
    }

    static void Complete(nvJitLinkHandle handle)
    {
        var result = nvJitLinkComplete(handle);
        if (result != nvJitLinkResult.NVJITLINK_SUCCESS)
        {
            Assert.Fail($"nvJitLink failed with {result}:\n{nvJitLinkGetErrorLogString(handle)}");
        }
    }
}
