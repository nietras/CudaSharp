using System;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
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
        var successIsOk = nvJitLinkResult.NVJITLINK_SUCCESS.IsOk();
        Assert.IsTrue(successIsOk);
        var successIsError = nvJitLinkResult.NVJITLINK_SUCCESS.IsError();
        Assert.IsFalse(successIsError);
        var invalidInputIsOk = nvJitLinkResult.NVJITLINK_ERROR_INVALID_INPUT.IsOk();
        Assert.IsFalse(invalidInputIsOk);
        var invalidInputIsError = nvJitLinkResult.NVJITLINK_ERROR_INVALID_INPUT.IsError();
        Assert.IsTrue(invalidInputIsError);
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
        var highestArchitecture = GetHighestArchitecture();
        var architectureOption = $"-arch=sm_{highestArchitecture}";
        nvJitLinkCreate(out var handle, [architectureOption]).Ok();
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
            var invalidPtxBytes = Encoding.UTF8.GetBytes(invalidPtx);
            var result = nvJitLinkAddData(
                handle,
                nvJitLinkInputType.NVJITLINK_INPUT_PTX,
                invalidPtxBytes,
                "invalid.ptx");

            Assert.AreEqual(nvJitLinkResult.NVJITLINK_ERROR_PTX_COMPILE, result);
            var errorLog = nvJitLinkGetErrorLogString(handle);
            var errorLogIsEmpty = string.IsNullOrWhiteSpace(errorLog);
            Assert.IsFalse(errorLogIsEmpty);
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
        var highestArchitecture = GetHighestArchitecture();
        var architectureOption = $"-arch=sm_{highestArchitecture}";
        nvJitLinkCreate(out var handle, [architectureOption]).Ok();
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
        var highestArchitecture = GetHighestArchitecture();
        var architectureOption = $"-arch=sm_{highestArchitecture}";
        nvJitLinkCreate(out var handle, ["-lto", architectureOption]).Ok();
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
        var highestArchitecture = GetHighestArchitecture();
        var architectureOption = $"-arch=sm_{highestArchitecture}";
        nvJitLinkCreate(out var handle, ["-lto", "-ptx", architectureOption]).Ok();
        try
        {
            nvJitLinkAddData(handle, nvJitLinkInputType.NVJITLINK_INPUT_LTOIR, ltoir, "increment.ltoir").Ok();
            Complete(handle);

            var linkedPtx = nvJitLinkGetLinkedPtxString(handle);

            var linkedPtxIsEmpty = string.IsNullOrWhiteSpace(linkedPtx);
            Assert.IsFalse(linkedPtxIsEmpty);
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
            var highestArchitecture = GetHighestArchitecture();
            var architectureOption = $"--gpu-architecture=compute_{highestArchitecture}";
            Compile(program, architectureOption);
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
            var highestArchitecture = GetHighestArchitecture();
            var architectureOption = $"--gpu-architecture=compute_{highestArchitecture}";
            Compile(program, architectureOption, "--relocatable-device-code=true", "-dlto");
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
            var log = nvrtcGetProgramLogString(program);
            Assert.Fail($"NVRTC compilation failed with {result}:\n{log}");
        }
    }

    static void Complete(nvJitLinkHandle handle)
    {
        var result = nvJitLinkComplete(handle);
        if (result != nvJitLinkResult.NVJITLINK_SUCCESS)
        {
            var errorLog = nvJitLinkGetErrorLogString(handle);
            Assert.Fail($"nvJitLink failed with {result}:\n{errorLog}");
        }
    }
}
