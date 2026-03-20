using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using static CudaSharp.nvrtc;

namespace CudaSharp.Test;

[TestClass]
public class nvrtcTest
{
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
}
