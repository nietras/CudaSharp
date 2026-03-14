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
}
