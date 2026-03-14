using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CudaSharp.Test;

[TestClass]
public class nvcudaTest
{
    public nvcudaTest()
    {
        try
        {
            nvcuda.cuInit(0);
        }
        catch (Exception ex)
        {
            Assert.Inconclusive($"CUDA initialization failed: {ex.Message}");
        }
    }

    [TestMethod]
    public void nvcudaTest_cuInit()
    {
        nvcuda.cuInit(0);
    }
}
