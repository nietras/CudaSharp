using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using static CudaSharp.nvcuda;

namespace CudaSharp.Test;

[TestClass]
public class nvcudaTest
{
    public nvcudaTest()
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
    public void nvcudaTest_cuInit()
    {
        cuInit();
    }

    [TestMethod]
    public void nvcudaTest_CUresult_ToStringFast()
    {
        Assert.EnumValuesToString<CUresult>(r => r.ToStringFast());
        var unknown = (CUresult)(int.MaxValue - 1);
        Assert.AreEqual("CUDA_ERROR_UNKNOWN:2147483646", unknown.ToStringFast());
    }
    [TestMethod]
    public void nvcudaTest_CUresult_Ok()
    {
        Assert.EnumValuesOkThrows<CUresult>(r => r == CUresult.CUDA_SUCCESS, r => r.Ok());
    }
    [TestMethod]
    public void nvcudaTest_CUresult_IsOk()
    {
        Assert.IsTrue(CUresult.CUDA_SUCCESS.IsOk());
        Assert.IsFalse(CUresult.CUDA_ERROR_DEINITIALIZED.IsOk());
        Assert.IsFalse(CUresult.CUDA_ERROR_NOT_READY.IsOk());
    }
    [TestMethod]
    public void nvcudaTest_CUresult_IsError()
    {
        Assert.IsFalse(CUresult.CUDA_SUCCESS.IsError());
        Assert.IsTrue(CUresult.CUDA_ERROR_DEINITIALIZED.IsError());
        Assert.IsTrue(CUresult.CUDA_ERROR_NOT_READY.IsError());
    }
}
