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
            cuInit(0);
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
        cuInit(0);
    }

    [TestMethod]
    public void nvcudaTest_CUresult()
    {
        AssertEnumToString<CUresult>(r => r.ToStringFast());
        var unknown = ((CUresult)int.MaxValue - 1);
        Assert.AreEqual("CUDA_ERROR_UNKNOWN", unknown.ToStringFast());
        Assert.AreEqual("2147483646", unknown.ToString());
    }

    public static void AssertEnumToString<TEnum>(Func<TEnum, string> toString)
        where TEnum : unmanaged, Enum
    {
        var values = Enum.GetValues<TEnum>();
        foreach (var value in values)
        {
            Assert.AreEqual(value.ToString(), toString(value));
        }
    }
}
