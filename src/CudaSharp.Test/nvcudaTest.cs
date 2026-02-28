using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CudaSharp.Test;

[TestClass]
public class nvcudaTest
{
    [TestMethod]
    public void nvcudaTest_Empty()
    {
        nvcuda.Empty();
    }
}
