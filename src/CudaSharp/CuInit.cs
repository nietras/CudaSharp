using static CudaSharp.nvcuda;

namespace CudaSharp;

public static class CuInit
{
    static volatile CUresult s_result = (CUresult)(int.MaxValue - 1);

    public static void EnsureInit()
    {
        if (s_result.IsError()) { TryEnsureInit().Ok(); }
    }
    public static CUresult TryEnsureInit()
    {
        // cuInit is needed per process and *can* be called multiple times, but
        // to centralize this and limit calls to it a static cached result is
        // used in managed code allowing to skip if already succeeded.
        if (s_result.IsError()) { s_result = cuInit(); }
        return s_result;
    }
}
