using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace CudaSharp;

static class Throws
{
    [DoesNotReturn]
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static void Throw<TResult>(TResult result, string message)
        where TResult : unmanaged, Enum
    {
        throw new CudaException<TResult>(result, message);
    }
}
