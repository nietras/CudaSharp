namespace CudaSharp;

using System;
using System.Runtime.InteropServices;

public static partial class cudart
{
    static cudart()
    {
        DllResolver.Register();
    }

    const string LibName = nameof(cudart);

    [LibraryImport(LibName)]
    public static partial int cudaFree(IntPtr devPtr);
}
