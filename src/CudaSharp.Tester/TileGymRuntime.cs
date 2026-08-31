using System;
using System.Runtime.CompilerServices;
using CudaSharp.Tile;
using static CudaSharp.nvcuda;

namespace CudaSharp.Tester;

sealed class TileGymRuntime : IDisposable
{
    public TileGymRuntime(int deviceOrdinal)
    {
        CuInit.EnsureInit();
        cuDeviceGet(out var device, deviceOrdinal).Ok();
        Device = device;
        Architecture = TileCppCompiler.GetArchitecture(device);
        cuDevicePrimaryCtxRetain(out var context, device).Ok();
        Context = context;
        try
        {
            cuCtxSetCurrent(context).Ok();
            cuStreamCreate(out var stream, 0).Ok();
            Stream = stream;
        }
        catch
        {
            cuDevicePrimaryCtxRelease(device).Ok();
            throw;
        }
    }

    public CUdevice Device { get; }
    public CUcontext Context { get; }
    public CUstream Stream { get; }
    public int Architecture { get; }

    public CudaBuffer<T> Allocate<T>(int length) where T : unmanaged => new(length);

    public void Dispose()
    {
        cuStreamDestroy(Stream).Ok();
        cuCtxSetCurrent(default).Ok();
        cuDevicePrimaryCtxRelease(Device).Ok();
    }
}

sealed class CudaBuffer<T> : IDisposable where T : unmanaged
{
    public CudaBuffer(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        Length = length;
        ByteLength = checked((nuint)length * (nuint)Unsafe.SizeOf<T>());
        cuMemAlloc_v2(out var pointer, ByteLength).Ok();
        Pointer = pointer;
    }

    public int Length { get; }
    public nuint ByteLength { get; }
    public CUdeviceptr Pointer { get; }

    public unsafe void CopyFrom(ReadOnlySpan<T> source)
    {
        if (source.Length != Length)
        {
            throw new ArgumentException("Source length must match the device buffer.", nameof(source));
        }
        fixed (T* pointer = source)
        {
            cuMemcpyHtoD_v2(Pointer, (IntPtr)pointer, ByteLength).Ok();
        }
    }

    public unsafe T[] CopyToHost()
    {
        var destination = GC.AllocateUninitializedArray<T>(Length);
        fixed (T* pointer = destination)
        {
            cuMemcpyDtoH_v2((IntPtr)pointer, Pointer, ByteLength).Ok();
        }
        return destination;
    }

    public void Clear() => cuMemsetD8_v2(Pointer, 0, ByteLength).Ok();

    public void Dispose() => cuMemFree_v2(Pointer).Ok();
}
