using System;
using CudaSharp.Mnist;
using static CudaSharp.nvcuda;

namespace CudaSharp.Mnist;

public sealed class GpuMemoryArena : IDisposable
{
    CUdeviceptr _basePtr;
    nuint _totalBytes;
    nuint _currentOffset;

    public CUdeviceptr BasePtr => _basePtr;
    public nuint TotalBytes => _totalBytes;

    public GpuMemoryArena()
    {
    }

    public void Allocate(nuint totalBytes)
    {
        if (_basePtr != default)
            throw new InvalidOperationException("Arena already allocated.");

        _totalBytes = totalBytes;
        cuMemAlloc(out _basePtr, _totalBytes).Ok();
        _currentOffset = 0;
    }

    public CUdeviceptr Rent(nuint sizeInBytes)
    {
        if (_basePtr == default)
            throw new InvalidOperationException("Arena not allocated yet.");

        // Ensure alignment (e.g. 256 bytes for good memory transaction alignment)
        nuint alignment = 256;
        nuint alignedSize = (sizeInBytes + alignment - 1) & ~(alignment - 1);

        if (_currentOffset + alignedSize > _totalBytes)
            throw new OutOfMemoryException($"Arena out of memory. Capacity: {_totalBytes}, Required: {_currentOffset + alignedSize}");

        var ptr = new CUdeviceptr(_basePtr.Value + (nint)_currentOffset);
        _currentOffset += alignedSize;
        return ptr;
    }

    public void Dispose()
    {
        if (_basePtr != default)
        {
            cuMemFree(_basePtr).Ok();
            _basePtr = default;
            _totalBytes = 0;
            _currentOffset = 0;
        }
    }
}
