using System.Runtime.CompilerServices;

namespace CudaSharp;

public static partial class nvcuda
{
    extension(CUresult result)
    {
        public void Ok()
        {
            if (result != CUresult.CUDA_SUCCESS)
            {
                Throws.Throw(result, result.ToStringFast());
            }
        }
        public bool IsOk() => result == CUresult.CUDA_SUCCESS;
        public bool IsError() => result != CUresult.CUDA_SUCCESS;

        public string ToStringFast() => result switch
        {
            CUresult.CUDA_SUCCESS => nameof(CUresult.CUDA_SUCCESS),
            CUresult.CUDA_ERROR_INVALID_VALUE => nameof(CUresult.CUDA_ERROR_INVALID_VALUE),
            CUresult.CUDA_ERROR_OUT_OF_MEMORY => nameof(CUresult.CUDA_ERROR_OUT_OF_MEMORY),
            CUresult.CUDA_ERROR_NOT_INITIALIZED => nameof(CUresult.CUDA_ERROR_NOT_INITIALIZED),
            CUresult.CUDA_ERROR_DEINITIALIZED => nameof(CUresult.CUDA_ERROR_DEINITIALIZED),
            CUresult.CUDA_ERROR_NO_DEVICE => nameof(CUresult.CUDA_ERROR_NO_DEVICE),
            CUresult.CUDA_ERROR_INVALID_DEVICE => nameof(CUresult.CUDA_ERROR_INVALID_DEVICE),
            CUresult.CUDA_ERROR_INVALID_CONTEXT => nameof(CUresult.CUDA_ERROR_INVALID_CONTEXT),
            CUresult.CUDA_ERROR_MAP_FAILED => nameof(CUresult.CUDA_ERROR_MAP_FAILED),
            CUresult.CUDA_ERROR_UNMAP_FAILED => nameof(CUresult.CUDA_ERROR_UNMAP_FAILED),
            CUresult.CUDA_ERROR_NOT_FOUND => nameof(CUresult.CUDA_ERROR_NOT_FOUND),
            CUresult.CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES => nameof(CUresult.CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES),
            CUresult.CUDA_ERROR_INVALID_IMAGE => nameof(CUresult.CUDA_ERROR_INVALID_IMAGE),
            CUresult.CUDA_ERROR_LAUNCH_FAILED => nameof(CUresult.CUDA_ERROR_LAUNCH_FAILED),
            CUresult.CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING => nameof(CUresult.CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING),
            CUresult.CUDA_ERROR_LAUNCH_TIMEOUT => nameof(CUresult.CUDA_ERROR_LAUNCH_TIMEOUT),
            CUresult.CUDA_ERROR_LAUNCH_PARAM_COUNT_MISMATCH => nameof(CUresult.CUDA_ERROR_LAUNCH_PARAM_COUNT_MISMATCH),
            CUresult.CUDA_ERROR_LAUNCH_PARAM_INVALID => nameof(CUresult.CUDA_ERROR_LAUNCH_PARAM_INVALID),
            CUresult.CUDA_ERROR_LAUNCH_PARAM_NOT_ADDRESSABLE => nameof(CUresult.CUDA_ERROR_LAUNCH_PARAM_NOT_ADDRESSABLE),
            CUresult.CUDA_ERROR_LAUNCH_PARAM_UNKNOWN => nameof(CUresult.CUDA_ERROR_LAUNCH_PARAM_UNKNOWN),
            CUresult.CUDA_ERROR_INVALID_DEVICE_FUNCTION => nameof(CUresult.CUDA_ERROR_INVALID_DEVICE_FUNCTION),
            CUresult.CUDA_ERROR_NOT_READY => nameof(CUresult.CUDA_ERROR_NOT_READY),
            CUresult.CUDA_ERROR_MODULE_NOT_FOUND => nameof(CUresult.CUDA_ERROR_MODULE_NOT_FOUND),
            CUresult.CUDA_ERROR_FILE_NOT_FOUND => nameof(CUresult.CUDA_ERROR_FILE_NOT_FOUND),
            CUresult.CUDA_ERROR_INVALID_DEVICE_POINTER => nameof(CUresult.CUDA_ERROR_INVALID_DEVICE_POINTER),
            CUresult.CUDA_ERROR_INVALID_PITCH_VALUE => nameof(CUresult.CUDA_ERROR_INVALID_PITCH_VALUE),
            CUresult.CUDA_ERROR_INVALID_CUDAARRAY => nameof(CUresult.CUDA_ERROR_INVALID_CUDAARRAY),
            CUresult.CUDA_ERROR_INVALID_TEXTURE => nameof(CUresult.CUDA_ERROR_INVALID_TEXTURE),
            CUresult.CUDA_ERROR_INVALID_GRAPHICS_CONTEXT => nameof(CUresult.CUDA_ERROR_INVALID_GRAPHICS_CONTEXT),
            CUresult.CUDA_ERROR_INVALID_SOURCE => nameof(CUresult.CUDA_ERROR_INVALID_SOURCE),
            CUresult.CUDA_ERROR_INVALID_ADDRESS => nameof(CUresult.CUDA_ERROR_INVALID_ADDRESS),
            _ => $"CUDA_ERROR_UNKNOWN:{result}",
        };
    }

    [SkipLocalsInit]
    public unsafe static CUresult cuLaunchKernel<T1, T2, T3, T4, T5>(CUfunction function,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes, CUstream stream,
        T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5)
        where T1 : unmanaged
        where T2 : unmanaged
        where T3 : unmanaged
        where T4 : unmanaged
        where T5 : unmanaged
    {
        var kernelParams = stackalloc IntPtr[]
        {
            (IntPtr)(&arg1),
            (IntPtr)(&arg2),
            (IntPtr)(&arg3),
            (IntPtr)(&arg4),
            (IntPtr)(&arg5)
        };
        return cuLaunchKernel(function,
            gridDimX, gridDimY, gridDimZ,
            blockDimX, blockDimY, blockDimZ,
            sharedMemBytes, stream,
            kernelParams, null);
    }

}
