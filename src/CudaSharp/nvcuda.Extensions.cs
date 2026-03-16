namespace CudaSharp;

public static partial class nvcuda
{
    extension(CUresult result)
    {
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
            _ => "CUDA_ERROR_UNKNOWN",
        };
    }
}
