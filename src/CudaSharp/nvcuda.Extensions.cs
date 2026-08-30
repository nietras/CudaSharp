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
            CUresult.CUDA_ERROR_PROFILER_DISABLED => nameof(CUresult.CUDA_ERROR_PROFILER_DISABLED),
            CUresult.CUDA_ERROR_PROFILER_NOT_INITIALIZED => nameof(CUresult.CUDA_ERROR_PROFILER_NOT_INITIALIZED),
            CUresult.CUDA_ERROR_PROFILER_ALREADY_STARTED => nameof(CUresult.CUDA_ERROR_PROFILER_ALREADY_STARTED),
            CUresult.CUDA_ERROR_PROFILER_ALREADY_STOPPED => nameof(CUresult.CUDA_ERROR_PROFILER_ALREADY_STOPPED),
            CUresult.CUDA_ERROR_STUB_LIBRARY => nameof(CUresult.CUDA_ERROR_STUB_LIBRARY),
            CUresult.CUDA_ERROR_CALL_REQUIRES_NEWER_DRIVER => nameof(CUresult.CUDA_ERROR_CALL_REQUIRES_NEWER_DRIVER),
            CUresult.CUDA_ERROR_DEVICE_UNAVAILABLE => nameof(CUresult.CUDA_ERROR_DEVICE_UNAVAILABLE),
            CUresult.CUDA_ERROR_NO_DEVICE => nameof(CUresult.CUDA_ERROR_NO_DEVICE),
            CUresult.CUDA_ERROR_INVALID_DEVICE => nameof(CUresult.CUDA_ERROR_INVALID_DEVICE),
            CUresult.CUDA_ERROR_DEVICE_NOT_LICENSED => nameof(CUresult.CUDA_ERROR_DEVICE_NOT_LICENSED),
            CUresult.CUDA_ERROR_INVALID_IMAGE => nameof(CUresult.CUDA_ERROR_INVALID_IMAGE),
            CUresult.CUDA_ERROR_INVALID_CONTEXT => nameof(CUresult.CUDA_ERROR_INVALID_CONTEXT),
            CUresult.CUDA_ERROR_CONTEXT_ALREADY_CURRENT => nameof(CUresult.CUDA_ERROR_CONTEXT_ALREADY_CURRENT),
            CUresult.CUDA_ERROR_MAP_FAILED => nameof(CUresult.CUDA_ERROR_MAP_FAILED),
            CUresult.CUDA_ERROR_UNMAP_FAILED => nameof(CUresult.CUDA_ERROR_UNMAP_FAILED),
            CUresult.CUDA_ERROR_ARRAY_IS_MAPPED => nameof(CUresult.CUDA_ERROR_ARRAY_IS_MAPPED),
            CUresult.CUDA_ERROR_ALREADY_MAPPED => nameof(CUresult.CUDA_ERROR_ALREADY_MAPPED),
            CUresult.CUDA_ERROR_NO_BINARY_FOR_GPU => nameof(CUresult.CUDA_ERROR_NO_BINARY_FOR_GPU),
            CUresult.CUDA_ERROR_ALREADY_ACQUIRED => nameof(CUresult.CUDA_ERROR_ALREADY_ACQUIRED),
            CUresult.CUDA_ERROR_NOT_MAPPED => nameof(CUresult.CUDA_ERROR_NOT_MAPPED),
            CUresult.CUDA_ERROR_NOT_MAPPED_AS_ARRAY => nameof(CUresult.CUDA_ERROR_NOT_MAPPED_AS_ARRAY),
            CUresult.CUDA_ERROR_NOT_MAPPED_AS_POINTER => nameof(CUresult.CUDA_ERROR_NOT_MAPPED_AS_POINTER),
            CUresult.CUDA_ERROR_ECC_UNCORRECTABLE => nameof(CUresult.CUDA_ERROR_ECC_UNCORRECTABLE),
            CUresult.CUDA_ERROR_UNSUPPORTED_LIMIT => nameof(CUresult.CUDA_ERROR_UNSUPPORTED_LIMIT),
            CUresult.CUDA_ERROR_CONTEXT_ALREADY_IN_USE => nameof(CUresult.CUDA_ERROR_CONTEXT_ALREADY_IN_USE),
            CUresult.CUDA_ERROR_PEER_ACCESS_UNSUPPORTED => nameof(CUresult.CUDA_ERROR_PEER_ACCESS_UNSUPPORTED),
            CUresult.CUDA_ERROR_INVALID_PTX => nameof(CUresult.CUDA_ERROR_INVALID_PTX),
            CUresult.CUDA_ERROR_INVALID_GRAPHICS_CONTEXT => nameof(CUresult.CUDA_ERROR_INVALID_GRAPHICS_CONTEXT),
            CUresult.CUDA_ERROR_NVLINK_UNCORRECTABLE => nameof(CUresult.CUDA_ERROR_NVLINK_UNCORRECTABLE),
            CUresult.CUDA_ERROR_JIT_COMPILER_NOT_FOUND => nameof(CUresult.CUDA_ERROR_JIT_COMPILER_NOT_FOUND),
            CUresult.CUDA_ERROR_UNSUPPORTED_PTX_VERSION => nameof(CUresult.CUDA_ERROR_UNSUPPORTED_PTX_VERSION),
            CUresult.CUDA_ERROR_JIT_COMPILATION_DISABLED => nameof(CUresult.CUDA_ERROR_JIT_COMPILATION_DISABLED),
            CUresult.CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY => nameof(CUresult.CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY),
            CUresult.CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC => nameof(CUresult.CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC),
            CUresult.CUDA_ERROR_CONTAINED => nameof(CUresult.CUDA_ERROR_CONTAINED),
            CUresult.CUDA_ERROR_INVALID_SOURCE => nameof(CUresult.CUDA_ERROR_INVALID_SOURCE),
            CUresult.CUDA_ERROR_FILE_NOT_FOUND => nameof(CUresult.CUDA_ERROR_FILE_NOT_FOUND),
            CUresult.CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND => nameof(CUresult.CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND),
            CUresult.CUDA_ERROR_SHARED_OBJECT_INIT_FAILED => nameof(CUresult.CUDA_ERROR_SHARED_OBJECT_INIT_FAILED),
            CUresult.CUDA_ERROR_OPERATING_SYSTEM => nameof(CUresult.CUDA_ERROR_OPERATING_SYSTEM),
            CUresult.CUDA_ERROR_INVALID_HANDLE => nameof(CUresult.CUDA_ERROR_INVALID_HANDLE),
            CUresult.CUDA_ERROR_ILLEGAL_STATE => nameof(CUresult.CUDA_ERROR_ILLEGAL_STATE),
            CUresult.CUDA_ERROR_LOSSY_QUERY => nameof(CUresult.CUDA_ERROR_LOSSY_QUERY),
            CUresult.CUDA_ERROR_NOT_FOUND => nameof(CUresult.CUDA_ERROR_NOT_FOUND),
            CUresult.CUDA_ERROR_NOT_READY => nameof(CUresult.CUDA_ERROR_NOT_READY),
            CUresult.CUDA_ERROR_ILLEGAL_ADDRESS => nameof(CUresult.CUDA_ERROR_ILLEGAL_ADDRESS),
            CUresult.CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES => nameof(CUresult.CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES),
            CUresult.CUDA_ERROR_LAUNCH_TIMEOUT => nameof(CUresult.CUDA_ERROR_LAUNCH_TIMEOUT),
            CUresult.CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING => nameof(CUresult.CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING),
            CUresult.CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => nameof(CUresult.CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED),
            CUresult.CUDA_ERROR_PEER_ACCESS_NOT_ENABLED => nameof(CUresult.CUDA_ERROR_PEER_ACCESS_NOT_ENABLED),
            CUresult.CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE => nameof(CUresult.CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE),
            CUresult.CUDA_ERROR_CONTEXT_IS_DESTROYED => nameof(CUresult.CUDA_ERROR_CONTEXT_IS_DESTROYED),
            CUresult.CUDA_ERROR_ASSERT => nameof(CUresult.CUDA_ERROR_ASSERT),
            CUresult.CUDA_ERROR_TOO_MANY_PEERS => nameof(CUresult.CUDA_ERROR_TOO_MANY_PEERS),
            CUresult.CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED => nameof(CUresult.CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED),
            CUresult.CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED => nameof(CUresult.CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED),
            CUresult.CUDA_ERROR_HARDWARE_STACK_ERROR => nameof(CUresult.CUDA_ERROR_HARDWARE_STACK_ERROR),
            CUresult.CUDA_ERROR_ILLEGAL_INSTRUCTION => nameof(CUresult.CUDA_ERROR_ILLEGAL_INSTRUCTION),
            CUresult.CUDA_ERROR_MISALIGNED_ADDRESS => nameof(CUresult.CUDA_ERROR_MISALIGNED_ADDRESS),
            CUresult.CUDA_ERROR_INVALID_ADDRESS_SPACE => nameof(CUresult.CUDA_ERROR_INVALID_ADDRESS_SPACE),
            CUresult.CUDA_ERROR_INVALID_PC => nameof(CUresult.CUDA_ERROR_INVALID_PC),
            CUresult.CUDA_ERROR_LAUNCH_FAILED => nameof(CUresult.CUDA_ERROR_LAUNCH_FAILED),
            CUresult.CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE => nameof(CUresult.CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE),
            CUresult.CUDA_ERROR_TENSOR_MEMORY_LEAK => nameof(CUresult.CUDA_ERROR_TENSOR_MEMORY_LEAK),
            CUresult.CUDA_ERROR_NOT_PERMITTED => nameof(CUresult.CUDA_ERROR_NOT_PERMITTED),
            CUresult.CUDA_ERROR_NOT_SUPPORTED => nameof(CUresult.CUDA_ERROR_NOT_SUPPORTED),
            CUresult.CUDA_ERROR_SYSTEM_NOT_READY => nameof(CUresult.CUDA_ERROR_SYSTEM_NOT_READY),
            CUresult.CUDA_ERROR_SYSTEM_DRIVER_MISMATCH => nameof(CUresult.CUDA_ERROR_SYSTEM_DRIVER_MISMATCH),
            CUresult.CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE => nameof(CUresult.CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE),
            CUresult.CUDA_ERROR_MPS_CONNECTION_FAILED => nameof(CUresult.CUDA_ERROR_MPS_CONNECTION_FAILED),
            CUresult.CUDA_ERROR_MPS_RPC_FAILURE => nameof(CUresult.CUDA_ERROR_MPS_RPC_FAILURE),
            CUresult.CUDA_ERROR_MPS_SERVER_NOT_READY => nameof(CUresult.CUDA_ERROR_MPS_SERVER_NOT_READY),
            CUresult.CUDA_ERROR_MPS_MAX_CLIENTS_REACHED => nameof(CUresult.CUDA_ERROR_MPS_MAX_CLIENTS_REACHED),
            CUresult.CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED => nameof(CUresult.CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED),
            CUresult.CUDA_ERROR_MPS_CLIENT_TERMINATED => nameof(CUresult.CUDA_ERROR_MPS_CLIENT_TERMINATED),
            CUresult.CUDA_ERROR_CDP_NOT_SUPPORTED => nameof(CUresult.CUDA_ERROR_CDP_NOT_SUPPORTED),
            CUresult.CUDA_ERROR_CDP_VERSION_MISMATCH => nameof(CUresult.CUDA_ERROR_CDP_VERSION_MISMATCH),
            CUresult.CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED => nameof(CUresult.CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED),
            CUresult.CUDA_ERROR_STREAM_CAPTURE_INVALIDATED => nameof(CUresult.CUDA_ERROR_STREAM_CAPTURE_INVALIDATED),
            CUresult.CUDA_ERROR_STREAM_CAPTURE_MERGE => nameof(CUresult.CUDA_ERROR_STREAM_CAPTURE_MERGE),
            CUresult.CUDA_ERROR_STREAM_CAPTURE_UNMATCHED => nameof(CUresult.CUDA_ERROR_STREAM_CAPTURE_UNMATCHED),
            CUresult.CUDA_ERROR_STREAM_CAPTURE_UNJOINED => nameof(CUresult.CUDA_ERROR_STREAM_CAPTURE_UNJOINED),
            CUresult.CUDA_ERROR_STREAM_CAPTURE_ISOLATION => nameof(CUresult.CUDA_ERROR_STREAM_CAPTURE_ISOLATION),
            CUresult.CUDA_ERROR_STREAM_CAPTURE_IMPLICIT => nameof(CUresult.CUDA_ERROR_STREAM_CAPTURE_IMPLICIT),
            CUresult.CUDA_ERROR_CAPTURED_EVENT => nameof(CUresult.CUDA_ERROR_CAPTURED_EVENT),
            CUresult.CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD => nameof(CUresult.CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD),
            CUresult.CUDA_ERROR_TIMEOUT => nameof(CUresult.CUDA_ERROR_TIMEOUT),
            CUresult.CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE => nameof(CUresult.CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE),
            CUresult.CUDA_ERROR_EXTERNAL_DEVICE => nameof(CUresult.CUDA_ERROR_EXTERNAL_DEVICE),
            CUresult.CUDA_ERROR_INVALID_CLUSTER_SIZE => nameof(CUresult.CUDA_ERROR_INVALID_CLUSTER_SIZE),
            CUresult.CUDA_ERROR_FUNCTION_NOT_LOADED => nameof(CUresult.CUDA_ERROR_FUNCTION_NOT_LOADED),
            CUresult.CUDA_ERROR_INVALID_RESOURCE_TYPE => nameof(CUresult.CUDA_ERROR_INVALID_RESOURCE_TYPE),
            CUresult.CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION => nameof(CUresult.CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION),
            CUresult.CUDA_ERROR_KEY_ROTATION => nameof(CUresult.CUDA_ERROR_KEY_ROTATION),
            CUresult.CUDA_ERROR_STREAM_DETACHED => nameof(CUresult.CUDA_ERROR_STREAM_DETACHED),
            CUresult.CUDA_ERROR_GRAPH_RECAPTURE_FAILURE => nameof(CUresult.CUDA_ERROR_GRAPH_RECAPTURE_FAILURE),
            CUresult.CUDA_ERROR_UNKNOWN => nameof(CUresult.CUDA_ERROR_UNKNOWN),
            _ => $"CUDA_ERROR_UNKNOWN:{result}",
        };
    }

    [SkipLocalsInit]
    public unsafe static CUresult cuLaunchKernel<T1, T2>(CUfunction function,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes, CUstream stream,
        T1 arg1, T2 arg2)
        where T1 : unmanaged
        where T2 : unmanaged
    {
        var kernelParams = stackalloc void*[]
        { &arg1, &arg2 };
        return cuLaunchKernel(function,
            gridDimX, gridDimY, gridDimZ,
            blockDimX, blockDimY, blockDimZ,
            sharedMemBytes, stream,
            kernelParams, null);
    }

    [SkipLocalsInit]
    public unsafe static CUresult cuLaunchKernel<T1, T2, T3>(CUfunction function,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes, CUstream stream,
        T1 arg1, T2 arg2, T3 arg3)
        where T1 : unmanaged
        where T2 : unmanaged
        where T3 : unmanaged
    {
        var kernelParams = stackalloc void*[]
        { &arg1, &arg2, &arg3 };
        return cuLaunchKernel(function,
            gridDimX, gridDimY, gridDimZ,
            blockDimX, blockDimY, blockDimZ,
            sharedMemBytes, stream,
            kernelParams, null);
    }

    [SkipLocalsInit]
    public unsafe static CUresult cuLaunchKernel<T1, T2, T3, T4>(CUfunction function,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes, CUstream stream,
        T1 arg1, T2 arg2, T3 arg3, T4 arg4)
        where T1 : unmanaged
        where T2 : unmanaged
        where T3 : unmanaged
        where T4 : unmanaged
    {
        var kernelParams = stackalloc void*[]
        { &arg1, &arg2, &arg3, &arg4 };
        return cuLaunchKernel(function,
            gridDimX, gridDimY, gridDimZ,
            blockDimX, blockDimY, blockDimZ,
            sharedMemBytes, stream,
            kernelParams, null);
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
        var kernelParams = stackalloc void*[]
        { &arg1, &arg2, &arg3, &arg4, &arg5 };
        return cuLaunchKernel(function,
            gridDimX, gridDimY, gridDimZ,
            blockDimX, blockDimY, blockDimZ,
            sharedMemBytes, stream,
            kernelParams, null);
    }

    [SkipLocalsInit]
    public unsafe static CUresult cuLaunchKernel<T1, T2, T3, T4, T5, T6>(CUfunction function,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes, CUstream stream,
        T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6)
        where T1 : unmanaged
        where T2 : unmanaged
        where T3 : unmanaged
        where T4 : unmanaged
        where T5 : unmanaged
        where T6 : unmanaged
    {
        var kernelParams = stackalloc void*[]
        { &arg1, &arg2, &arg3, &arg4, &arg5, &arg6 };
        return cuLaunchKernel(function,
            gridDimX, gridDimY, gridDimZ,
            blockDimX, blockDimY, blockDimZ,
            sharedMemBytes, stream,
            kernelParams, null);
    }
}
