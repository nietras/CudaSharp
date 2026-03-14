using System.Runtime.InteropServices;

namespace CudaSharp;

public static partial class nvcuda
{
    public readonly record struct CUcontext(IntPtr Value);
    public readonly record struct CUdevice(int Value);
    public readonly record struct CUdeviceptr(IntPtr Value);
    public readonly record struct CUevent(IntPtr Value);
    public readonly record struct CUfunction(IntPtr Value);
    public readonly record struct CUmodule(IntPtr Value);
    public readonly record struct CUstream(IntPtr Value);
    public readonly record struct CUsurfObject(ulong Value);
    public readonly record struct CUsurfref(IntPtr Value);
    public readonly record struct CUtexObject(ulong Value);
    public readonly record struct CUtexref(IntPtr Value);
    public readonly record struct CUarray(IntPtr Value);
    public readonly record struct CUgraph(IntPtr Value);
    public readonly record struct CUgraphNode(IntPtr Value);
    public readonly record struct CUgraphExec(IntPtr Value);
    public readonly record struct CUexternalMemory(IntPtr Value);
    public readonly record struct CUexternalSemaphore(IntPtr Value);

    /// <summary>
    /// CUDA result codes
    /// </summary>
    public enum CUresult
    {
        CUDA_SUCCESS = 0,
        CUDA_ERROR_INVALID_VALUE = 1,
        CUDA_ERROR_OUT_OF_MEMORY = 2,
        CUDA_ERROR_NOT_INITIALIZED = 3,
        CUDA_ERROR_DEINITIALIZED = 4,
        CUDA_ERROR_NO_DEVICE = 100,
        CUDA_ERROR_INVALID_DEVICE = 101,
        CUDA_ERROR_INVALID_CONTEXT = 201,
        CUDA_ERROR_MAP_FAILED = 205,
        CUDA_ERROR_UNMAP_FAILED = 206,
        CUDA_ERROR_NOT_FOUND = 300,
        CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
        CUDA_ERROR_INVALID_IMAGE = 200,
        CUDA_ERROR_LAUNCH_FAILED = 702,
        CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
        CUDA_ERROR_LAUNCH_TIMEOUT = 704,
        CUDA_ERROR_LAUNCH_PARAM_COUNT_MISMATCH = 705,
        CUDA_ERROR_LAUNCH_PARAM_INVALID = 706,
        CUDA_ERROR_LAUNCH_PARAM_NOT_ADDRESSABLE = 707,
        CUDA_ERROR_LAUNCH_PARAM_UNKNOWN = 708,
        CUDA_ERROR_INVALID_DEVICE_FUNCTION = 709,
        CUDA_ERROR_NOT_READY = 600,
        CUDA_ERROR_MODULE_NOT_FOUND = 304,
        CUDA_ERROR_FILE_NOT_FOUND = 301,
        CUDA_ERROR_INVALID_DEVICE_POINTER = 400,
        CUDA_ERROR_INVALID_PITCH_VALUE = 210,
        CUDA_ERROR_INVALID_CUDAARRAY = 211,
        CUDA_ERROR_INVALID_TEXTURE = 212,
        CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 213,
        CUDA_ERROR_INVALID_SOURCE = 302,
        CUDA_ERROR_INVALID_ADDRESS = 401,
    }

    public enum CUevent_flags
    {
        CU_EVENT_DEFAULT = 0x0,
        CU_EVENT_BLOCKING_SYNC = 0x1,
        CU_EVENT_DISABLE_TIMING = 0x2,
        CU_EVENT_INTERPROCESS = 0x4,
    }

    public enum CUctx_flags
    {
        CU_CTX_SCHED_AUTO = 0x00,
        CU_CTX_SCHED_SPIN = 0x01,
        CU_CTX_SCHED_YIELD = 0x02,
        CU_CTX_SCHED_BLOCKING_SYNC = 0x04,
        CU_CTX_MAP_HOST = 0x08,
        CU_CTX_LMEM_RESIZE_TO_MAX = 0x10,
    }

    public enum CUmemhostalloc_flags
    {
        CU_MEMHOSTALLOC_PORTABLE = 0x01,
        CU_MEMHOSTALLOC_DEVICEMAP = 0x02,
        CU_MEMHOSTALLOC_WRITECOMBINED = 0x04,
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUdevprop
    {
        public int maxThreadsPerBlock;
        public int maxThreadsDim0;
        public int maxThreadsDim1;
        public int maxThreadsDim2;
        public int maxGridSize0;
        public int maxGridSize1;
        public int maxGridSize2;
        public int sharedMemPerBlock;
        public int totalConstantMemory;
        public int SIMDWidth;
        public int memPitch;
        public int regsPerBlock;
        public int clockRate;
        public int textureAlign;
    }

    public enum CUdevice_attribute
    {
        CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
        CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
        CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
        CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
        CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
        CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
        CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
        CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
        CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
        CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
        CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
        CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
        CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
        CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
        CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUuuid
    {
        public unsafe fixed byte bytes[16];
    }

    public enum CUaddress_mode
    {
        CU_TR_ADDRESS_MODE_WRAP = 0,
        CU_TR_ADDRESS_MODE_CLAMP = 1,
        CU_TR_ADDRESS_MODE_MIRROR = 2,
        CU_TR_ADDRESS_MODE_BORDER = 3,
    }

    public enum CUfilter_mode
    {
        CU_TR_FILTER_MODE_POINT = 0,
        CU_TR_FILTER_MODE_LINEAR = 1,
    }

    public enum CUarray_format
    {
        CU_AD_FORMAT_UNSIGNED_INT8 = 0x01,
        CU_AD_FORMAT_UNSIGNED_INT16 = 0x02,
        CU_AD_FORMAT_UNSIGNED_INT32 = 0x03,
        CU_AD_FORMAT_SIGNED_INT8 = 0x08,
        CU_AD_FORMAT_SIGNED_INT16 = 0x09,
        CU_AD_FORMAT_SIGNED_INT32 = 0x0a,
        CU_AD_FORMAT_HALF = 0x10,
        CU_AD_FORMAT_FLOAT = 0x20,
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_ARRAY_DESCRIPTOR
    {
        public nuint Width;
        public nuint Height;
        public CUarray_format Format;
        public uint NumChannels;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_RESOURCE_DESC
    {
        public CUresourcetype resType;
        public CUDA_RESOURCE_DESC_UNION res;
        public uint flags;
    }

    [StructLayout(LayoutKind.Explicit)]
    public struct CUDA_RESOURCE_DESC_UNION
    {
        [FieldOffset(0)]
        public CUDA_RESOURCE_DESC_ARRAY array;
        [FieldOffset(0)]
        public CUDA_RESOURCE_DESC_MIPMAPPED_ARRAY mipmap;
        [FieldOffset(0)]
        public CUDA_RESOURCE_DESC_LINEAR linear;
        [FieldOffset(0)]
        public CUDA_RESOURCE_DESC_PITCH2D pitch2D;
        [FieldOffset(0)]
        public unsafe fixed int reserved[16];
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_RESOURCE_DESC_ARRAY
    {
        public CUarray hArray;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_RESOURCE_DESC_MIPMAPPED_ARRAY
    {
        public IntPtr hMipmappedArray;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_RESOURCE_DESC_LINEAR
    {
        public CUdeviceptr devPtr;
        public CUarray_format format;
        public uint numChannels;
        public nuint sizeInBytes;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_RESOURCE_DESC_PITCH2D
    {
        public CUdeviceptr devPtr;
        public CUarray_format format;
        public uint numChannels;
        public nuint width;
        public nuint height;
        public nuint pitchInBytes;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_TEXTURE_DESC
    {
        public CUaddress_mode addressMode0;
        public CUaddress_mode addressMode1;
        public CUaddress_mode addressMode2;
        public CUfilter_mode filterMode;
        public uint flags;
        public uint maxAnisotropy;
        public CUfilter_mode mipmapFilterMode;
        public float mipmapLevelBias;
        public float minMipmapLevelClamp;
        public float maxMipmapLevelClamp;
        public float borderColor0;
        public float borderColor1;
        public float borderColor2;
        public float borderColor3;
        public unsafe fixed int reserved[15];
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_RESOURCE_VIEW_DESC
    {
        public CUresourceViewFormat format;
        public nuint width;
        public nuint height;
        public nuint depth;
        public uint firstMipmapLevel;
        public uint lastMipmapLevel;
        public uint firstLayer;
        public uint lastLayer;
        public unsafe fixed uint reserved[16];
    }

    public enum CUresourcetype
    {
        CU_RESOURCE_TYPE_ARRAY = 0x00,
        CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 0x01,
        CU_RESOURCE_TYPE_LINEAR = 0x02,
        CU_RESOURCE_TYPE_PITCH2D = 0x03,
    }

    public enum CUresourceViewFormat
    {
        CU_RESVIEWFORMAT_NONE = 0x00,
        CU_RESVIEWFORMAT_UINT_1CHANNEL = 0x01,
        CU_RESVIEWFORMAT_UINT_2CHANNEL = 0x02,
        CU_RESVIEWFORMAT_UINT_4CHANNEL = 0x03,
        CU_RESVIEWFORMAT_SINT_1CHANNEL = 0x04,
        CU_RESVIEWFORMAT_SINT_2CHANNEL = 0x05,
        CU_RESVIEWFORMAT_SINT_4CHANNEL = 0x06,
        CU_RESVIEWFORMAT_FLOAT_1CHANNEL = 0x07,
        CU_RESVIEWFORMAT_FLOAT_2CHANNEL = 0x08,
        CU_RESVIEWFORMAT_FLOAT_4CHANNEL = 0x09,
        CU_RESVIEWFORMAT_HALF_1CHANNEL = 0x0a,
        CU_RESVIEWFORMAT_HALF_2CHANNEL = 0x0b,
        CU_RESVIEWFORMAT_HALF_4CHANNEL = 0x0c,
    }

    public enum CUlimit
    {
        CU_LIMIT_STACK_SIZE = 0x00,
        CU_LIMIT_PRINTF_FIFO_SIZE = 0x01,
        CU_LIMIT_MALLOC_HEAP_SIZE = 0x02,
        CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 0x03,
        CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 0x04,
        CU_LIMIT_MAX_L2_FETCH_GRANULARITY = 0x05,
        CU_LIMIT_PERSISTENT_L2_CACHE_SIZE = 0x06,
    }

    public enum CUfunc_cache
    {
        CU_FUNC_CACHE_PREFER_NONE = 0x00,
        CU_FUNC_CACHE_PREFER_SHARED = 0x01,
        CU_FUNC_CACHE_PREFER_L1 = 0x02,
        CU_FUNC_CACHE_PREFER_EQUAL = 0x03,
    }

    public enum CUsharedconfig
    {
        CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = 0x00,
        CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = 0x01,
        CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = 0x02,
    }

    public enum CUprofiler_outputMode
    {
        CU_OUT_KEY_VALUE_PAIR = 0x00,
        CU_OUT_CSV = 0x01,
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_MEMCPY2D
    {
        public nuint srcXInBytes;
        public nuint srcY;
        public CUmemorytype srcMemoryType;
        public IntPtr srcHost;
        public CUdeviceptr srcDevice;
        public CUarray srcArray;
        public nuint srcPitch;

        public nuint dstXInBytes;
        public nuint dstY;
        public CUmemorytype dstMemoryType;
        public IntPtr dstHost;
        public CUdeviceptr dstDevice;
        public CUarray dstArray;
        public nuint dstPitch;

        public nuint WidthInBytes;
        public nuint Height;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_MEMCPY3D
    {
        public nuint srcXInBytes;
        public nuint srcY;
        public nuint srcZ;
        public nuint srcLOD;
        public CUmemorytype srcMemoryType;
        public IntPtr srcHost;
        public CUdeviceptr srcDevice;
        public CUarray srcArray;
        public IntPtr srcContext;
        public nuint srcPitch;
        public nuint srcHeight;

        public nuint dstXInBytes;
        public nuint dstY;
        public nuint dstZ;
        public nuint dstLOD;
        public CUmemorytype dstMemoryType;
        public IntPtr dstHost;
        public CUdeviceptr dstDevice;
        public CUarray dstArray;
        public IntPtr dstContext;
        public nuint dstPitch;
        public nuint dstHeight;

        public nuint WidthInBytes;
        public nuint Height;
        public nuint Depth;
    }

    public enum CUmemorytype
    {
        CU_MEMORYTYPE_HOST = 0x01,
        CU_MEMORYTYPE_DEVICE = 0x02,
        CU_MEMORYTYPE_ARRAY = 0x03,
        CU_MEMORYTYPE_UNIFIED = 0x04,
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_KERNEL_NODE_PARAMS
    {
        public CUfunction func;
        public uint gridDimX;
        public uint gridDimY;
        public uint gridDimZ;
        public uint blockDimX;
        public uint blockDimY;
        public uint blockDimZ;
        public uint sharedMemBytes;
        public IntPtr kernelParams;
        public IntPtr extra;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_MEMCPY_NODE_PARAMS
    {
        public CUDA_MEMCPY3D copyParams;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_MEMSET_NODE_PARAMS
    {
        public CUdeviceptr dst;
        public nuint pitch;
        public uint value;
        public uint elementSize;
        public nuint width;
        public nuint height;
    }

    public enum CUstreamBatchMemOpType
    {
        CU_STREAM_MEM_OP_WAIT_VALUE_32 = 1,
        CU_STREAM_MEM_OP_WRITE_VALUE_32 = 2,
        CU_STREAM_MEM_OP_WAIT_VALUE_64 = 4,
        CU_STREAM_MEM_OP_WRITE_VALUE_64 = 5,
        CU_STREAM_MEM_OP_BARRIER = 6,
        CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = 3,
    }

    [StructLayout(LayoutKind.Explicit)]
    public struct CUstreamBatchMemOpParams
    {
        [FieldOffset(0)] public CUstreamBatchMemOpType operation;
        [FieldOffset(8)] public CUstreamWaitValue_params waitValue;
        [FieldOffset(8)] public CUstreamWriteValue_params writeValue;
        [FieldOffset(8)] public CUstreamFlushRemoteWrites_params flushRemoteWrites;
        [FieldOffset(8)] public CUstreamMemOpBarrier_params barrier;
        [FieldOffset(8)] public unsafe fixed ulong pad[6];
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUstreamWaitValue_params
    {
        public CUstreamBatchMemOpType operation;
        public CUdeviceptr address;
        public CUstreamWaitValue_params_union value;
        public uint flags;
        public CUdeviceptr alias;
    }

    [StructLayout(LayoutKind.Explicit)]
    public struct CUstreamWaitValue_params_union
    {
        [FieldOffset(0)] public uint value;
        [FieldOffset(0)] public ulong value64;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUstreamWriteValue_params
    {
        public CUstreamBatchMemOpType operation;
        public CUdeviceptr address;
        public CUstreamWriteValue_params_union value;
        public uint flags;
        public CUdeviceptr alias;
    }

    [StructLayout(LayoutKind.Explicit)]
    public struct CUstreamWriteValue_params_union
    {
        [FieldOffset(0)] public uint value;
        [FieldOffset(0)] public ulong value64;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUstreamFlushRemoteWrites_params
    {
        public CUstreamBatchMemOpType operation;
        public uint flags;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUstreamMemOpBarrier_params
    {
        public CUstreamBatchMemOpType operation;
        public uint flags;
    }

    public enum CUstreamWaitValue_flags
    {
        CU_STREAM_WAIT_VALUE_GEQ = 0x0,
        CU_STREAM_WAIT_VALUE_EQ = 0x1,
        CU_STREAM_WAIT_VALUE_AND = 0x2,
        CU_STREAM_WAIT_VALUE_NOR = 0x3,
        CU_STREAM_WAIT_VALUE_FLUSH = 1 << 30,
    }

    public enum CUstreamWriteValue_flags
    {
        CU_STREAM_WRITE_VALUE_DEFAULT = 0x0,
        CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER = 0x1,
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC
    {
        public CUexternalMemoryHandleType type;
        public CUDA_EXTERNAL_MEMORY_HANDLE_DESC_UNION handle;
        public ulong size;
        public uint flags;
    }

    [StructLayout(LayoutKind.Explicit)]
    public struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC_UNION
    {
        [FieldOffset(0)] public int fd;
        [FieldOffset(0)] public CUDA_EXTERNAL_MEMORY_HANDLE_DESC_WIN32 win32;
        [FieldOffset(0)] public IntPtr nvSciBufObject;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC_WIN32
    {
        public IntPtr handle;
        public IntPtr name;
    }

    public enum CUexternalMemoryHandleType
    {
        CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = 1,
        CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 = 2,
        CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3,
        CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP = 4,
        CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE = 5,
        CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE = 6,
        CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT = 7,
        CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF = 8,
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC
    {
        public ulong offset;
        public ulong size;
        public uint flags;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC
    {
        public ulong offset;
        public CUDA_ARRAY3D_DESCRIPTOR arrayDesc;
        public uint numLevels;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_ARRAY3D_DESCRIPTOR
    {
        public nuint Width;
        public nuint Height;
        public nuint Depth;
        public CUarray_format Format;
        public uint NumChannels;
        public uint Flags;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC
    {
        public CUexternalSemaphoreHandleType type;
        public CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_UNION handle;
        public uint flags;
    }

    [StructLayout(LayoutKind.Explicit)]
    public struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_UNION
    {
        [FieldOffset(0)] public int fd;
        [FieldOffset(0)] public CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_WIN32 win32;
        [FieldOffset(0)] public IntPtr nvSciSyncObj;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_WIN32
    {
        public IntPtr handle;
        public IntPtr name;
    }

    public enum CUexternalSemaphoreHandleType
    {
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD = 1,
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32 = 2,
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3,
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE = 4,
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE = 5,
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC = 6,
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX = 7,
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT = 8,
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD = 9,
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32 = 10,
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS
    {
        public CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_PARAMS params_;
        public uint flags;
        public unsafe fixed uint reserved[16];
    }

    [StructLayout(LayoutKind.Explicit)]
    public struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_PARAMS
    {
        [FieldOffset(0)] public CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_FENCE fence;
        [FieldOffset(0)] public IntPtr nvSciSyncObj;
        [FieldOffset(0)] public CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_KEYED_MUTEX keyedMutex;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_FENCE
    {
        public ulong value;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_KEYED_MUTEX
    {
        public ulong key;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS
    {
        public CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_PARAMS params_;
        public uint flags;
        public unsafe fixed uint reserved[16];
    }

    [StructLayout(LayoutKind.Explicit)]
    public struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_PARAMS
    {
        [FieldOffset(0)] public CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_FENCE fence;
        [FieldOffset(0)] public IntPtr nvSciSyncObj;
        [FieldOffset(0)] public CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_KEYED_MUTEX keyedMutex;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_FENCE
    {
        public ulong value;
        public uint timeoutMs;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_KEYED_MUTEX
    {
        public ulong key;
        public uint timeoutMs;
    }
}
