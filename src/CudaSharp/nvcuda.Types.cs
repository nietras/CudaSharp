namespace CudaSharp;

public static partial class nvcuda
{
    public readonly record struct CUcontext(IntPtr Value);
    public readonly record struct CUdevice(int Value);
    public readonly record struct CUdeviceptr(IntPtr Value);
    public readonly record struct CUevent(IntPtr Value);
    public readonly record struct CUfunction(IntPtr Value);
    public readonly record struct CUkernel(IntPtr Value);
    public readonly record struct CUlinkState(IntPtr Value);
    public readonly record struct CUlibrary(IntPtr Value);
    public readonly record struct CUmodule(IntPtr Value);
    public readonly record struct CUstream(IntPtr Value);
    public readonly record struct CUsurfObject(ulong Value);
    public readonly record struct CUsurfref(IntPtr Value);
    public readonly record struct CUtexObject(ulong Value);
    public readonly record struct CUtexref(IntPtr Value);
    public readonly record struct CUarray(IntPtr Value);
    public readonly record struct CUmipmappedArray(IntPtr Value);
    public readonly record struct CUgraph(IntPtr Value);
    public readonly record struct CUgraphNode(IntPtr Value);
    public readonly record struct CUgraphDeviceNode(IntPtr Value);
    public readonly record struct CUgraphExec(IntPtr Value);
    public readonly record struct CUexternalMemory(IntPtr Value);
    public readonly record struct CUexternalSemaphore(IntPtr Value);
    public readonly record struct CUgraphicsResource(IntPtr Value);

    /// <summary>CUDA driver API error codes.</summary>
    public enum CUresult
    {
        /// <summary>The API call returned with no errors.</summary>
        CUDA_SUCCESS = 0,
        /// <summary>One or more parameters passed to the API call is not within an acceptable range of values.</summary>
        CUDA_ERROR_INVALID_VALUE = 1,
        /// <summary>The API call was unable to allocate enough memory or other resources to perform the requested operation.</summary>
        CUDA_ERROR_OUT_OF_MEMORY = 2,
        /// <summary>The CUDA driver has not been initialized with cuInit() or initialization has failed.</summary>
        CUDA_ERROR_NOT_INITIALIZED = 3,
        /// <summary>The CUDA driver is in the process of shutting down.</summary>
        CUDA_ERROR_DEINITIALIZED = 4,
        /// <summary>The profiler is not initialized for this run.</summary>
        CUDA_ERROR_PROFILER_DISABLED = 5,
        /// <summary>Deprecated. It is no longer an error to attempt to enable or disable profiling without initialization.</summary>
        CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,
        /// <summary>Deprecated. It is no longer an error to call cuProfilerStart() when profiling is already enabled.</summary>
        CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,
        /// <summary>Deprecated. It is no longer an error to call cuProfilerStop() when profiling is already disabled.</summary>
        CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,
        /// <summary>The CUDA driver that the application has loaded is a stub library.</summary>
        CUDA_ERROR_STUB_LIBRARY = 34,
        /// <summary>The API call requires a newer CUDA driver than the one currently installed.</summary>
        CUDA_ERROR_CALL_REQUIRES_NEWER_DRIVER = 36,
        /// <summary>The requested CUDA device is unavailable at the current time.</summary>
        CUDA_ERROR_DEVICE_UNAVAILABLE = 46,
        /// <summary>No CUDA-capable devices were detected by the installed CUDA driver.</summary>
        CUDA_ERROR_NO_DEVICE = 100,
        /// <summary>The supplied device ordinal does not correspond to a valid CUDA device or the requested action is invalid for the specified device.</summary>
        CUDA_ERROR_INVALID_DEVICE = 101,
        /// <summary>The Grid license is not applied.</summary>
        CUDA_ERROR_DEVICE_NOT_LICENSED = 102,
        /// <summary>The device kernel image or CUDA module is invalid.</summary>
        CUDA_ERROR_INVALID_IMAGE = 200,
        /// <summary>There is no context bound to the current thread, or the supplied context handle is not valid.</summary>
        CUDA_ERROR_INVALID_CONTEXT = 201,
        /// <summary>Deprecated. The context supplied as a parameter was already the active context.</summary>
        CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
        /// <summary>A map or register operation has failed.</summary>
        CUDA_ERROR_MAP_FAILED = 205,
        /// <summary>An unmap or unregister operation has failed.</summary>
        CUDA_ERROR_UNMAP_FAILED = 206,
        /// <summary>The specified array is currently mapped and cannot be destroyed.</summary>
        CUDA_ERROR_ARRAY_IS_MAPPED = 207,
        /// <summary>The resource is already mapped.</summary>
        CUDA_ERROR_ALREADY_MAPPED = 208,
        /// <summary>No kernel image is available that is suitable for the device.</summary>
        CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
        /// <summary>The resource has already been acquired.</summary>
        CUDA_ERROR_ALREADY_ACQUIRED = 210,
        /// <summary>The resource is not mapped.</summary>
        CUDA_ERROR_NOT_MAPPED = 211,
        /// <summary>The mapped resource is not available for access as an array.</summary>
        CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
        /// <summary>The mapped resource is not available for access as a pointer.</summary>
        CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,
        /// <summary>An uncorrectable ECC error was detected during execution.</summary>
        CUDA_ERROR_ECC_UNCORRECTABLE = 214,
        /// <summary>The CUlimit passed to the API call is not supported by the active device.</summary>
        CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
        /// <summary>The CUcontext can only be bound to a single CPU thread at a time but is already bound to a CPU thread.</summary>
        CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
        /// <summary>Peer access is not supported across the given devices.</summary>
        CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,
        /// <summary>PTX JIT compilation failed.</summary>
        CUDA_ERROR_INVALID_PTX = 218,
        /// <summary>An error occurred with the OpenGL or DirectX context.</summary>
        CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,
        /// <summary>An uncorrectable NVLink error was detected during execution.</summary>
        CUDA_ERROR_NVLINK_UNCORRECTABLE = 220,
        /// <summary>The PTX JIT compiler library was not found.</summary>
        CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,
        /// <summary>The provided PTX was compiled with an unsupported toolchain.</summary>
        CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222,
        /// <summary>PTX JIT compilation was disabled.</summary>
        CUDA_ERROR_JIT_COMPILATION_DISABLED = 223,
        /// <summary>The CUexecAffinityType passed to the API call is not supported by the active device.</summary>
        CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY = 224,
        /// <summary>The code to be compiled by the PTX JIT contains an unsupported call to cudaDeviceSynchronize.</summary>
        CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC = 225,
        /// <summary>An exception occurred on the device and is now contained by the GPU error containment capability.</summary>
        CUDA_ERROR_CONTAINED = 226,
        /// <summary>The device kernel source is invalid, including device-code compilation or linker errors.</summary>
        CUDA_ERROR_INVALID_SOURCE = 300,
        /// <summary>The specified file was not found.</summary>
        CUDA_ERROR_FILE_NOT_FOUND = 301,
        /// <summary>A link to a shared object failed to resolve.</summary>
        CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
        /// <summary>Initialization of a shared object failed.</summary>
        CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
        /// <summary>An operating-system call failed.</summary>
        CUDA_ERROR_OPERATING_SYSTEM = 304,
        /// <summary>A resource handle passed to the API call was not valid.</summary>
        CUDA_ERROR_INVALID_HANDLE = 400,
        /// <summary>A resource required by the API call is not in a valid state to perform the requested operation.</summary>
        CUDA_ERROR_ILLEGAL_STATE = 401,
        /// <summary>An attempt was made to introspect an object in a way that would discard semantically important information.</summary>
        CUDA_ERROR_LOSSY_QUERY = 402,
        /// <summary>A named symbol was not found.</summary>
        CUDA_ERROR_NOT_FOUND = 500,
        /// <summary>Previously issued asynchronous operations have not completed yet.</summary>
        CUDA_ERROR_NOT_READY = 600,
        /// <summary>While executing a kernel, the device encountered a load or store instruction on an invalid memory address.</summary>
        CUDA_ERROR_ILLEGAL_ADDRESS = 700,
        /// <summary>The launch did not occur because it did not have appropriate resources.</summary>
        CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
        /// <summary>The device kernel took too long to execute.</summary>
        CUDA_ERROR_LAUNCH_TIMEOUT = 702,
        /// <summary>The kernel launch uses an incompatible texturing mode.</summary>
        CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
        /// <summary>cuCtxEnablePeerAccess() is trying to re-enable peer access to a context where it is already enabled.</summary>
        CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
        /// <summary>cuCtxDisablePeerAccess() is trying to disable peer access which has not been enabled.</summary>
        CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
        /// <summary>The primary context for the specified device has already been initialized.</summary>
        CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
        /// <summary>The context current to the calling thread has been destroyed or is an uninitialized primary context.</summary>
        CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
        /// <summary>A device-side assert was triggered during kernel execution.</summary>
        CUDA_ERROR_ASSERT = 710,
        /// <summary>The hardware resources required to enable peer access have been exhausted.</summary>
        CUDA_ERROR_TOO_MANY_PEERS = 711,
        /// <summary>The memory range passed to cuMemHostRegister() has already been registered.</summary>
        CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
        /// <summary>The pointer passed to cuMemHostUnregister() does not correspond to a registered memory region.</summary>
        CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
        /// <summary>While executing a kernel, the device encountered a stack error.</summary>
        CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
        /// <summary>While executing a kernel, the device encountered an illegal instruction.</summary>
        CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
        /// <summary>While executing a kernel, the device encountered a load or store instruction on an unaligned memory address.</summary>
        CUDA_ERROR_MISALIGNED_ADDRESS = 716,
        /// <summary>While executing a kernel, the device encountered an instruction operating on an invalid address space.</summary>
        CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
        /// <summary>The device program counter wrapped its address space.</summary>
        CUDA_ERROR_INVALID_PC = 718,
        /// <summary>An exception occurred on the device while executing a kernel.</summary>
        CUDA_ERROR_LAUNCH_FAILED = 719,
        /// <summary>The number of blocks launched per grid for a cooperative kernel exceeds the permitted maximum.</summary>
        CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
        /// <summary>An exception occurred on the device while exiting a kernel using tensor memory because tensor memory was not completely deallocated.</summary>
        CUDA_ERROR_TENSOR_MEMORY_LEAK = 721,
        /// <summary>The attempted operation is not permitted.</summary>
        CUDA_ERROR_NOT_PERMITTED = 800,
        /// <summary>The attempted operation is not supported on the current system or device.</summary>
        CUDA_ERROR_NOT_SUPPORTED = 801,
        /// <summary>The system is not yet ready to start any CUDA work.</summary>
        CUDA_ERROR_SYSTEM_NOT_READY = 802,
        /// <summary>There is a mismatch between the display-driver and CUDA-driver versions.</summary>
        CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803,
        /// <summary>The visible hardware does not support the configured forward-compatibility mode.</summary>
        CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,
        /// <summary>The MPS client failed to connect to the MPS control daemon or MPS server.</summary>
        CUDA_ERROR_MPS_CONNECTION_FAILED = 805,
        /// <summary>The remote procedural call between the MPS server and MPS client failed.</summary>
        CUDA_ERROR_MPS_RPC_FAILURE = 806,
        /// <summary>The MPS server is not ready to accept new MPS client requests.</summary>
        CUDA_ERROR_MPS_SERVER_NOT_READY = 807,
        /// <summary>The hardware resources required to create an MPS client have been exhausted.</summary>
        CUDA_ERROR_MPS_MAX_CLIENTS_REACHED = 808,
        /// <summary>The hardware resources required to support device connections have been exhausted.</summary>
        CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED = 809,
        /// <summary>The MPS client has been terminated by the server.</summary>
        CUDA_ERROR_MPS_CLIENT_TERMINATED = 810,
        /// <summary>The module uses CUDA Dynamic Parallelism, but the current configuration does not support it.</summary>
        CUDA_ERROR_CDP_NOT_SUPPORTED = 811,
        /// <summary>The module contains an unsupported interaction between different versions of CUDA Dynamic Parallelism.</summary>
        CUDA_ERROR_CDP_VERSION_MISMATCH = 812,
        /// <summary>The operation is not permitted when the stream is capturing.</summary>
        CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900,
        /// <summary>The current capture sequence on the stream was invalidated due to a previous error.</summary>
        CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901,
        /// <summary>The operation would have resulted in a merge of two independent capture sequences.</summary>
        CUDA_ERROR_STREAM_CAPTURE_MERGE = 902,
        /// <summary>The capture was not initiated in this stream.</summary>
        CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903,
        /// <summary>The capture sequence contains a fork that was not joined to the primary stream.</summary>
        CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904,
        /// <summary>A dependency would have been created across the capture sequence boundary.</summary>
        CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905,
        /// <summary>A disallowed implicit dependency on a current capture sequence from cudaStreamLegacy was detected.</summary>
        CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906,
        /// <summary>The operation is not permitted on an event last recorded in a capturing stream.</summary>
        CUDA_ERROR_CAPTURED_EVENT = 907,
        /// <summary>A stream capture sequence not initiated with relaxed capture mode was passed to cuStreamEndCapture in a different thread.</summary>
        CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908,
        /// <summary>The timeout specified for the wait operation has elapsed.</summary>
        CUDA_ERROR_TIMEOUT = 909,
        /// <summary>The graph update was not performed because it violated constraints specific to instantiated graph update.</summary>
        CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910,
        /// <summary>An error occurred in a device outside of the GPU.</summary>
        CUDA_ERROR_EXTERNAL_DEVICE = 911,
        /// <summary>A kernel launch failed due to cluster misconfiguration.</summary>
        CUDA_ERROR_INVALID_CLUSTER_SIZE = 912,
        /// <summary>A function handle is not loaded when calling an API that requires a loaded function.</summary>
        CUDA_ERROR_FUNCTION_NOT_LOADED = 913,
        /// <summary>One or more resources passed to the operation are not valid resource types.</summary>
        CUDA_ERROR_INVALID_RESOURCE_TYPE = 914,
        /// <summary>One or more resources are insufficient or non-applicable for the operation.</summary>
        CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION = 915,
        /// <summary>An error occurred during the key rotation sequence.</summary>
        CUDA_ERROR_KEY_ROTATION = 916,
        /// <summary>The requested operation is not permitted because the stream is in a detached state.</summary>
        CUDA_ERROR_STREAM_DETACHED = 917,
        /// <summary>Graph recapture failed and had to be terminated.</summary>
        CUDA_ERROR_GRAPH_RECAPTURE_FAILURE = 918,
        /// <summary>An unknown internal error has occurred.</summary>
        CUDA_ERROR_UNKNOWN = 999,
    }

    /// <summary>CUDA function attributes.</summary>
    public enum CUfunction_attribute
    {
        CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
        CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
        CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,
        CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
        CU_FUNC_ATTRIBUTE_NUM_REGS = 4,
        CU_FUNC_ATTRIBUTE_PTX_VERSION = 5,
        CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6,
        CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
        CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
        CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET = 10,
        CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH = 11,
        CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT = 12,
        CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH = 13,
        CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED = 14,
        CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 15,
    }

    /// <summary>Result of querying a CUDA driver entry point.</summary>
    public enum CUdriverProcAddressQueryResult
    {
        CU_GET_PROC_ADDRESS_SUCCESS = 0,
        CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND = 1,
        CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT = 2,
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

    public enum CUstreamCaptureMode
    {
        CU_STREAM_CAPTURE_MODE_GLOBAL = 0,
        CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1,
        CU_STREAM_CAPTURE_MODE_RELAXED = 2,
    }

    /// <summary>
    /// CUDA JIT compiler and linker options.
    /// </summary>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g5527fa8030d5cabedc781a04dbd1997d"/>
    public enum CUjit_option
    {
        CU_JIT_MAX_REGISTERS = 0,
        CU_JIT_THREADS_PER_BLOCK = 1,
        CU_JIT_WALL_TIME = 2,
        CU_JIT_INFO_LOG_BUFFER = 3,
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
        CU_JIT_ERROR_LOG_BUFFER = 5,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
        CU_JIT_OPTIMIZATION_LEVEL = 7,
        CU_JIT_TARGET_FROM_CUCONTEXT = 8,
        CU_JIT_TARGET = 9,
    }

    /// <summary>
    /// CUDA JIT linker input types.
    /// </summary>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc78e5cb421c428676861189048888958"/>
    public enum CUjitInputType
    {
        CU_JIT_INPUT_CUBIN = 0,
        CU_JIT_INPUT_PTX = 1,
        CU_JIT_INPUT_FATBINARY = 2,
        CU_JIT_INPUT_OBJECT = 3,
        CU_JIT_INPUT_LIBRARY = 4,
        CU_JIT_INPUT_NVVM = 5,
    }

    /// <summary>
    /// CUDA library loading options.
    /// </summary>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g1ac4f8471f550201cab6f17a49416989"/>
    public enum CUlibraryOption
    {
        CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE = 0,
        CU_LIBRARY_BINARY_IS_PRESERVED = 1,
        CU_LIBRARY_NUM_OPTIONS = 2,
    }

    /// <summary>
    /// Flags for instantiating a graph.
    /// </summary>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g070bf5517d3a7915667c256eefce4956"/>
    public enum CUgraphInstantiate_flags : ulong
    {
        CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH = 1,
        CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD = 2,
        CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH = 4,
        CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY = 8,
    }

    /// <summary>
    /// Graph instantiation results.
    /// </summary>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g863484740f7d9f82c908d228f791cc56"/>
    public enum CUgraphInstantiateResult
    {
        CUDA_GRAPH_INSTANTIATE_SUCCESS = 0,
        CUDA_GRAPH_INSTANTIATE_ERROR = 1,
        CUDA_GRAPH_INSTANTIATE_INVALID_STRUCTURE = 2,
        CUDA_GRAPH_INSTANTIATE_NODE_OPERATION_NOT_SUPPORTED = 3,
        CUDA_GRAPH_INSTANTIATE_MULTIPLE_CTXS_NOT_SUPPORTED = 4,
        CUDA_GRAPH_INSTANTIATE_CONDITIONAL_HANDLE_UNUSED = 5,
    }

    /// <summary>
    /// Graph instantiation parameters.
    /// </summary>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__GRAPH__INSTANTIATE__PARAMS.html"/>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_GRAPH_INSTANTIATE_PARAMS
    {
        public ulong flags;
        public CUstream hUploadStream;
        public CUgraphNode hErrNode_out;
        public CUgraphInstantiateResult result_out;
    }

    public enum CUaccessProperty
    {
        CU_ACCESS_PROPERTY_NORMAL = 0,
        CU_ACCESS_PROPERTY_STREAMING = 1,
        CU_ACCESS_PROPERTY_PERSISTING = 2,
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUaccessPolicyWindow
    {
        public IntPtr base_ptr;
        public nuint num_bytes;
        public float hitRatio;
        public CUaccessProperty hitProp;
        public CUaccessProperty missProp;
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
        public unsafe fixed int reserved[32];
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

    public enum CUsynchronizationPolicy
    {
        CU_SYNC_POLICY_AUTO = 1,
        CU_SYNC_POLICY_SPIN = 2,
        CU_SYNC_POLICY_YIELD = 3,
        CU_SYNC_POLICY_BLOCKING_SYNC = 4,
    }

    public enum CUclusterSchedulingPolicy
    {
        CU_CLUSTER_SCHEDULING_POLICY_DEFAULT = 0,
        CU_CLUSTER_SCHEDULING_POLICY_SPREAD = 1,
        CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING = 2,
    }

    public enum CUlaunchMemSyncDomain
    {
        CU_LAUNCH_MEM_SYNC_DOMAIN_DEFAULT = 0,
        CU_LAUNCH_MEM_SYNC_DOMAIN_REMOTE = 1,
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUlaunchMemSyncDomainMap
    {
        public byte default_;
        public byte remote;
    }

    public enum CUlaunchAttributeID
    {
        CU_LAUNCH_ATTRIBUTE_IGNORE = 0,
        CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1,
        CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 2,
        CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY = 3,
        CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4,
        CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 5,
        CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6,
        CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT = 7,
        CU_LAUNCH_ATTRIBUTE_PRIORITY = 8,
        CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP = 9,
        CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN = 10,
        CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION = 11,
        CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT = 12,
        CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE = 13,
        CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 14,
        CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING = 16,
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUlaunchAttributeDim3
    {
        public uint x;
        public uint y;
        public uint z;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUlaunchAttributeProgrammaticEvent
    {
        public CUevent event_;
        public int flags;
        public int triggerAtBlockStart;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUlaunchAttributeLaunchCompletionEvent
    {
        public CUevent event_;
        public int flags;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUlaunchAttributeDeviceUpdatableKernelNode
    {
        public int deviceUpdatable;
        public CUgraphDeviceNode devNode;
    }

    [StructLayout(LayoutKind.Explicit, Size = 64)]
    public struct CUlaunchAttributeValue
    {
        [FieldOffset(0)] public CUaccessPolicyWindow accessPolicyWindow;
        [FieldOffset(0)] public int cooperative;
        [FieldOffset(0)] public CUsynchronizationPolicy syncPolicy;
        [FieldOffset(0)] public CUlaunchAttributeDim3 clusterDim;
        [FieldOffset(0)] public CUclusterSchedulingPolicy clusterSchedulingPolicyPreference;
        [FieldOffset(0)] public int programmaticStreamSerializationAllowed;
        [FieldOffset(0)] public CUlaunchAttributeProgrammaticEvent programmaticEvent;
        [FieldOffset(0)] public int priority;
        [FieldOffset(0)] public CUlaunchMemSyncDomainMap memSyncDomainMap;
        [FieldOffset(0)] public CUlaunchMemSyncDomain memSyncDomain;
        [FieldOffset(0)] public CUlaunchAttributeDim3 preferredClusterDim;
        [FieldOffset(0)] public CUlaunchAttributeLaunchCompletionEvent launchCompletionEvent;
        [FieldOffset(0)] public CUlaunchAttributeDeviceUpdatableKernelNode deviceUpdatableKernelNode;
        [FieldOffset(0)] public uint sharedMemCarveout;
        [FieldOffset(0)] public uint nvlinkUtilCentricScheduling;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUlaunchAttribute
    {
        public CUlaunchAttributeID id;
        public int pad0;
        public int pad1;
        public CUlaunchAttributeValue value;
    }

    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct CUlaunchConfig
    {
        public uint gridDimX;
        public uint gridDimY;
        public uint gridDimZ;
        public uint blockDimX;
        public uint blockDimY;
        public uint blockDimZ;
        public uint sharedMemBytes;
        public CUstream hStream;
        public CUlaunchAttribute* attrs;
        public uint numAttrs;
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
