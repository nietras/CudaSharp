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
        /// <summary>Max number of threads per block.</summary>
        CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
        /// <summary>Size of shared memory used by the kernel in bytes.</summary>
        CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
        /// <summary>Size of constant memory used by the kernel in bytes.</summary>
        CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,
        /// <summary>Size of local memory used by the kernel in bytes.</summary>
        CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
        /// <summary>Number of registers used by the kernel.</summary>
        CU_FUNC_ATTRIBUTE_NUM_REGS = 4,
        /// <summary>PTX version used to compile the kernel.</summary>
        CU_FUNC_ATTRIBUTE_PTX_VERSION = 5,
        /// <summary>Binary version of the kernel.</summary>
        CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6,
        /// <summary>Cache mode of the kernel.</summary>
        CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,
        /// <summary>Max size of dynamically allocated shared memory in bytes.</summary>
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
        /// <summary>Preferred shared memory carveout for the kernel.</summary>
        CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
        /// <summary>Cluster size must be set for the kernel.</summary>
        CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET = 10,
        /// <summary>Required cluster width for the kernel.</summary>
        CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH = 11,
        /// <summary>Required cluster height for the kernel.</summary>
        CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT = 12,
        /// <summary>Required cluster depth for the kernel.</summary>
        CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH = 13,
        /// <summary>Non-portable cluster size is allowed for the kernel.</summary>
        CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED = 14,
        /// <summary>Cluster scheduling policy preference for the kernel.</summary>
        CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 15,
    }

    /// <summary>Result of querying a CUDA driver entry point.</summary>
    public enum CUdriverProcAddressQueryResult
    {
        /// <summary>The symbol was found and the query was successful.</summary>
        CU_GET_PROC_ADDRESS_SUCCESS = 0,
        /// <summary>The symbol was not found.</summary>
        CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND = 1,
        /// <summary>The driver version is not sufficient for the requested version.</summary>
        CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT = 2,
    }

    /// <summary>Event creation flags.</summary>
    public enum CUevent_flags
    {
        /// <summary>No flags, default behavior.</summary>
        CU_EVENT_DEFAULT = 0x0,
        /// <summary>Create an event that blocks on completion.</summary>
        CU_EVENT_BLOCKING_SYNC = 0x1,
        /// <summary>Create an event that disables timing.</summary>
        CU_EVENT_DISABLE_TIMING = 0x2,
        /// <summary>Create an interprocess event.</summary>
        CU_EVENT_INTERPROCESS = 0x4,
    }

    /// <summary>Context creation flags.</summary>
    public enum CUctx_flags
    {
        /// <summary>Schedule automatically.</summary>
        CU_CTX_SCHED_AUTO = 0x00,
        /// <summary>Spin wait for other tasks to complete.</summary>
        CU_CTX_SCHED_SPIN = 0x01,
        /// <summary>Yield to other tasks.</summary>
        CU_CTX_SCHED_YIELD = 0x02,
        /// <summary>Use blocking sync.</summary>
        CU_CTX_SCHED_BLOCKING_SYNC = 0x04,
        /// <summary>Map host memory.</summary>
        CU_CTX_MAP_HOST = 0x08,
        /// <summary>Resize local memory to maximum.</summary>
        CU_CTX_LMEM_RESIZE_TO_MAX = 0x10,
    }

    /// <summary>Host memory allocation flags.</summary>
    public enum CUmemhostalloc_flags
    {
        /// <summary>Memory is portable across devices.</summary>
        CU_MEMHOSTALLOC_PORTABLE = 0x01,
        /// <summary>Memory can be mapped to CUDA device.</summary>
        CU_MEMHOSTALLOC_DEVICEMAP = 0x02,
        /// <summary>Memory is write-combined.</summary>
        CU_MEMHOSTALLOC_WRITECOMBINED = 0x04,
    }

    /// <summary>Stream capture modes.</summary>
    public enum CUstreamCaptureMode
    {
        /// <summary>Global stream capture mode.</summary>
        CU_STREAM_CAPTURE_MODE_GLOBAL = 0,
        /// <summary>Thread-local stream capture mode.</summary>
        CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1,
        /// <summary>Relaxed stream capture mode.</summary>
        CU_STREAM_CAPTURE_MODE_RELAXED = 2,
    }

    /// <summary>
    /// CUDA JIT compiler and linker options.
    /// </summary>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g5527fa8030d5cabedc781a04dbd1997d"/>
    public enum CUjit_option
    {
        /// <summary>Max registers per thread.</summary>
        CU_JIT_MAX_REGISTERS = 0,
        /// <summary>Threads per block.</summary>
        CU_JIT_THREADS_PER_BLOCK = 1,
        /// <summary>Wall time for the JIT compilation.</summary>
        CU_JIT_WALL_TIME = 2,
        /// <summary>Buffer for the info log.</summary>
        CU_JIT_INFO_LOG_BUFFER = 3,
        /// <summary>Size of the info log buffer.</summary>
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
        /// <summary>Buffer for the error log.</summary>
        CU_JIT_ERROR_LOG_BUFFER = 5,
        /// <summary>Size of the error log buffer.</summary>
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
        /// <summary>Optimization level.</summary>
        CU_JIT_OPTIMIZATION_LEVEL = 7,
        /// <summary>Target from CUDA context.</summary>
        CU_JIT_TARGET_FROM_CUCONTEXT = 8,
        /// <summary>Target GPU architecture.</summary>
        CU_JIT_TARGET = 9,
    }

    /// <summary>
    /// CUDA JIT linker input types.
    /// </summary>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc78e5cb421c428676861189048888958"/>
    public enum CUjitInputType
    {
        /// <summary>CUBIN input.</summary>
        CU_JIT_INPUT_CUBIN = 0,
        /// <summary>PTX input.</summary>
        CU_JIT_INPUT_PTX = 1,
        /// <summary>Fat binary input.</summary>
        CU_JIT_INPUT_FATBINARY = 2,
        /// <summary>Object file input.</summary>
        CU_JIT_INPUT_OBJECT = 3,
        /// <summary>Library file input.</summary>
        CU_JIT_INPUT_LIBRARY = 4,
        /// <summary>NVVM input.</summary>
        CU_JIT_INPUT_NVVM = 5,
    }

    /// <summary>
    /// CUDA library loading options.
    /// </summary>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g1ac4f8471f550201cab6f17a49416989"/>
    public enum CUlibraryOption
    {
        /// <summary>Load host universal function and data table.</summary>
        CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE = 0,
        /// <summary>Preserve library binary.</summary>
        CU_LIBRARY_BINARY_IS_PRESERVED = 1,
        /// <summary>Number of options.</summary>
        CU_LIBRARY_NUM_OPTIONS = 2,
    }

    /// <summary>
    /// Flags for instantiating a graph.
    /// </summary>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g070bf5517d3a7915667c256eefce4956"/>
    public enum CUgraphInstantiate_flags : ulong
    {
        /// <summary>Automatically free resources on launch.</summary>
        CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH = 1,
        /// <summary>Upload resources for the graph.</summary>
        CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD = 2,
        /// <summary>Launch the graph on the device.</summary>
        CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH = 4,
        /// <summary>Use node priority for scheduling.</summary>
        CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY = 8,
    }

    /// <summary>
    /// Graph instantiation results.
    /// </summary>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g863484740f7d9f82c908d228f791cc56"/>
    public enum CUgraphInstantiateResult
    {
        /// <summary>Instantiation was successful.</summary>
        CUDA_GRAPH_INSTANTIATE_SUCCESS = 0,
        /// <summary>Instantiation encountered an error.</summary>
        CUDA_GRAPH_INSTANTIATE_ERROR = 1,
        /// <summary>Invalid structure for instantiation.</summary>
        CUDA_GRAPH_INSTANTIATE_INVALID_STRUCTURE = 2,
        /// <summary>Node operation not supported.</summary>
        CUDA_GRAPH_INSTANTIATE_NODE_OPERATION_NOT_SUPPORTED = 3,
        /// <summary>Multiple contexts not supported.</summary>
        CUDA_GRAPH_INSTANTIATE_MULTIPLE_CTXS_NOT_SUPPORTED = 4,
        /// <summary>Conditional handle unused.</summary>
        CUDA_GRAPH_INSTANTIATE_CONDITIONAL_HANDLE_UNUSED = 5,
    }

    /// <summary>
    /// Graph instantiation parameters.
    /// </summary>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__GRAPH__INSTANTIATE__PARAMS.html"/>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_GRAPH_INSTANTIATE_PARAMS
    {
        /// <summary>Flags for instantiation.</summary>
        public ulong flags;
        /// <summary>Stream for uploading resources.</summary>
        public CUstream hUploadStream;
        /// <summary>Node where an error occurred.</summary>
        public CUgraphNode hErrNode_out;
        /// <summary>Result of the instantiation.</summary>
        public CUgraphInstantiateResult result_out;
    }

    /// <summary>Access properties for memory.</summary>
    public enum CUaccessProperty
    {
        /// <summary>Normal access.</summary>
        CU_ACCESS_PROPERTY_NORMAL = 0,
        /// <summary>Streaming access.</summary>
        CU_ACCESS_PROPERTY_STREAMING = 1,
        /// <summary>Persisting access.</summary>
        CU_ACCESS_PROPERTY_PERSISTING = 2,
    }

    /// <summary>Memory access policy window for a stream or kernel launch.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUaccessPolicyWindow
    {
        public IntPtr base_ptr;
        public nuint num_bytes;
        public float hitRatio;
        public CUaccessProperty hitProp;
        public CUaccessProperty missProp;
    }

    /// <summary>Legacy CUDA device properties.</summary>
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

    /// <summary>CUDA device attributes.</summary>
    public enum CUdevice_attribute
    {
        /// <summary>Max threads per block.</summary>
        CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
        /// <summary>Max block dimension X.</summary>
        CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
        /// <summary>Max block dimension Y.</summary>
        CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
        /// <summary>Max block dimension Z.</summary>
        CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
        /// <summary>Max grid dimension X.</summary>
        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
        /// <summary>Max grid dimension Y.</summary>
        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
        /// <summary>Max grid dimension Z.</summary>
        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
        /// <summary>Max shared memory per block.</summary>
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
        /// <summary>Total constant memory.</summary>
        CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
        /// <summary>Warp size.</summary>
        CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
        /// <summary>Max pitch.</summary>
        CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
        /// <summary>Max registers per block.</summary>
        CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
        /// <summary>Clock rate.</summary>
        CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
        /// <summary>Texture alignment.</summary>
        CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
        /// <summary>GPU overlap.</summary>
        CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
        /// <summary>Multiprocessor count.</summary>
        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
        /// <summary>Kernel execution timeout.</summary>
        CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
        /// <summary>Integrated GPU.</summary>
        CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
        /// <summary>Can map host memory.</summary>
        CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
        /// <summary>Compute mode.</summary>
        CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
        /// <summary>Major compute capability version number.</summary>
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
        /// <summary>Minor compute capability version number.</summary>
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    }

    /// <summary>Universally unique identifier for a CUDA device.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUuuid
    {
        public unsafe fixed byte bytes[16];
    }

    /// <summary>Texture address modes.</summary>
    public enum CUaddress_mode
    {
        /// <summary>Wrap texture coordinates.</summary>
        CU_TR_ADDRESS_MODE_WRAP = 0,
        /// <summary>Clamp texture coordinates.</summary>
        CU_TR_ADDRESS_MODE_CLAMP = 1,
        /// <summary>Mirror texture coordinates.</summary>
        CU_TR_ADDRESS_MODE_MIRROR = 2,
        /// <summary>Border texture coordinates.</summary>
        CU_TR_ADDRESS_MODE_BORDER = 3,
    }

    /// <summary>Texture filter modes.</summary>
    public enum CUfilter_mode
    {
        /// <summary>Point sampling.</summary>
        CU_TR_FILTER_MODE_POINT = 0,
        /// <summary>Linear filtering.</summary>
        CU_TR_FILTER_MODE_LINEAR = 1,
    }

    /// <summary>CUDA array formats.</summary>
    public enum CUarray_format
    {
        /// <summary>Unsigned 8-bit integer.</summary>
        CU_AD_FORMAT_UNSIGNED_INT8 = 0x01,
        /// <summary>Unsigned 16-bit integer.</summary>
        CU_AD_FORMAT_UNSIGNED_INT16 = 0x02,
        /// <summary>Unsigned 32-bit integer.</summary>
        CU_AD_FORMAT_UNSIGNED_INT32 = 0x03,
        /// <summary>Signed 8-bit integer.</summary>
        CU_AD_FORMAT_SIGNED_INT8 = 0x08,
        /// <summary>Signed 16-bit integer.</summary>
        CU_AD_FORMAT_SIGNED_INT16 = 0x09,
        /// <summary>Signed 32-bit integer.</summary>
        CU_AD_FORMAT_SIGNED_INT32 = 0x0a,
        /// <summary>Half precision floating point.</summary>
        CU_AD_FORMAT_HALF = 0x10,
        /// <summary>Single precision floating point.</summary>
        CU_AD_FORMAT_FLOAT = 0x20,
    }

    /// <summary>CUDA array descriptor.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_ARRAY_DESCRIPTOR
    {
        /// <summary>Width of the array.</summary>
        public nuint Width;
        /// <summary>Height of the array.</summary>
        public nuint Height;
        /// <summary>Format of the array elements.</summary>
        public CUarray_format Format;
        /// <summary>Number of channels per element.</summary>
        public uint NumChannels;
    }

    /// <summary>CUDA resource descriptor.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_RESOURCE_DESC
    {
        /// <summary>Type of the resource.</summary>
        public CUresourcetype resType;
        /// <summary>Union of resource descriptors.</summary>
        public CUDA_RESOURCE_DESC_UNION res;
        /// <summary>Flags for the resource.</summary>
        public uint flags;
    }

    /// <summary>Union of CUDA resource descriptors.</summary>
    [StructLayout(LayoutKind.Explicit)]
    public struct CUDA_RESOURCE_DESC_UNION
    {
        /// <summary>Array resource descriptor.</summary>
        [FieldOffset(0)]
        public CUDA_RESOURCE_DESC_ARRAY array;
        /// <summary>Mipmapped array resource descriptor.</summary>
        [FieldOffset(0)]
        public CUDA_RESOURCE_DESC_MIPMAPPED_ARRAY mipmap;
        /// <summary>Linear resource descriptor.</summary>
        [FieldOffset(0)]
        public CUDA_RESOURCE_DESC_LINEAR linear;
        /// <summary>2D pitched resource descriptor.</summary>
        [FieldOffset(0)]
        public CUDA_RESOURCE_DESC_PITCH2D pitch2D;
        [FieldOffset(0)]
        public unsafe fixed int reserved[32];
    }

    /// <summary>CUDA array resource descriptor.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_RESOURCE_DESC_ARRAY
    {
        /// <summary>Handle to the array.</summary>
        public CUarray hArray;
    }

    /// <summary>CUDA mipmapped array resource descriptor.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_RESOURCE_DESC_MIPMAPPED_ARRAY
    {
        /// <summary>Handle to the mipmapped array.</summary>
        public IntPtr hMipmappedArray;
    }

    /// <summary>CUDA linear resource descriptor.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_RESOURCE_DESC_LINEAR
    {
        /// <summary>Device pointer to the memory.</summary>
        public CUdeviceptr devPtr;
        /// <summary>Format of the data.</summary>
        public CUarray_format format;
        /// <summary>Number of channels in the data.</summary>
        public uint numChannels;
        /// <summary>Size of the resource in bytes.</summary>
        public nuint sizeInBytes;
    }

    /// <summary>CUDA 2D pitched resource descriptor.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_RESOURCE_DESC_PITCH2D
    {
        /// <summary>Device pointer to the memory.</summary>
        public CUdeviceptr devPtr;
        /// <summary>Format of the data.</summary>
        public CUarray_format format;
        /// <summary>Number of channels in the data.</summary>
        public uint numChannels;
        /// <summary>Width of the resource.</summary>
        public nuint width;
        /// <summary>Height of the resource.</summary>
        public nuint height;
        /// <summary>Pitch of the resource in bytes.</summary>
        public nuint pitchInBytes;
    }

    /// <summary>CUDA texture descriptor.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_TEXTURE_DESC
    {
        /// <summary>Texture address mode for dimension 0.</summary>
        public CUaddress_mode addressMode0;
        /// <summary>Texture address mode for dimension 1.</summary>
        public CUaddress_mode addressMode1;
        /// <summary>Texture address mode for dimension 2.</summary>
        public CUaddress_mode addressMode2;
        /// <summary>Texture filter mode.</summary>
        public CUfilter_mode filterMode;
        /// <summary>Flags for texture creation.</summary>
        public uint flags;
        /// <summary>Max anisotropy for the texture.</summary>
        public uint maxAnisotropy;
        /// <summary>Mipmap filter mode.</summary>
        public CUfilter_mode mipmapFilterMode;
        /// <summary>Mipmap level bias.</summary>
        public float mipmapLevelBias;
        /// <summary>Minimum mipmap level clamp.</summary>
        public float minMipmapLevelClamp;
        /// <summary>Maximum mipmap level clamp.</summary>
        public float maxMipmapLevelClamp;
        /// <summary>Border color for the texture.</summary>
        public float borderColor0;
        /// <summary>Border color for the texture.</summary>
        public float borderColor1;
        /// <summary>Border color for the texture.</summary>
        public float borderColor2;
        /// <summary>Border color for the texture.</summary>
        public float borderColor3;
        public unsafe fixed int reserved[15];
    }

    /// <summary>CUDA resource view descriptor.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_RESOURCE_VIEW_DESC
    {
        /// <summary>Format of the resource view.</summary>
        public CUresourceViewFormat format;
        /// <summary>Width of the resource view.</summary>
        public nuint width;
        /// <summary>Height of the resource view.</summary>
        public nuint height;
        /// <summary>Depth of the resource view.</summary>
        public nuint depth;
        /// <summary>First mipmap level.</summary>
        public uint firstMipmapLevel;
        /// <summary>Last mipmap level.</summary>
        public uint lastMipmapLevel;
        /// <summary>First layer.</summary>
        public uint firstLayer;
        /// <summary>Last layer.</summary>
        public uint lastLayer;
        public unsafe fixed uint reserved[16];
    }

    /// <summary>CUDA resource types.</summary>
    public enum CUresourcetype
    {
        /// <summary>Array resource.</summary>
        CU_RESOURCE_TYPE_ARRAY = 0x00,
        /// <summary>Mipmapped array resource.</summary>
        CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 0x01,
        /// <summary>Linear resource.</summary>
        CU_RESOURCE_TYPE_LINEAR = 0x02,
        /// <summary>2D pitched resource.</summary>
        CU_RESOURCE_TYPE_PITCH2D = 0x03,
    }

    /// <summary>CUDA resource view formats.</summary>
    public enum CUresourceViewFormat
    {
        /// <summary>No format.</summary>
        CU_RESVIEWFORMAT_NONE = 0x00,
        /// <summary>Unsigned 1-channel view.</summary>
        CU_RESVIEWFORMAT_UINT_1CHANNEL = 0x01,
        /// <summary>Unsigned 2-channel view.</summary>
        CU_RESVIEWFORMAT_UINT_2CHANNEL = 0x02,
        /// <summary>Unsigned 4-channel view.</summary>
        CU_RESVIEWFORMAT_UINT_4CHANNEL = 0x03,
        /// <summary>Signed 1-channel view.</summary>
        CU_RESVIEWFORMAT_SINT_1CHANNEL = 0x04,
        /// <summary>Signed 2-channel view.</summary>
        CU_RESVIEWFORMAT_SINT_2CHANNEL = 0x05,
        /// <summary>Signed 4-channel view.</summary>
        CU_RESVIEWFORMAT_SINT_4CHANNEL = 0x06,
        /// <summary>Float 1-channel view.</summary>
        CU_RESVIEWFORMAT_FLOAT_1CHANNEL = 0x07,
        /// <summary>Float 2-channel view.</summary>
        CU_RESVIEWFORMAT_FLOAT_2CHANNEL = 0x08,
        /// <summary>Float 4-channel view.</summary>
        CU_RESVIEWFORMAT_FLOAT_4CHANNEL = 0x09,
        /// <summary>Half 1-channel view.</summary>
        CU_RESVIEWFORMAT_HALF_1CHANNEL = 0x0a,
        /// <summary>Half 2-channel view.</summary>
        CU_RESVIEWFORMAT_HALF_2CHANNEL = 0x0b,
        /// <summary>Half 4-channel view.</summary>
        CU_RESVIEWFORMAT_HALF_4CHANNEL = 0x0c,
    }

    /// <summary>CUDA limits.</summary>
    public enum CUlimit
    {
        /// <summary>Stack size limit.</summary>
        CU_LIMIT_STACK_SIZE = 0x00,
        /// <summary>Print FIFO size limit.</summary>
        CU_LIMIT_PRINTF_FIFO_SIZE = 0x01,
        /// <summary>Malloc heap size limit.</summary>
        CU_LIMIT_MALLOC_HEAP_SIZE = 0x02,
        /// <summary>Device runtime sync depth limit.</summary>
        CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 0x03,
        /// <summary>Device runtime pending launch count limit.</summary>
        CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 0x04,
        /// <summary>Max L2 fetch granularity limit.</summary>
        CU_LIMIT_MAX_L2_FETCH_GRANULARITY = 0x05,
        /// <summary>Persistent L2 cache size limit.</summary>
        CU_LIMIT_PERSISTENT_L2_CACHE_SIZE = 0x06,
    }

    /// <summary>CUDA function cache preferences.</summary>
    public enum CUfunc_cache
    {
        /// <summary>No preference.</summary>
        CU_FUNC_CACHE_PREFER_NONE = 0x00,
        /// <summary>Prefer shared memory.</summary>
        CU_FUNC_CACHE_PREFER_SHARED = 0x01,
        /// <summary>Prefer L1 cache.</summary>
        CU_FUNC_CACHE_PREFER_L1 = 0x02,
        /// <summary>Equal preference.</summary>
        CU_FUNC_CACHE_PREFER_EQUAL = 0x03,
    }

    /// <summary>CUDA shared memory configurations.</summary>
    public enum CUsharedconfig
    {
        /// <summary>Default bank size for shared memory.</summary>
        CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = 0x00,
        /// <summary>Four-byte bank size for shared memory.</summary>
        CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = 0x01,
        /// <summary>Eight-byte bank size for shared memory.</summary>
        CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = 0x02,
    }

    /// <summary>CUDA profiler output modes.</summary>
    public enum CUprofiler_outputMode
    {
        /// <summary>Key-value pair output.</summary>
        CU_OUT_KEY_VALUE_PAIR = 0x00,
        /// <summary>CSV output.</summary>
        CU_OUT_CSV = 0x01,
    }

    /// <summary>2D memory copy parameters.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_MEMCPY2D
    {
        /// <summary>Source X offset in bytes.</summary>
        public nuint srcXInBytes;
        /// <summary>Source Y offset.</summary>
        public nuint srcY;
        /// <summary>Type of source memory.</summary>
        public CUmemorytype srcMemoryType;
        /// <summary>Host pointer to source memory.</summary>
        public IntPtr srcHost;
        /// <summary>Device pointer to source memory.</summary>
        public CUdeviceptr srcDevice;
        /// <summary>Handle to source array.</summary>
        public CUarray srcArray;
        /// <summary>Source pitch.</summary>
        public nuint srcPitch;

        /// <summary>Destination X offset in bytes.</summary>
        public nuint dstXInBytes;
        /// <summary>Destination Y offset.</summary>
        public nuint dstY;
        /// <summary>Type of destination memory.</summary>
        public CUmemorytype dstMemoryType;
        /// <summary>Host pointer to destination memory.</summary>
        public IntPtr dstHost;
        /// <summary>Device pointer to destination memory.</summary>
        public CUdeviceptr dstDevice;
        /// <summary>Handle to destination array.</summary>
        public CUarray dstArray;
        /// <summary>Destination pitch.</summary>
        public nuint dstPitch;

        /// <summary>Width of the memory region in bytes.</summary>
        public nuint WidthInBytes;
        /// <summary>Height of the memory region.</summary>
        public nuint Height;
    }

    /// <summary>3D memory copy parameters.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_MEMCPY3D
    {
        /// <summary>Source X offset in bytes.</summary>
        public nuint srcXInBytes;
        /// <summary>Source Y offset.</summary>
        public nuint srcY;
        /// <summary>Source Z offset.</summary>
        public nuint srcZ;
        /// <summary>Source level of detail.</summary>
        public nuint srcLOD;
        /// <summary>Type of source memory.</summary>
        public CUmemorytype srcMemoryType;
        /// <summary>Host pointer to source memory.</summary>
        public IntPtr srcHost;
        /// <summary>Device pointer to source memory.</summary>
        public CUdeviceptr srcDevice;
        /// <summary>Handle to source array.</summary>
        public CUarray srcArray;
        /// <summary>Context for source memory.</summary>
        public IntPtr srcContext;
        /// <summary>Source pitch.</summary>
        public nuint srcPitch;
        /// <summary>Source height.</summary>
        public nuint srcHeight;

        /// <summary>Destination X offset in bytes.</summary>
        public nuint dstXInBytes;
        /// <summary>Destination Y offset.</summary>
        public nuint dstY;
        /// <summary>Destination Z offset.</summary>
        public nuint dstZ;
        /// <summary>Destination level of detail.</summary>
        public nuint dstLOD;
        /// <summary>Type of destination memory.</summary>
        public CUmemorytype dstMemoryType;
        /// <summary>Host pointer to destination memory.</summary>
        public IntPtr dstHost;
        /// <summary>Device pointer to destination memory.</summary>
        public CUdeviceptr dstDevice;
        /// <summary>Handle to destination array.</summary>
        public CUarray dstArray;
        /// <summary>Context for destination memory.</summary>
        public IntPtr dstContext;
        /// <summary>Destination pitch.</summary>
        public nuint dstPitch;
        /// <summary>Destination height.</summary>
        public nuint dstHeight;

        /// <summary>Width of the memory region in bytes.</summary>
        public nuint WidthInBytes;
        /// <summary>Height of the memory region.</summary>
        public nuint Height;
        /// <summary>Depth of the memory region.</summary>
        public nuint Depth;
    }

    /// <summary>CUDA memory types.</summary>
    public enum CUmemorytype
    {
        /// <summary>Host memory type.</summary>
        CU_MEMORYTYPE_HOST = 0x01,
        /// <summary>Device memory type.</summary>
        CU_MEMORYTYPE_DEVICE = 0x02,
        /// <summary>Array memory type.</summary>
        CU_MEMORYTYPE_ARRAY = 0x03,
        /// <summary>Unified memory type.</summary>
        CU_MEMORYTYPE_UNIFIED = 0x04,
    }

    /// <summary>Stream synchronization policies.</summary>
    public enum CUsynchronizationPolicy
    {
        /// <summary>Automatic synchronization.</summary>
        CU_SYNC_POLICY_AUTO = 1,
        /// <summary>Spin wait for synchronization.</summary>
        CU_SYNC_POLICY_SPIN = 2,
        /// <summary>Yield for synchronization.</summary>
        CU_SYNC_POLICY_YIELD = 3,
        /// <summary>Blocking synchronization.</summary>
        CU_SYNC_POLICY_BLOCKING_SYNC = 4,
    }

    /// <summary>Cluster scheduling policies.</summary>
    public enum CUclusterSchedulingPolicy
    {
        /// <summary>Default scheduling policy.</summary>
        CU_CLUSTER_SCHEDULING_POLICY_DEFAULT = 0,
        /// <summary>Spread scheduling policy.</summary>
        CU_CLUSTER_SCHEDULING_POLICY_SPREAD = 1,
        /// <summary>Load balancing scheduling policy.</summary>
        CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING = 2,
    }

    /// <summary>Launch memory synchronization domains.</summary>
    public enum CUlaunchMemSyncDomain
    {
        /// <summary>Default synchronization domain.</summary>
        CU_LAUNCH_MEM_SYNC_DOMAIN_DEFAULT = 0,
        /// <summary>Remote synchronization domain.</summary>
        CU_LAUNCH_MEM_SYNC_DOMAIN_REMOTE = 1,
    }

    /// <summary>Maps launch memory synchronization domains to hardware domains.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUlaunchMemSyncDomainMap
    {
        public byte default_;
        public byte remote;
    }

    /// <summary>Launch attribute IDs.</summary>
    public enum CUlaunchAttributeID
    {
        /// <summary>Ignore the attribute.</summary>
        CU_LAUNCH_ATTRIBUTE_IGNORE = 0,
        /// <summary>Memory access policy window attribute.</summary>
        CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1,
        /// <summary>Cooperative launch attribute.</summary>
        CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 2,
        /// <summary>Synchronization policy attribute.</summary>
        CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY = 3,
        /// <summary>Cluster dimension attribute.</summary>
        CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4,
        /// <summary>Cluster scheduling policy preference attribute.</summary>
        CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 5,
        /// <summary>Programmatic stream serialization attribute.</summary>
        CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6,
        /// <summary>Programmatic event attribute.</summary>
        CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT = 7,
        /// <summary>Priority attribute.</summary>
        CU_LAUNCH_ATTRIBUTE_PRIORITY = 8,
        /// <summary>Memory synchronization domain map attribute.</summary>
        CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP = 9,
        /// <summary>Memory synchronization domain attribute.</summary>
        CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN = 10,
        /// <summary>Preferred cluster dimension attribute.</summary>
        CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION = 11,
        /// <summary>Launch completion event attribute.</summary>
        CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT = 12,
        /// <summary>Device updatable kernel node attribute.</summary>
        CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE = 13,
        /// <summary>Preferred shared memory carveout attribute.</summary>
        CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 14,
        /// <summary>NVLink util-centric scheduling attribute.</summary>
        CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING = 16,
    }

    /// <summary>Three-dimensional launch attribute value.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUlaunchAttributeDim3
    {
        public uint x;
        public uint y;
        public uint z;
    }

    /// <summary>Programmatic event launch attribute value.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUlaunchAttributeProgrammaticEvent
    {
        public CUevent event_;
        public int flags;
        public int triggerAtBlockStart;
    }

    /// <summary>Launch-completion event attribute value.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUlaunchAttributeLaunchCompletionEvent
    {
        public CUevent event_;
        public int flags;
    }

    /// <summary>Device-updatable graph kernel node attribute value.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUlaunchAttributeDeviceUpdatableKernelNode
    {
        public int deviceUpdatable;
        public CUgraphDeviceNode devNode;
    }

    /// <summary>Composite type for launch attribute value.</summary>
    /// <remarks>
    /// This structure is used to represent the value of a launch attribute, which can be
    /// of different types depending on the attribute ID.
    /// </remarks>
    [StructLayout(LayoutKind.Explicit, Size = 64)]
    public struct CUlaunchAttributeValue
    {
        /// <summary>Access policy window for the launch.</summary>
        [FieldOffset(0)] public CUaccessPolicyWindow accessPolicyWindow;
        /// <summary>Cooperative launch flag.</summary>
        [FieldOffset(0)] public int cooperative;
        /// <summary>Synchronization policy for the launch.</summary>
        [FieldOffset(0)] public CUsynchronizationPolicy syncPolicy;
        /// <summary>Cluster dimension for the launch.</summary>
        [FieldOffset(0)] public CUlaunchAttributeDim3 clusterDim;
        /// <summary>Cluster scheduling policy preference.</summary>
        [FieldOffset(0)] public CUclusterSchedulingPolicy clusterSchedulingPolicyPreference;
        /// <summary>Programmatic stream serialization allowed flag.</summary>
        [FieldOffset(0)] public int programmaticStreamSerializationAllowed;
        /// <summary>Programmatic event for the launch.</summary>
        [FieldOffset(0)] public CUlaunchAttributeProgrammaticEvent programmaticEvent;
        /// <summary>Priority for the launch.</summary>
        [FieldOffset(0)] public int priority;
        /// <summary>Memory synchronization domain map for the launch.</summary>
        [FieldOffset(0)] public CUlaunchMemSyncDomainMap memSyncDomainMap;
        /// <summary>Memory synchronization domain for the launch.</summary>
        [FieldOffset(0)] public CUlaunchMemSyncDomain memSyncDomain;
        /// <summary>Preferred cluster dimension for the launch.</summary>
        [FieldOffset(0)] public CUlaunchAttributeDim3 preferredClusterDim;
        /// <summary>Launch completion event for the launch.</summary>
        [FieldOffset(0)] public CUlaunchAttributeLaunchCompletionEvent launchCompletionEvent;
        /// <summary>Device updatable kernel node for the launch.</summary>
        [FieldOffset(0)] public CUlaunchAttributeDeviceUpdatableKernelNode deviceUpdatableKernelNode;
        /// <summary>Shared memory carveout for the launch.</summary>
        [FieldOffset(0)] public uint sharedMemCarveout;
        /// <summary>NVLink util-centric scheduling flag.</summary>
        [FieldOffset(0)] public uint nvlinkUtilCentricScheduling;
    }

    /// <summary>Launch attribute structure.</summary>
    /// <remarks>
    /// This structure is used to specify attributes for a kernel launch, including
    /// access policy, cooperative launch, synchronization policy, and more.
    /// </remarks>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUlaunchAttribute
    {
        /// <summary>ID of the launch attribute.</summary>
        public CUlaunchAttributeID id;
        /// <summary>Padding for alignment.</summary>
        public int pad0;
        /// <summary>Padding for alignment.</summary>
        public int pad1;
        /// <summary>Value of the launch attribute.</summary>
        public CUlaunchAttributeValue value;
    }

    /// <summary>Launch configuration structure.</summary>
    /// <remarks>
    /// This structure is used to specify the configuration for launching a kernel,
    /// including grid and block dimensions, shared memory size, stream, and attributes.
    /// </remarks>
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct CUlaunchConfig
    {
        /// <summary>Grid dimension X.</summary>
        public uint gridDimX;
        /// <summary>Grid dimension Y.</summary>
        public uint gridDimY;
        /// <summary>Grid dimension Z.</summary>
        public uint gridDimZ;
        /// <summary>Block dimension X.</summary>
        public uint blockDimX;
        /// <summary>Block dimension Y.</summary>
        public uint blockDimY;
        /// <summary>Block dimension Z.</summary>
        public uint blockDimZ;
        /// <summary>Shared memory size in bytes.</summary>
        public uint sharedMemBytes;
        /// <summary>Stream handle.</summary>
        public CUstream hStream;
        /// <summary>Pointer to launch attributes.</summary>
        public CUlaunchAttribute* attrs;
        /// <summary>Number of launch attributes.</summary>
        public uint numAttrs;
    }

    /// <summary>CUDA kernel node parameters.</summary>
    /// <remarks>
    /// This structure is used to specify parameters for a CUDA kernel node in a graph,
    /// including the function, grid and block dimensions, shared memory size, and more.
    /// </remarks>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_KERNEL_NODE_PARAMS
    {
        /// <summary>Function handle.</summary>
        public CUfunction func;
        /// <summary>Grid dimension X.</summary>
        public uint gridDimX;
        /// <summary>Grid dimension Y.</summary>
        public uint gridDimY;
        /// <summary>Grid dimension Z.</summary>
        public uint gridDimZ;
        /// <summary>Block dimension X.</summary>
        public uint blockDimX;
        /// <summary>Block dimension Y.</summary>
        public uint blockDimY;
        /// <summary>Block dimension Z.</summary>
        public uint blockDimZ;
        /// <summary>Shared memory size in bytes.</summary>
        public uint sharedMemBytes;
        /// <summary>Pointer to kernel parameters.</summary>
        public IntPtr kernelParams;
        /// <summary>Pointer to extra parameters.</summary>
        public IntPtr extra;
    }

    /// <summary>CUDA memcpy node parameters.</summary>
    /// <remarks>
    /// This structure is used to specify parameters for a CUDA memcpy node in a graph,
    /// including the 3D copy parameters.
    /// </remarks>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_MEMCPY_NODE_PARAMS
    {
        /// <summary>Memory copy parameters.</summary>
        public CUDA_MEMCPY3D copyParams;
    }

    /// <summary>CUDA memset node parameters.</summary>
    /// <remarks>
    /// This structure is used to specify parameters for a CUDA memset node in a graph,
    /// including the destination, pitch, value, element size, width, and height.
    /// </remarks>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_MEMSET_NODE_PARAMS
    {
        /// <summary>Destination device pointer.</summary>
        public CUdeviceptr dst;
        /// <summary>Pitch of the destination memory.</summary>
        public nuint pitch;
        /// <summary>Value to set.</summary>
        public uint value;
        /// <summary>Element size.</summary>
        public uint elementSize;
        /// <summary>Width of the region to set.</summary>
        public nuint width;
        /// <summary>Height of the region to set.</summary>
        public nuint height;
    }

    /// <summary>Stream batch memory operation types.</summary>
    public enum CUstreamBatchMemOpType
    {
        /// <summary>Wait for a 32-bit value.</summary>
        CU_STREAM_MEM_OP_WAIT_VALUE_32 = 1,
        /// <summary>Write a 32-bit value.</summary>
        CU_STREAM_MEM_OP_WRITE_VALUE_32 = 2,
        /// <summary>Wait for a 64-bit value.</summary>
        CU_STREAM_MEM_OP_WAIT_VALUE_64 = 4,
        /// <summary>Write a 64-bit value.</summary>
        CU_STREAM_MEM_OP_WRITE_VALUE_64 = 5,
        /// <summary>Barrier operation.</summary>
        CU_STREAM_MEM_OP_BARRIER = 6,
        /// <summary>Flush remote writes.</summary>
        CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = 3,
    }

    /// <summary>Stream batch memory operation parameters.</summary>
    [StructLayout(LayoutKind.Explicit)]
    public struct CUstreamBatchMemOpParams
    {
        /// <summary>Memory operation type.</summary>
        [FieldOffset(0)] public CUstreamBatchMemOpType operation;
        /// <summary>Parameters for wait value operation.</summary>
        [FieldOffset(8)] public CUstreamWaitValue_params waitValue;
        /// <summary>Parameters for write value operation.</summary>
        [FieldOffset(8)] public CUstreamWriteValue_params writeValue;
        /// <summary>Parameters for flush remote writes operation.</summary>
        [FieldOffset(8)] public CUstreamFlushRemoteWrites_params flushRemoteWrites;
        /// <summary>Parameters for barrier operation.</summary>
        [FieldOffset(8)] public CUstreamMemOpBarrier_params barrier;
        [FieldOffset(8)] public unsafe fixed ulong pad[6];
    }

    /// <summary>Parameters for waiting on a stream value.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUstreamWaitValue_params
    {
        /// <summary>Operation type.</summary>
        public CUstreamBatchMemOpType operation;
        /// <summary>Device pointer to the address.</summary>
        public CUdeviceptr address;
        /// <summary>Expected value.</summary>
        public CUstreamWaitValue_params_union value;
        /// <summary>Flags for the wait operation.</summary>
        public uint flags;
        /// <summary>Alias device pointer.</summary>
        public CUdeviceptr alias;
    }

    /// <summary>Union for stream wait value parameters.</summary>
    [StructLayout(LayoutKind.Explicit)]
    public struct CUstreamWaitValue_params_union
    {
        /// <summary>32-bit value.</summary>
        [FieldOffset(0)] public uint value;
        /// <summary>64-bit value.</summary>
        [FieldOffset(0)] public ulong value64;
    }

    /// <summary>Parameters for writing a stream value.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUstreamWriteValue_params
    {
        /// <summary>Operation type.</summary>
        public CUstreamBatchMemOpType operation;
        /// <summary>Device pointer to the address.</summary>
        public CUdeviceptr address;
        /// <summary>Value to write.</summary>
        public CUstreamWriteValue_params_union value;
        /// <summary>Flags for the write operation.</summary>
        public uint flags;
        /// <summary>Alias device pointer.</summary>
        public CUdeviceptr alias;
    }

    /// <summary>Union for stream write value parameters.</summary>
    [StructLayout(LayoutKind.Explicit)]
    public struct CUstreamWriteValue_params_union
    {
        /// <summary>32-bit value.</summary>
        [FieldOffset(0)] public uint value;
        /// <summary>64-bit value.</summary>
        [FieldOffset(0)] public ulong value64;
    }

    /// <summary>Parameters for flushing remote writes.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUstreamFlushRemoteWrites_params
    {
        /// <summary>Operation type.</summary>
        public CUstreamBatchMemOpType operation;
        /// <summary>Flags for the flush operation.</summary>
        public uint flags;
    }

    /// <summary>Parameters for stream memory operation barrier.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUstreamMemOpBarrier_params
    {
        /// <summary>Operation type.</summary>
        public CUstreamBatchMemOpType operation;
        /// <summary>Flags for the barrier operation.</summary>
        public uint flags;
    }

    /// <summary>Stream wait value flags.</summary>
    public enum CUstreamWaitValue_flags
    {
        /// <summary>Wait for value greater than or equal to the specified value.</summary>
        CU_STREAM_WAIT_VALUE_GEQ = 0x0,
        /// <summary>Wait for value equal to the specified value.</summary>
        CU_STREAM_WAIT_VALUE_EQ = 0x1,
        /// <summary>Perform a logical AND with the specified value.</summary>
        CU_STREAM_WAIT_VALUE_AND = 0x2,
        /// <summary>Perform a logical NOR with the specified value.</summary>
        CU_STREAM_WAIT_VALUE_NOR = 0x3,
        /// <summary>Flush remote writes before the wait.</summary>
        CU_STREAM_WAIT_VALUE_FLUSH = 1 << 30,
    }

    /// <summary>Stream write value flags.</summary>
    public enum CUstreamWriteValue_flags
    {
        /// <summary>Default write value behavior.</summary>
        CU_STREAM_WRITE_VALUE_DEFAULT = 0x0,
        /// <summary>No memory barrier after the write.</summary>
        CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER = 0x1,
    }

    /// <summary>External memory handle descriptor.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC
    {
        /// <summary>Type of external memory handle.</summary>
        public CUexternalMemoryHandleType type;
        /// <summary>Union of handle descriptors.</summary>
        public CUDA_EXTERNAL_MEMORY_HANDLE_DESC_UNION handle;
        /// <summary>Size of the external memory.</summary>
        public ulong size;
        /// <summary>Flags for external memory creation.</summary>
        public uint flags;
    }

    /// <summary>Union of external memory handle descriptors.</summary>
    [StructLayout(LayoutKind.Explicit)]
    public struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC_UNION
    {
        /// <summary>File descriptor for the handle.</summary>
        [FieldOffset(0)] public int fd;
        /// <summary>Windows handle for the handle.</summary>
        [FieldOffset(0)] public CUDA_EXTERNAL_MEMORY_HANDLE_DESC_WIN32 win32;
        /// <summary>NVSCI buffer object for the handle.</summary>
        [FieldOffset(0)] public IntPtr nvSciBufObject;
    }

    /// <summary>Windows-specific external memory handle descriptor.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC_WIN32
    {
        /// <summary>Handle to the memory.</summary>
        public IntPtr handle;
        /// <summary>Name of the memory region.</summary>
        public IntPtr name;
    }

    /// <summary>CUDA external memory handle types.</summary>
    public enum CUexternalMemoryHandleType
    {
        /// <summary>Opaque file descriptor.</summary>
        CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = 1,
        /// <summary>Opaque Windows handle.</summary>
        CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 = 2,
        /// <summary>Opaque Windows handle for Kernel-Mode.</summary>
        CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3,
        /// <summary>D3D12 heap handle.</summary>
        CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP = 4,
        /// <summary>D3D12 resource handle.</summary>
        CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE = 5,
        /// <summary>D3D11 resource handle.</summary>
        CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE = 6,
        /// <summary>D3D11 resource handle for Kernel-Mode.</summary>
        CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT = 7,
        /// <summary>NVSCI buffer handle.</summary>
        CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF = 8,
    }

    /// <summary>External memory buffer descriptor.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC
    {
        /// <summary>Offset from the base of the external memory.</summary>
        public ulong offset;
        /// <summary>Size of the buffer.</summary>
        public ulong size;
        /// <summary>Flags for buffer creation.</summary>
        public uint flags;
    }

    /// <summary>External memory mipmapped array descriptor.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC
    {
        /// <summary>Offset from the base of the external memory.</summary>
        public ulong offset;
        /// <summary>Descriptor for the array.</summary>
        public CUDA_ARRAY3D_DESCRIPTOR arrayDesc;
        /// <summary>Number of mipmap levels.</summary>
        public uint numLevels;
    }

    /// <summary>Descriptor for a 3D array.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_ARRAY3D_DESCRIPTOR
    {
        /// <summary>Width of the array.</summary>
        public nuint Width;
        /// <summary>Height of the array.</summary>
        public nuint Height;
        /// <summary>Depth of the array.</summary>
        public nuint Depth;
        /// <summary>Format of the array elements.</summary>
        public CUarray_format Format;
        /// <summary>Number of channels per element.</summary>
        public uint NumChannels;
        /// <summary>Flags for the array.</summary>
        public uint Flags;
    }

    /// <summary>External semaphore handle descriptor.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC
    {
        /// <summary>Type of external semaphore handle.</summary>
        public CUexternalSemaphoreHandleType type;
        /// <summary>Union of handle descriptors.</summary>
        public CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_UNION handle;
        /// <summary>Flags for external semaphore creation.</summary>
        public uint flags;
    }

    /// <summary>Union of external semaphore handle descriptors.</summary>
    [StructLayout(LayoutKind.Explicit)]
    public struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_UNION
    {
        /// <summary>File descriptor for the handle.</summary>
        [FieldOffset(0)] public int fd;
        /// <summary>Windows handle for the handle.</summary>
        [FieldOffset(0)] public CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_WIN32 win32;
        /// <summary>NVSCI sync object for the handle.</summary>
        [FieldOffset(0)] public IntPtr nvSciSyncObj;
    }

    /// <summary>Windows-specific external semaphore handle descriptor.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_WIN32
    {
        /// <summary>Handle to the semaphore.</summary>
        public IntPtr handle;
        /// <summary>Name of the semaphore.</summary>
        public IntPtr name;
    }

    /// <summary>CUDA external semaphore handle types.</summary>
    public enum CUexternalSemaphoreHandleType
    {
        /// <summary>Opaque file descriptor.</summary>
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD = 1,
        /// <summary>Opaque Windows handle.</summary>
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32 = 2,
        /// <summary>Opaque Windows handle for Kernel-Mode.</summary>
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3,
        /// <summary>D3D12 fence handle.</summary>
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE = 4,
        /// <summary>D3D11 fence handle.</summary>
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE = 5,
        /// <summary>NVSCI sync handle.</summary>
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC = 6,
        /// <summary>D3D11 keyed mutex handle.</summary>
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX = 7,
        /// <summary>D3D11 keyed mutex handle for Kernel-Mode.</summary>
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT = 8,
        /// <summary>Timeline semaphore file descriptor.</summary>
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD = 9,
        /// <summary>Timeline semaphore Windows handle.</summary>
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32 = 10,
    }

    /// <summary>External semaphore signal parameters.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS
    {
        /// <summary>Parameters for the signal operation.</summary>
        public CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_PARAMS params_;
        /// <summary>Flags for the signal operation.</summary>
        public uint flags;
        public unsafe fixed uint reserved[16];
    }

    /// <summary>Union of parameters for signaling an external semaphore.</summary>
    [StructLayout(LayoutKind.Explicit)]
    public struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_PARAMS
    {
        /// <summary>Fence parameters for signaling.</summary>
        [FieldOffset(0)] public CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_FENCE fence;
        /// <summary>NVSCI sync object for signaling.</summary>
        [FieldOffset(0)] public IntPtr nvSciSyncObj;
        /// <summary>Keyed mutex parameters for signaling.</summary>
        [FieldOffset(0)] public CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_KEYED_MUTEX keyedMutex;
    }

    /// <summary>Fence parameters for signaling an external semaphore.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_FENCE
    {
        /// <summary>Value for the fence.</summary>
        public ulong value;
    }

    /// <summary>Keyed mutex parameters for signaling an external semaphore.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_KEYED_MUTEX
    {
        /// <summary>Key for the mutex.</summary>
        public ulong key;
    }

    /// <summary>External semaphore wait parameters.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS
    {
        /// <summary>Parameters for the wait operation.</summary>
        public CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_PARAMS params_;
        /// <summary>Flags for the wait operation.</summary>
        public uint flags;
        public unsafe fixed uint reserved[16];
    }

    /// <summary>Union of parameters for waiting on an external semaphore.</summary>
    [StructLayout(LayoutKind.Explicit)]
    public struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_PARAMS
    {
        /// <summary>Fence parameters for waiting.</summary>
        [FieldOffset(0)] public CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_FENCE fence;
        /// <summary>NVSCI sync object for waiting.</summary>
        [FieldOffset(0)] public IntPtr nvSciSyncObj;
        /// <summary>Keyed mutex parameters for waiting.</summary>
        [FieldOffset(0)] public CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_KEYED_MUTEX keyedMutex;
    }

    /// <summary>Fence parameters for waiting on an external semaphore.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_FENCE
    {
        /// <summary>Value for the fence.</summary>
        public ulong value;
        /// <summary>Timeout in milliseconds.</summary>
        public uint timeoutMs;
    }

    /// <summary>Keyed mutex parameters for waiting on an external semaphore.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_KEYED_MUTEX
    {
        /// <summary>Key for the mutex.</summary>
        public ulong key;
        /// <summary>Timeout in milliseconds.</summary>
        public uint timeoutMs;
    }
}
