namespace CudaSharp;

/// <summary>
/// CUDA Driver API.
/// </summary>
/// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/index.html"/>
#pragma warning disable IDE1006 // Naming Styles
public static partial class nvcuda
#pragma warning restore IDE1006 // Naming Styles
{
    static nvcuda()
    {
        DllResolver.Register();
    }

    const string LibName = nameof(nvcuda);

    // Initialization and error handling

    /// <summary>
    /// Initialize the CUDA driver API.
    /// </summary>
    /// <param name="flags">Initialization flags. Should be 0.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuInit(uint flags = 0);

    /// <summary>
    /// Returns the latest version of CUDA supported by the driver.
    /// </summary>
    /// <param name="driverVersion">Returns the CUDA driver version.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuDriverGetVersion(out int driverVersion);

    /// <summary>Returns the symbolic name for a CUDA error code.</summary>
    /// <param name="error">Error code.</param>
    /// <param name="name">Returned pointer to a null-terminated error name.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuGetErrorName(CUresult error, out IntPtr name);

    /// <summary>Returns the description for a CUDA error code.</summary>
    /// <param name="error">Error code.</param>
    /// <param name="description">Returned pointer to a null-terminated error description.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuGetErrorString(CUresult error, out IntPtr description);

    /// <summary>Returns a CUDA driver entry point compatible with the requested CUDA version.</summary>
    /// <param name="symbol">Driver API symbol name.</param>
    /// <param name="pfn">Returned function pointer.</param>
    /// <param name="cudaVersion">Exact CUDA API version requested.</param>
    /// <param name="flags">Entry-point selection flags.</param>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial CUresult cuGetProcAddress(string symbol, out IntPtr pfn, int cudaVersion, ulong flags);

    /// <summary>Returns a CUDA driver entry point and reports why the query succeeded or failed.</summary>
    /// <param name="symbol">Driver API symbol name.</param>
    /// <param name="pfn">Returned function pointer.</param>
    /// <param name="cudaVersion">Exact CUDA API version requested.</param>
    /// <param name="flags">Entry-point selection flags.</param>
    /// <param name="symbolStatus">Returned symbol-query status.</param>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial CUresult cuGetProcAddress_v2(
        string symbol, out IntPtr pfn, int cudaVersion, ulong flags,
        out CUdriverProcAddressQueryResult symbolStatus);

    // Device management

    /// <summary>
    /// Returns the number of compute-capable devices.
    /// </summary>
    /// <param name="count">Returns the number of devices.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuDeviceGetCount(out int count);

    /// <summary>
    /// Returns a handle to a compute device.
    /// </summary>
    /// <param name="device">Returned device handle.</param>
    /// <param name="ordinal">Device number to get handle for.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuDeviceGet(out CUdevice device, int ordinal);

    /// <summary>
    /// Returns an identifier string for the device.
    /// </summary>
    /// <param name="name">Returned identifier string.</param>
    /// <param name="len">Maximum length of string to store in name.</param>
    /// <param name="dev">Device to get identifier string for.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuDeviceGetName(Span<byte> name, int len, CUdevice dev);

    /// <summary>
    /// Returns the total amount of memory on the device.
    /// </summary>
    /// <param name="bytes">Returned memory bytes.</param>
    /// <param name="dev">Device to get memory size for.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuDeviceTotalMem(out nuint bytes, CUdevice dev);

    /// <summary>
    /// Returns the compute capability of the device.
    /// </summary>
    /// <param name="major">Major revision number.</param>
    /// <param name="minor">Minor revision number.</param>
    /// <param name="dev">Device handle.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuDeviceComputeCapability(out int major, out int minor, CUdevice dev);

    /// <summary>
    /// Returns information about the device.
    /// </summary>
    /// <param name="pi">Returned attribute value.</param>
    /// <param name="attrib">Device attribute to query.</param>
    /// <param name="dev">Device handle.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuDeviceGetAttribute(out int pi, CUdevice_attribute attrib, CUdevice dev);

    /// <summary>
    /// Returns properties for a selected device.
    /// </summary>
    /// <param name="pProp">Returned properties.</param>
    /// <param name="dev">Device handle.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuDeviceGetProperties(out CUdevprop pProp, CUdevice dev);

    /// <summary>
    /// Returns the UUID of the device.
    /// </summary>
    /// <param name="uuid">Returned UUID.</param>
    /// <param name="dev">Device handle.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuDeviceGetUuid(out CUuuid uuid, CUdevice dev);

    /// <summary>
    /// Returns the LUID and node mask of the device.
    /// </summary>
    /// <param name="luid">Returned LUID.</param>
    /// <param name="deviceNodeMask">Returned node mask.</param>
    /// <param name="dev">Device handle.</param>
    [LibraryImport(LibName)]
    public static unsafe partial CUresult cuDeviceGetLuid(byte* luid, out uint deviceNodeMask, CUdevice dev);

    /// <summary>Returns the device associated with a PCI bus identifier.</summary>
    /// <param name="dev">Returned device.</param>
    /// <param name="pciBusId">Null-terminated PCI bus identifier.</param>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial CUresult cuDeviceGetByPCIBusId(out CUdevice dev, string pciBusId);

    /// <summary>Returns the PCI bus identifier for a device.</summary>
    /// <param name="pciBusId">Buffer that receives the null-terminated PCI bus identifier.</param>
    /// <param name="len">Buffer length in bytes.</param>
    /// <param name="dev">Device.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuDeviceGetPCIBusId(Span<byte> pciBusId, int len, CUdevice dev);

    // Context management

    /// <summary>
    /// Create a CUDA context.
    /// </summary>
    /// <param name="pctx">Returned context handle.</param>
    /// <param name="flags">Context creation flags.</param>
    /// <param name="dev">Device to create context on.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuCtxCreate(out CUcontext pctx, CUctx_flags flags, CUdevice dev);

    /// <summary>Creates a CUDA context using the version 2 ABI.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuCtxCreate_v2(out CUcontext pctx, CUctx_flags flags, CUdevice dev);

    /// <summary>
    /// Destroy a CUDA context.
    /// </summary>
    /// <param name="ctx">Context to destroy.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuCtxDestroy(CUcontext ctx);

    /// <summary>
    /// Returns the CUDA context bound to the calling CPU thread.
    /// </summary>
    /// <param name="pctx">Returned context handle.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuCtxGetCurrent(out CUcontext pctx);

    /// <summary>
    /// Binds the specified CUDA context to the calling CPU thread.
    /// </summary>
    /// <param name="ctx">Context to bind.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuCtxSetCurrent(CUcontext ctx);

    /// <summary>
    /// Enables direct access to memory allocations on a peer device.
    /// </summary>
    /// <param name="peerContext">Peer context to enable access to.</param>
    /// <param name="Flags">Reserved, must be 0.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuCtxEnablePeerAccess(CUcontext peerContext, uint Flags);

    /// <summary>
    /// Disables direct access to memory allocations on a peer device.
    /// </summary>
    /// <param name="peerContext">Peer context to disable access to.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuCtxDisablePeerAccess(CUcontext peerContext);

    /// <summary>
    /// Queries if a device may directly access a peer device's memory.
    /// </summary>
    /// <param name="canAccessPeer">Returned access capability.</param>
    /// <param name="dev">Device from which allocations on peerDev are to be accessed.</param>
    /// <param name="peerDev">Peer device.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuDeviceCanAccessPeer(out int canAccessPeer, CUdevice dev, CUdevice peerDev);

    /// <summary>
    /// Pushes a context on the current CPU thread.
    /// </summary>
    /// <param name="ctx">Context to push.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuCtxPushCurrent(CUcontext ctx);

    /// <summary>
    /// Pops the current CUDA context from the current CPU thread.
    /// </summary>
    /// <param name="pctx">Returned popped context.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuCtxPopCurrent(out CUcontext pctx);

    /// <summary>
    /// Returns the device ID for the current context.
    /// </summary>
    /// <param name="device">Returned device ID.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuCtxGetDevice(out CUdevice device);

    /// <summary>
    /// Returns resource limits.
    /// </summary>
    /// <param name="pvalue">Returned limit value.</param>
    /// <param name="limit">Limit to query.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuCtxGetLimit(out nuint pvalue, CUlimit limit);

    /// <summary>
    /// Set resource limits.
    /// </summary>
    /// <param name="limit">Limit to set.</param>
    /// <param name="value">Limit value.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuCtxSetLimit(CUlimit limit, nuint value);

    /// <summary>
    /// Returns the preferred cache configuration for the current context.
    /// </summary>
    /// <param name="pconfig">Returned cache configuration.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuCtxGetCacheConfig(out CUfunc_cache pconfig);

    /// <summary>
    /// Sets the preferred cache configuration for the current context.
    /// </summary>
    /// <param name="config">Requested cache configuration.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuCtxSetCacheConfig(CUfunc_cache config);

    /// <summary>
    /// Returns the current shared memory configuration for the current context.
    /// </summary>
    /// <param name="pConfig">Returned shared memory configuration.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuCtxGetSharedMemConfig(out CUsharedconfig pConfig);

    /// <summary>
    /// Sets the shared memory configuration for the current context.
    /// </summary>
    /// <param name="config">Requested shared memory configuration.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuCtxSetSharedMemConfig(CUsharedconfig config);

    /// <summary>Returns the API version used to create a context.</summary>
    /// <param name="ctx">Context.</param>
    /// <param name="version">Returned API version.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuCtxGetApiVersion(CUcontext ctx, out uint version);

    /// <summary>Returns the flags for the current context.</summary>
    /// <param name="flags">Returned context flags.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuCtxGetFlags(out CUctx_flags flags);

    /// <summary>Returns the supported stream-priority range for the current context.</summary>
    /// <param name="leastPriority">Returned least-favorable priority.</param>
    /// <param name="greatestPriority">Returned greatest-favorable priority.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuCtxGetStreamPriorityRange(out int leastPriority, out int greatestPriority);

    /// <summary>Waits for all preceding work in the current context to complete.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuCtxSynchronize();

    // Profiler control

    /// <summary>
    /// Initialize the profiling.
    /// </summary>
    /// <param name="configFile">Name of the config file.</param>
    /// <param name="outputMode">Output mode.</param>
    /// <param name="mode">Profiler output mode.</param>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial CUresult cuProfilerInitialize(
        string configFile, string outputMode,
        CUprofiler_outputMode mode);

    /// <summary>
    /// Enable profiling.
    /// </summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuProfilerStart();

    /// <summary>
    /// Disable profiling.
    /// </summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuProfilerStop();

    // Primary context management

    /// <summary>
    /// Retain the primary context on the GPU.
    /// </summary>
    /// <param name="pctx">Returned context handle.</param>
    /// <param name="dev">Device to get primary context for.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuDevicePrimaryCtxRetain(out CUcontext pctx, CUdevice dev);

    /// <summary>
    /// Release the primary context on the GPU.
    /// </summary>
    /// <param name="dev">Device to release primary context for.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuDevicePrimaryCtxRelease(CUdevice dev);

    /// <summary>
    /// Set flags for the primary context.
    /// </summary>
    /// <param name="dev">Device to set flags for.</param>
    /// <param name="flags">Flags to set.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, uint flags);

    /// <summary>
    /// Get the state of the primary context.
    /// </summary>
    /// <param name="dev">Device to get state for.</param>
    /// <param name="flags">Returned flags.</param>
    /// <param name="active">Returned active status.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuDevicePrimaryCtxGetState(CUdevice dev, out uint flags, out int active);

    /// <summary>
    /// Destroy all allocations and reset all state on the primary context.
    /// </summary>
    /// <param name="dev">Device to reset.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuDevicePrimaryCtxReset(CUdevice dev);

    // Module, library, and linker management

    /// <summary>
    /// Loads a compute module.
    /// </summary>
    /// <param name="module">Returned module.</param>
    /// <param name="fname">Filename of module to load.</param>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial CUresult cuModuleLoad(out CUmodule module, string fname);

    /// <summary>
    /// Loads a compute module from a memory buffer.
    /// </summary>
    /// <param name="module">Returned module.</param>
    /// <param name="image">Module data to load.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuModuleLoadData(out CUmodule module, ReadOnlySpan<byte> image);

    /// <summary>Loads a compute module from a fat binary image.</summary>
    /// <param name="module">Returned module.</param>
    /// <param name="fatCubin">Fat binary image.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuModuleLoadFatBinary(out CUmodule module, ReadOnlySpan<byte> fatCubin);

    /// <summary>
    /// Loads a compute module from a memory buffer with JIT options.
    /// </summary>
    /// <param name="module">Returned module.</param>
    /// <param name="image">Module data to load.</param>
    /// <param name="numOptions">Number of JIT options.</param>
    /// <param name="options">JIT options.</param>
    /// <param name="optionValues">JIT option values.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g04ce266ce03720f479eab76136b90c0b"/>
    [LibraryImport(LibName)]
    public unsafe static partial CUresult cuModuleLoadDataEx(
        out CUmodule module,
        ReadOnlySpan<byte> image,
        uint numOptions,
        CUjit_option* options,
        void** optionValues);

    /// <summary>
    /// Loads a CUDA library from a memory buffer.
    /// </summary>
    /// <param name="library">Returned library handle.</param>
    /// <param name="code">Library image data.</param>
    /// <param name="jitOptions">JIT options.</param>
    /// <param name="jitOptionValues">JIT option values.</param>
    /// <param name="numJitOptions">Number of JIT options.</param>
    /// <param name="libraryOptions">Library loading options.</param>
    /// <param name="libraryOptionValues">Library loading option values.</param>
    /// <param name="numLibraryOptions">Number of library loading options.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g1ceda3d5439f0f307c4617f3f144ed39"/>
    [LibraryImport(LibName)]
    public unsafe static partial CUresult cuLibraryLoadData(
        out CUlibrary library,
        ReadOnlySpan<byte> code,
        CUjit_option* jitOptions,
        void** jitOptionValues,
        uint numJitOptions,
        CUlibraryOption* libraryOptions,
        void** libraryOptionValues,
        uint numLibraryOptions);

    /// <summary>
    /// Unloads a CUDA library.
    /// </summary>
    /// <param name="library">Library handle to unload.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1gff7274f8d79f5f1b28ee1daeedfe52c"/>
    [LibraryImport(LibName)]
    public static partial CUresult cuLibraryUnload(CUlibrary library);

    /// <summary>
    /// Gets a kernel handle from a CUDA library.
    /// </summary>
    /// <param name="kernel">Returned kernel handle.</param>
    /// <param name="library">Library handle.</param>
    /// <param name="name">Kernel name.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g6f8f15f71bb32e4505d0952f392e143c"/>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial CUresult cuLibraryGetKernel(
        out CUkernel kernel,
        CUlibrary library,
        string name);

    /// <summary>
    /// Gets a module handle for the current context from a CUDA library.
    /// </summary>
    /// <param name="module">Returned module handle.</param>
    /// <param name="library">Library handle.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1gcf52df4b6ca93511d7a6ff4827fd9fb6"/>
    [LibraryImport(LibName)]
    public static partial CUresult cuLibraryGetModule(out CUmodule module, CUlibrary library);

    /// <summary>
    /// Gets a function handle for a kernel in the current context.
    /// </summary>
    /// <param name="function">Returned function handle.</param>
    /// <param name="kernel">Kernel handle.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g67df7712e6c0aef0f3b731f17df0ee06"/>
    [LibraryImport(LibName)]
    public static partial CUresult cuKernelGetFunction(out CUfunction function, CUkernel kernel);

    /// <summary>
    /// Creates a pending JIT linker invocation.
    /// </summary>
    /// <param name="numOptions">Number of linker options.</param>
    /// <param name="options">Linker options.</param>
    /// <param name="optionValues">Option values cast to pointers.</param>
    /// <param name="stateOut">Returned linker state.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g66f3f648936f2d1a4469af044bb6877f"/>
    [LibraryImport(LibName)]
    public unsafe static partial CUresult cuLinkCreate(
        uint numOptions,
        CUjit_option* options,
        void** optionValues,
        out CUlinkState stateOut);

    /// <summary>
    /// Adds an input buffer to a pending JIT linker invocation.
    /// </summary>
    /// <param name="state">Linker state.</param>
    /// <param name="type">Input type.</param>
    /// <param name="data">Input data pointer.</param>
    /// <param name="size">Input size in bytes.</param>
    /// <param name="name">Optional input name for diagnostics.</param>
    /// <param name="numOptions">Number of input-specific options.</param>
    /// <param name="options">Input-specific options.</param>
    /// <param name="optionValues">Input-specific option values.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g8b1d53da30f3bfda52d58a3b16130ad9"/>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    public unsafe static partial CUresult cuLinkAddData(
        CUlinkState state,
        CUjitInputType type,
        void* data,
        nuint size,
        string name,
        uint numOptions,
        CUjit_option* options,
        void** optionValues);

    /// <summary>
    /// Adds a file input to a pending JIT linker invocation.
    /// </summary>
    /// <param name="state">Linker state.</param>
    /// <param name="type">Input type.</param>
    /// <param name="path">Input file path.</param>
    /// <param name="numOptions">Number of input-specific options.</param>
    /// <param name="options">Input-specific options.</param>
    /// <param name="optionValues">Input-specific option values.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g07f1b6ee4635b2a31337c289f4c5ca2e"/>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    public unsafe static partial CUresult cuLinkAddFile(
        CUlinkState state,
        CUjitInputType type,
        string path,
        uint numOptions,
        CUjit_option* options,
        void** optionValues);

    /// <summary>
    /// Completes a pending JIT linker invocation.
    /// </summary>
    /// <param name="state">Linker state.</param>
    /// <param name="cubinOut">Returned linked cubin pointer.</param>
    /// <param name="sizeOut">Returned linked cubin size.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g3d5adfca26af1022ac7565594343e9f6"/>
    [LibraryImport(LibName)]
    public static partial CUresult cuLinkComplete(
        CUlinkState state,
        out IntPtr cubinOut,
        out nuint sizeOut);

    /// <summary>
    /// Destroys a pending JIT linker invocation.
    /// </summary>
    /// <param name="state">Linker state.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g66d5ea4b2d3fd3dfc9af1d9305f22f63"/>
    [LibraryImport(LibName)]
    public static partial CUresult cuLinkDestroy(CUlinkState state);

    /// <summary>
    /// Returns a function handle.
    /// </summary>
    /// <param name="hfunc">Returned function handle.</param>
    /// <param name="hmod">Module to retrieve function from.</param>
    /// <param name="name">Name of function to retrieve.</param>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial CUresult cuModuleGetFunction(out CUfunction hfunc, CUmodule hmod, string name);

    /// <summary>Returns the number of functions in a CUDA module.</summary>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html" />
    [LibraryImport(LibName)]
    public static partial CUresult cuModuleGetFunctionCount(out uint count, CUmodule module);

    /// <summary>Enumerates functions in a CUDA module.</summary>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html" />
    [LibraryImport(LibName)]
    public static partial CUresult cuModuleEnumerateFunctions(Span<CUfunction> functions, uint count, CUmodule module);

    /// <summary>Returns the name of a CUDA function.</summary>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__FUNCTION.html" />
    [LibraryImport(LibName)]
    public static partial CUresult cuFuncGetName(out IntPtr name, CUfunction function);

    /// <summary>Returns a global variable from a module.</summary>
    /// <param name="dptr">Returned device pointer.</param>
    /// <param name="bytes">Returned variable size in bytes.</param>
    /// <param name="hmod">Module.</param>
    /// <param name="name">Variable name.</param>
    [LibraryImport(LibName, EntryPoint = "cuModuleGetGlobal_v2", StringMarshalling = StringMarshalling.Utf8)]
    public static partial CUresult cuModuleGetGlobal(
        out CUdeviceptr dptr, out nuint bytes,
        CUmodule hmod, string name);

    /// <summary>Returns a texture reference from a module.</summary>
    /// <param name="texRef">Returned texture reference.</param>
    /// <param name="hmod">Module.</param>
    /// <param name="name">Texture-reference name.</param>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial CUresult cuModuleGetTexRef(out CUtexref texRef, CUmodule hmod, string name);

    /// <summary>Returns a surface reference from a module.</summary>
    /// <param name="surfRef">Returned surface reference.</param>
    /// <param name="hmod">Module.</param>
    /// <param name="name">Surface-reference name.</param>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial CUresult cuModuleGetSurfRef(out CUsurfref surfRef, CUmodule hmod, string name);

    /// <summary>
    /// Unloads a module.
    /// </summary>
    /// <param name="hmod">Module to unload.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuModuleUnload(CUmodule hmod);

    // Array management

    /// <summary>Creates a one- or two-dimensional CUDA array.</summary>
    /// <param name="array">Returned array.</param>
    /// <param name="descriptor">Array descriptor.</param>
    [LibraryImport(LibName, EntryPoint = "cuArrayCreate_v2")]
    public static partial CUresult cuArrayCreate(out CUarray array, in CUDA_ARRAY_DESCRIPTOR descriptor);

    /// <summary>Returns the descriptor for a CUDA array.</summary>
    /// <param name="descriptor">Returned array descriptor.</param>
    /// <param name="array">Array.</param>
    [LibraryImport(LibName, EntryPoint = "cuArrayGetDescriptor_v2")]
    public static partial CUresult cuArrayGetDescriptor(out CUDA_ARRAY_DESCRIPTOR descriptor, CUarray array);

    /// <summary>Creates a one-, two-, or three-dimensional CUDA array.</summary>
    /// <param name="array">Returned array.</param>
    /// <param name="descriptor">Three-dimensional array descriptor.</param>
    [LibraryImport(LibName, EntryPoint = "cuArray3DCreate_v2")]
    public static partial CUresult cuArray3DCreate(out CUarray array, in CUDA_ARRAY3D_DESCRIPTOR descriptor);

    /// <summary>Returns the three-dimensional descriptor for a CUDA array.</summary>
    /// <param name="descriptor">Returned descriptor.</param>
    /// <param name="array">Array.</param>
    [LibraryImport(LibName, EntryPoint = "cuArray3DGetDescriptor_v2")]
    public static partial CUresult cuArray3DGetDescriptor(out CUDA_ARRAY3D_DESCRIPTOR descriptor, CUarray array);

    /// <summary>Destroys a CUDA array.</summary>
    /// <param name="array">Array to destroy.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuArrayDestroy(CUarray array);

    /// <summary>Creates a CUDA mipmapped array.</summary>
    /// <param name="mipmappedArray">Returned mipmapped array.</param>
    /// <param name="descriptor">Array descriptor.</param>
    /// <param name="numMipmapLevels">Number of mipmap levels.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuMipmappedArrayCreate(
        out CUmipmappedArray mipmappedArray,
        in CUDA_ARRAY3D_DESCRIPTOR descriptor,
        uint numMipmapLevels);

    /// <summary>Returns one level of a CUDA mipmapped array.</summary>
    /// <param name="levelArray">Returned level array.</param>
    /// <param name="mipmappedArray">Mipmapped array.</param>
    /// <param name="level">Mipmap level.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuMipmappedArrayGetLevel(
        out CUarray levelArray, CUmipmappedArray mipmappedArray, uint level);

    /// <summary>Destroys a CUDA mipmapped array.</summary>
    /// <param name="mipmappedArray">Mipmapped array to destroy.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuMipmappedArrayDestroy(CUmipmappedArray mipmappedArray);

    // Texture references

    /// <summary>
    /// Creates a texture reference.
    /// </summary>
    /// <param name="pTexRef">Returned texture reference.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuTexRefCreate(out CUtexref pTexRef);

    /// <summary>
    /// Destroys a texture reference.
    /// </summary>
    /// <param name="hTexRef">Texture reference to destroy.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuTexRefDestroy(CUtexref hTexRef);

    /// <summary>
    /// Binds an array to a texture reference.
    /// </summary>
    /// <param name="hTexRef">Texture reference to bind.</param>
    /// <param name="hArray">Array to bind.</param>
    /// <param name="Flags">Texture attachment flags.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, uint Flags);

    /// <summary>
    /// Binds an address to a texture reference.
    /// </summary>
    /// <param name="ByteOffset">Returned byte offset.</param>
    /// <param name="hTexRef">Texture reference to bind.</param>
    /// <param name="dptr">Device pointer to bind.</param>
    /// <param name="bytes">Size of memory to bind.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuTexRefSetAddress(
        out nuint ByteOffset,
        CUtexref hTexRef, CUdeviceptr dptr,
        nuint bytes);

    /// <summary>
    /// Binds an address to a texture reference.
    /// </summary>
    /// <param name="hTexRef">Texture reference to bind.</param>
    /// <param name="desc">Array descriptor.</param>
    /// <param name="dptr">Device pointer to bind.</param>
    /// <param name="Pitch">Pitch of linear memory.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuTexRefSetAddress2D(
        CUtexref hTexRef, in CUDA_ARRAY_DESCRIPTOR desc,
        CUdeviceptr dptr, nuint Pitch);

    /// <summary>
    /// Sets the format for a texture reference.
    /// </summary>
    /// <param name="hTexRef">Texture reference to set format for.</param>
    /// <param name="fmt">Format to set.</param>
    /// <param name="NumPackedComponents">Number of components.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents);

    /// <summary>
    /// Sets the addressing mode for a texture reference.
    /// </summary>
    /// <param name="hTexRef">Texture reference.</param>
    /// <param name="dim">Dimension.</param>
    /// <param name="am">Addressing mode.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am);

    /// <summary>
    /// Sets the filtering mode for a texture reference.
    /// </summary>
    /// <param name="hTexRef">Texture reference.</param>
    /// <param name="fm">Filtering mode.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm);

    /// <summary>
    /// Sets the flags for a texture reference.
    /// </summary>
    /// <param name="hTexRef">Texture reference.</param>
    /// <param name="Flags">Flags.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuTexRefSetFlags(CUtexref hTexRef, uint Flags);

    /// <summary>
    /// Gets the address associated with a texture reference.
    /// </summary>
    /// <param name="pdptr">Returned device pointer.</param>
    /// <param name="hTexRef">Texture reference.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuTexRefGetAddress(out CUdeviceptr pdptr, CUtexref hTexRef);

    /// <summary>
    /// Gets the array bound to a texture reference.
    /// </summary>
    /// <param name="phArray">Returned array.</param>
    /// <param name="hTexRef">Texture reference.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuTexRefGetArray(out CUarray phArray, CUtexref hTexRef);

    /// <summary>
    /// Gets the addressing mode used by a texture reference.
    /// </summary>
    /// <param name="pam">Returned addressing mode.</param>
    /// <param name="hTexRef">Texture reference.</param>
    /// <param name="dim">Dimension.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuTexRefGetAddressMode(out CUaddress_mode pam, CUtexref hTexRef, int dim);

    /// <summary>
    /// Gets the filter mode used by a texture reference.
    /// </summary>
    /// <param name="pfm">Returned filter mode.</param>
    /// <param name="hTexRef">Texture reference.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuTexRefGetFilterMode(out CUfilter_mode pfm, CUtexref hTexRef);

    /// <summary>
    /// Gets the format used by a texture reference.
    /// </summary>
    /// <param name="pFormat">Returned format.</param>
    /// <param name="pNumPackedComponents">Returned number of components.</param>
    /// <param name="hTexRef">Texture reference.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuTexRefGetFormat(
        out CUarray_format pFormat, out int pNumPackedComponents,
        CUtexref hTexRef);

    /// <summary>
    /// Gets the flags used by a texture reference.
    /// </summary>
    /// <param name="pFlags">Returned flags.</param>
    /// <param name="hTexRef">Texture reference.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuTexRefGetFlags(out uint pFlags, CUtexref hTexRef);

    // Surface references

    /// <summary>
    /// Creates a surface reference.
    /// </summary>
    /// <param name="pSurfRef">Returned surface reference.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuSurfRefCreate(out CUsurfref pSurfRef);

    /// <summary>
    /// Destroys a surface reference.
    /// </summary>
    /// <param name="hSurfRef">Surface reference to destroy.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuSurfRefDestroy(CUsurfref hSurfRef);

    /// <summary>
    /// Sets the array for a surface reference.
    /// </summary>
    /// <param name="hSurfRef">Surface reference.</param>
    /// <param name="hArray">Array to bind.</param>
    /// <param name="Flags">Flags.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, uint Flags);

    /// <summary>
    /// Gets the array bound to a surface reference.
    /// </summary>
    /// <param name="phArray">Returned array.</param>
    /// <param name="hSurfRef">Surface reference.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuSurfRefGetArray(out CUarray phArray, CUsurfref hSurfRef);

    // Texture objects

    /// <summary>
    /// Creates a texture object.
    /// </summary>
    /// <param name="pTexObject">Returned texture object.</param>
    /// <param name="pResDesc">Resource descriptor.</param>
    /// <param name="pTexDesc">Texture descriptor.</param>
    /// <param name="pResViewDesc">Resource view descriptor.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuTexObjectCreate(
        out CUtexObject pTexObject, in CUDA_RESOURCE_DESC pResDesc,
        in CUDA_TEXTURE_DESC pTexDesc,
        in CUDA_RESOURCE_VIEW_DESC pResViewDesc);

    /// <summary>
    /// Destroys a texture object.
    /// </summary>
    /// <param name="texObject">Texture object to destroy.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuTexObjectDestroy(CUtexObject texObject);

    /// <summary>
    /// Returns a texture object's resource descriptor.
    /// </summary>
    /// <param name="pResDesc">Returned resource descriptor.</param>
    /// <param name="texObject">Texture object.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuTexObjectGetResourceDesc(out CUDA_RESOURCE_DESC pResDesc, CUtexObject texObject);

    /// <summary>
    /// Returns a texture object's texture descriptor.
    /// </summary>
    /// <param name="pTexDesc">Returned texture descriptor.</param>
    /// <param name="texObject">Texture object.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuTexObjectGetTextureDesc(out CUDA_TEXTURE_DESC pTexDesc, CUtexObject texObject);

    /// <summary>
    /// Returns a texture object's resource view descriptor.
    /// </summary>
    /// <param name="pResViewDesc">Returned resource view descriptor.</param>
    /// <param name="texObject">Texture object.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuTexObjectGetResourceViewDesc(out CUDA_RESOURCE_VIEW_DESC pResViewDesc, CUtexObject texObject);

    // Surface objects

    /// <summary>
    /// Creates a surface object.
    /// </summary>
    /// <param name="pSurfObject">Returned surface object.</param>
    /// <param name="pResDesc">Resource descriptor.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuSurfObjectCreate(out CUsurfObject pSurfObject, in CUDA_RESOURCE_DESC pResDesc);

    /// <summary>
    /// Destroys a surface object.
    /// </summary>
    /// <param name="surfObject">Surface object to destroy.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuSurfObjectDestroy(CUsurfObject surfObject);

    /// <summary>
    /// Returns a surface object's resource descriptor.
    /// </summary>
    /// <param name="pResDesc">Returned resource descriptor.</param>
    /// <param name="surfObject">Surface object.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuSurfObjectGetResourceDesc(out CUDA_RESOURCE_DESC pResDesc, CUsurfObject surfObject);

    // Execution control and occupancy

    /// <summary>Returns an attribute of a CUDA function.</summary>
    /// <param name="value">Returned attribute value.</param>
    /// <param name="attribute">Attribute to query.</param>
    /// <param name="function">Function.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuFuncGetAttribute(
        out int value, CUfunction_attribute attribute, CUfunction function);

    /// <summary>Sets an attribute on a CUDA function.</summary>
    /// <param name="attribute">Attribute to set.</param>
    /// <param name="value">Attribute value.</param>
    /// <param name="function">Function.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuFuncSetAttribute(
        CUfunction_attribute attribute, int value, CUfunction function);

    /// <summary>Sets the preferred cache configuration for a CUDA function.</summary>
    /// <param name="function">Function.</param>
    /// <param name="config">Preferred cache configuration.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuFuncSetCacheConfig(CUfunction function, CUfunc_cache config);

    /// <summary>Sets the shared-memory bank size for a CUDA function.</summary>
    /// <param name="function">Function.</param>
    /// <param name="config">Shared-memory configuration.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuFuncSetSharedMemConfig(CUfunction function, CUsharedconfig config);

    /// <summary>Launches a CUDA function on a grid.</summary>
    [LibraryImport(LibName)]
    public unsafe static partial CUresult cuLaunchKernel(CUfunction f,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes, CUstream hStream,
        void** kernelParams, void** extra);

    /// <summary>Launches a CUDA function using an extensible launch configuration.</summary>
    [LibraryImport(LibName)]
    public unsafe static partial CUresult cuLaunchKernelEx(
        in CUlaunchConfig config,
        CUfunction f,
        void** kernelParams,
        void** extra);

    /// <summary>Returns the maximum active blocks per multiprocessor for a function.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(
        out int numBlocks, CUfunction func,
        int blockSize, nuint dynamicSMemSize);

    /// <summary>Returns the maximum active blocks per multiprocessor using occupancy flags.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        out int numBlocks, CUfunction func,
        int blockSize, nuint dynamicSMemSize, uint flags);

    // Memory management

    /// <summary>Allocates device memory.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemAlloc(out CUdeviceptr dptr, nuint bytesize);

    /// <summary>Allocates device memory using the version 2 ABI.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemAlloc_v2(out CUdeviceptr dptr, nuint bytesize);

    /// <summary>Frees device memory.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemFree(CUdeviceptr dptr);

    /// <summary>Frees device memory using the version 2 ABI.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemFree_v2(CUdeviceptr dptr);

    /// <summary>Copies memory from host to device.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, IntPtr srcHost, nuint bytesize);

    /// <summary>Copies memory from host to device using the version 2 ABI.</summary>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html" />
    [LibraryImport(LibName)]
    public static partial CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, IntPtr srcHost, nuint bytesize);

    /// <summary>Copies memory from device to host.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemcpyDtoH(IntPtr dstHost, CUdeviceptr srcDevice, nuint bytesize);

    /// <summary>Copies memory from device to host using the version 2 ABI.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemcpyDtoH_v2(IntPtr dstHost, CUdeviceptr srcDevice, nuint bytesize);

    /// <summary>Asynchronously copies memory from host to device.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemcpyHtoDAsync(
        CUdeviceptr dstDevice, IntPtr srcHost,
        nuint bytesize, CUstream hStream);

    /// <summary>Asynchronously copies memory from device to host.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemcpyDtoHAsync(
        IntPtr dstHost, CUdeviceptr srcDevice,
        nuint bytesize, CUstream hStream);

    /// <summary>Asynchronously copies memory between device allocations.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemcpyDtoDAsync(
        CUdeviceptr dstDevice, CUdeviceptr srcDevice,
        nuint bytesize, CUstream hStream);

    /// <summary>Copies memory between two unified virtual addresses.</summary>
    /// <param name="destination">Destination device address.</param>
    /// <param name="source">Source device address.</param>
    /// <param name="bytes">Number of bytes to copy.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemcpy(CUdeviceptr destination, CUdeviceptr source, nuint bytes);

    /// <summary>Copies memory between two device allocations.</summary>
    /// <param name="destination">Destination device pointer.</param>
    /// <param name="source">Source device pointer.</param>
    /// <param name="bytes">Number of bytes to copy.</param>
    [LibraryImport(LibName, EntryPoint = "cuMemcpyDtoD_v2")]
    public static partial CUresult cuMemcpyDtoD(CUdeviceptr destination, CUdeviceptr source, nuint bytes);

    /// <summary>Copies memory between devices in different contexts.</summary>
    /// <param name="destination">Destination device pointer.</param>
    /// <param name="destinationContext">Destination context.</param>
    /// <param name="source">Source device pointer.</param>
    /// <param name="sourceContext">Source context.</param>
    /// <param name="bytes">Number of bytes to copy.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemcpyPeer(
        CUdeviceptr destination, CUcontext destinationContext,
        CUdeviceptr source, CUcontext sourceContext,
        nuint bytes);

    /// <summary>Asynchronously copies memory between devices in different contexts.</summary>
    /// <param name="destination">Destination device pointer.</param>
    /// <param name="destinationContext">Destination context.</param>
    /// <param name="source">Source device pointer.</param>
    /// <param name="sourceContext">Source context.</param>
    /// <param name="bytes">Number of bytes to copy.</param>
    /// <param name="stream">Stream.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemcpyPeerAsync(
        CUdeviceptr destination, CUcontext destinationContext,
        CUdeviceptr source, CUcontext sourceContext,
        nuint bytes, CUstream stream);

    /// <summary>Allocates pitched device memory.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemAllocPitch(
        out CUdeviceptr dptr, out nuint pPitch,
        nuint WidthInBytes, nuint Height, uint ElementSizeBytes);

    /// <summary>Frees page-locked host memory.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemFreeHost(IntPtr p);

    /// <summary>Allocates page-locked host memory with flags.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemHostAlloc(out IntPtr pp, nuint bytesize, uint Flags);

    /// <summary>Returns free and total device memory for the current context.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemGetInfo(out nuint free, out nuint total);

    /// <summary>Returns the allocation base and size for a device pointer.</summary>
    /// <param name="basePointer">Returned allocation base.</param>
    /// <param name="size">Returned allocation size.</param>
    /// <param name="pointer">Address within the allocation.</param>
    [LibraryImport(LibName, EntryPoint = "cuMemGetAddressRange_v2")]
    public static partial CUresult cuMemGetAddressRange(
        out CUdeviceptr basePointer, out nuint size, CUdeviceptr pointer);

    /// <summary>Allocates page-locked host memory.</summary>
    /// <param name="pointer">Returned host pointer.</param>
    /// <param name="bytes">Allocation size in bytes.</param>
    [LibraryImport(LibName, EntryPoint = "cuMemAllocHost_v2")]
    public static partial CUresult cuMemAllocHost(out IntPtr pointer, nuint bytes);

    /// <summary>Returns the device pointer corresponding to mapped host memory.</summary>
    /// <param name="devicePointer">Returned device pointer.</param>
    /// <param name="hostPointer">Host pointer.</param>
    /// <param name="flags">Reserved; must be zero.</param>
    [LibraryImport(LibName, EntryPoint = "cuMemHostGetDevicePointer_v2")]
    public static partial CUresult cuMemHostGetDevicePointer(
        out CUdeviceptr devicePointer, IntPtr hostPointer, uint flags = 0);

    /// <summary>Returns the flags used to allocate page-locked host memory.</summary>
    /// <param name="flags">Returned allocation flags.</param>
    /// <param name="hostPointer">Host pointer.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemHostGetFlags(out CUmemhostalloc_flags flags, IntPtr hostPointer);

    /// <summary>Registers an existing host-memory range as page-locked.</summary>
    /// <param name="hostPointer">Host pointer.</param>
    /// <param name="bytes">Range size in bytes.</param>
    /// <param name="flags">Registration flags.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemHostRegister(IntPtr hostPointer, nuint bytes, uint flags);

    /// <summary>Unregisters a page-locked host-memory range.</summary>
    /// <param name="hostPointer">Host pointer.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemHostUnregister(IntPtr hostPointer);

    /// <summary>Performs a two-dimensional memory copy.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemcpy2D(in CUDA_MEMCPY2D pCopy);

    /// <summary>Performs a three-dimensional memory copy.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemcpy3D(in CUDA_MEMCPY3D pCopy);

    /// <summary>Asynchronously performs a two-dimensional memory copy.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemcpy2DAsync(in CUDA_MEMCPY2D pCopy, CUstream hStream);

    /// <summary>Asynchronously performs a three-dimensional memory copy.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemcpy3DAsync(in CUDA_MEMCPY3D pCopy, CUstream hStream);

    /// <summary>Sets eight-bit elements in device memory.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemsetD8(CUdeviceptr dstDevice, byte uc, nuint N);

    [LibraryImport(LibName, EntryPoint = "cuMemsetD8_v2")]
    public static partial CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, byte uc, nuint N);

    /// <summary>Sets sixteen-bit elements in device memory.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemsetD16(CUdeviceptr dstDevice, ushort us, nuint N);

    /// <summary>Sets thirty-two-bit elements in device memory.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemsetD32(CUdeviceptr dstDevice, uint ui, nuint N);

    /// <summary>Sets thirty-two-bit elements in device memory using the version 2 ABI.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemsetD32_v2(CUdeviceptr dstDevice, uint ui, nuint N);

    /// <summary>Sets eight-bit elements in a two-dimensional device-memory region.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemsetD2D8(
        CUdeviceptr dstDevice, nuint dstPitch,
        byte uc, nuint Width, nuint Height);

    /// <summary>Sets sixteen-bit elements in a two-dimensional device-memory region.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemsetD2D16(
        CUdeviceptr dstDevice, nuint dstPitch,
        ushort us, nuint Width, nuint Height);

    /// <summary>Sets thirty-two-bit elements in a two-dimensional device-memory region.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemsetD2D32(
        CUdeviceptr dstDevice, nuint dstPitch,
        uint ui, nuint Width, nuint Height);

    /// <summary>Asynchronously sets eight-bit elements in device memory.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemsetD8Async(CUdeviceptr dstDevice, byte uc, nuint N, CUstream hStream);

    /// <summary>Asynchronously sets sixteen-bit elements in device memory.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemsetD16Async(CUdeviceptr dstDevice, ushort us, nuint N, CUstream hStream);

    /// <summary>Asynchronously sets thirty-two-bit elements in device memory.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemsetD32Async(CUdeviceptr dstDevice, uint ui, nuint N, CUstream hStream);

    /// <summary>Asynchronously sets eight-bit elements in a two-dimensional device-memory region.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemsetD2D8Async(
        CUdeviceptr dstDevice, nuint dstPitch,
        byte uc, nuint Width, nuint Height,
        CUstream hStream);

    /// <summary>Asynchronously sets sixteen-bit elements in a two-dimensional device-memory region.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemsetD2D16Async(
        CUdeviceptr dstDevice, nuint dstPitch,
        ushort us, nuint Width, nuint Height,
        CUstream hStream);

    /// <summary>Asynchronously sets thirty-two-bit elements in a two-dimensional device-memory region.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuMemsetD2D32Async(
        CUdeviceptr dstDevice, nuint dstPitch,
        uint ui, nuint Width, nuint Height,
        CUstream hStream);

    // External resource interoperability

    /// <summary>
    /// Imports an external memory object.
    /// </summary>
    /// <param name="extMem">Returned external memory handle.</param>
    /// <param name="memHandleDesc">Memory handle descriptor.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuImportExternalMemory(
        out CUexternalMemory extMem,
        in CUDA_EXTERNAL_MEMORY_HANDLE_DESC memHandleDesc);

    /// <summary>
    /// Maps a buffer onto an imported memory object.
    /// </summary>
    /// <param name="devPtr">Returned device pointer.</param>
    /// <param name="extMem">External memory handle.</param>
    /// <param name="bufferDesc">Buffer descriptor.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuExternalMemoryGetMappedBuffer(
        out CUdeviceptr devPtr, CUexternalMemory extMem,
        in CUDA_EXTERNAL_MEMORY_BUFFER_DESC bufferDesc);

    /// <summary>
    /// Maps a mipmapped array onto an imported memory object.
    /// </summary>
    /// <param name="mipmappedArray">Returned mipmapped array.</param>
    /// <param name="extMem">External memory handle.</param>
    /// <param name="mipmapDesc">Mipmapped array descriptor.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuExternalMemoryGetMappedMipmappedArray(
        out IntPtr mipmappedArray, CUexternalMemory extMem,
        in CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC mipmapDesc);

    /// <summary>
    /// Destroys an external memory object.
    /// </summary>
    /// <param name="extMem">External memory object to destroy.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuDestroyExternalMemory(CUexternalMemory extMem);

    /// <summary>
    /// Imports an external semaphore.
    /// </summary>
    /// <param name="extSem">Returned external semaphore.</param>
    /// <param name="semHandleDesc">Semaphore descriptor.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuImportExternalSemaphore(
        out CUexternalSemaphore extSem,
        in CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC semHandleDesc);

    /// <summary>
    /// Signals a set of external semaphores.
    /// </summary>
    /// <param name="extSemArray">Array of external semaphores.</param>
    /// <param name="paramsArray">Array of signal parameters.</param>
    /// <param name="numSemaphores">Number of semaphores.</param>
    /// <param name="stream">Stream.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuSignalExternalSemaphoresAsync(
        ReadOnlySpan<CUexternalSemaphore> extSemArray,
        ReadOnlySpan<CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS> paramsArray,
        uint numSemaphores, CUstream stream);

    /// <summary>
    /// Waits on a set of external semaphores.
    /// </summary>
    /// <param name="extSemArray">Array of external semaphores.</param>
    /// <param name="paramsArray">Array of wait parameters.</param>
    /// <param name="numSemaphores">Number of semaphores.</param>
    /// <param name="stream">Stream.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuWaitExternalSemaphoresAsync(
        ReadOnlySpan<CUexternalSemaphore> extSemArray,
        ReadOnlySpan<CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS> paramsArray,
        uint numSemaphores, CUstream stream);

    /// <summary>
    /// Destroys an external semaphore.
    /// </summary>
    /// <param name="extSem">Semaphore to destroy.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuDestroyExternalSemaphore(CUexternalSemaphore extSem);

    // Graphics interoperability

    /// <summary>Unregisters a graphics resource.</summary>
    /// <param name="resource">Graphics resource.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuGraphicsUnregisterResource(CUgraphicsResource resource);

    /// <summary>Maps graphics resources for CUDA access.</summary>
    /// <param name="count">Number of resources.</param>
    /// <param name="resources">Graphics resources.</param>
    /// <param name="stream">Stream in which synchronization is performed.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuGraphicsMapResources(
        uint count, ReadOnlySpan<CUgraphicsResource> resources, CUstream stream);

    /// <summary>Unmaps graphics resources from CUDA access.</summary>
    /// <param name="count">Number of resources.</param>
    /// <param name="resources">Graphics resources.</param>
    /// <param name="stream">Stream in which synchronization is performed.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuGraphicsUnmapResources(
        uint count, ReadOnlySpan<CUgraphicsResource> resources, CUstream stream);

    /// <summary>Sets mapping flags for a graphics resource.</summary>
    /// <param name="resource">Graphics resource.</param>
    /// <param name="flags">Mapping flags.</param>
    [LibraryImport(LibName, EntryPoint = "cuGraphicsResourceSetMapFlags_v2")]
    public static partial CUresult cuGraphicsResourceSetMapFlags(CUgraphicsResource resource, uint flags);

    /// <summary>Returns a device pointer for a mapped graphics resource.</summary>
    /// <param name="devicePointer">Returned device pointer.</param>
    /// <param name="size">Returned mapped size in bytes.</param>
    /// <param name="resource">Mapped graphics resource.</param>
    [LibraryImport(LibName, EntryPoint = "cuGraphicsResourceGetMappedPointer_v2")]
    public static partial CUresult cuGraphicsResourceGetMappedPointer(
        out CUdeviceptr devicePointer, out nuint size, CUgraphicsResource resource);

    /// <summary>Returns an array for a mapped graphics subresource.</summary>
    /// <param name="array">Returned array.</param>
    /// <param name="resource">Mapped graphics resource.</param>
    /// <param name="arrayIndex">Array index.</param>
    /// <param name="mipLevel">Mipmap level.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuGraphicsSubResourceGetMappedArray(
        out CUarray array, CUgraphicsResource resource, uint arrayIndex, uint mipLevel);

    /// <summary>Returns a mipmapped array for a mapped graphics resource.</summary>
    /// <param name="mipmappedArray">Returned mipmapped array.</param>
    /// <param name="resource">Mapped graphics resource.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuGraphicsResourceGetMappedMipmappedArray(
        out CUmipmappedArray mipmappedArray, CUgraphicsResource resource);

    // Stream management

    /// <summary>Creates a CUDA stream.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuStreamCreate(out CUstream pStream, uint Flags);

    /// <summary>Creates a stream with a requested priority.</summary>
    /// <param name="stream">Returned stream.</param>
    /// <param name="flags">Stream creation flags.</param>
    /// <param name="priority">Requested priority.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuStreamCreateWithPriority(
        out CUstream stream, uint flags, int priority);

    /// <summary>Returns the priority of a stream.</summary>
    /// <param name="stream">Stream.</param>
    /// <param name="priority">Returned priority.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuStreamGetPriority(CUstream stream, out int priority);

    /// <summary>Returns the creation flags of a stream.</summary>
    /// <param name="stream">Stream.</param>
    /// <param name="flags">Returned flags.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuStreamGetFlags(CUstream stream, out uint flags);

    /// <summary>Returns the context associated with a stream.</summary>
    /// <param name="stream">Stream.</param>
    /// <param name="context">Returned context.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuStreamGetCtx(CUstream stream, out CUcontext context);

    /// <summary>Queries whether all preceding operations in a stream have completed.</summary>
    /// <param name="stream">Stream.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuStreamQuery(CUstream stream);

    /// <summary>Makes future work in a stream wait for an event.</summary>
    /// <param name="stream">Waiting stream.</param>
    /// <param name="eventHandle">Event to wait for.</param>
    /// <param name="flags">Reserved; must be zero.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuStreamWaitEvent(
        CUstream stream, CUevent eventHandle, uint flags = 0);

    /// <summary>Begins capturing work submitted to a stream.</summary>
    [LibraryImport(LibName, EntryPoint = "cuStreamBeginCapture_v2")]
    public static partial CUresult cuStreamBeginCapture(CUstream hStream, CUstreamCaptureMode mode);

    /// <summary>Ends stream capture and returns the captured graph.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuStreamEndCapture(CUstream hStream, out CUgraph phGraph);

    /// <summary>Waits for all preceding operations in a stream to complete.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuStreamSynchronize(CUstream hStream);

    /// <summary>Destroys a CUDA stream.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuStreamDestroy(CUstream hStream);

    /// <summary>
    /// Wait on a memory location.
    /// </summary>
    /// <param name="stream">Stream.</param>
    /// <param name="addr">Address to wait on.</param>
    /// <param name="value">Value.</param>
    /// <param name="flags">Flags.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuStreamWaitValue32(CUstream stream, CUdeviceptr addr, uint value, uint flags);

    /// <summary>
    /// Wait on a memory location (64-bit).
    /// </summary>
    /// <param name="stream">Stream.</param>
    /// <param name="addr">Address to wait on.</param>
    /// <param name="value">Value.</param>
    /// <param name="flags">Flags.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuStreamWaitValue64(CUstream stream, CUdeviceptr addr, ulong value, uint flags);

    /// <summary>
    /// Write a value to memory.
    /// </summary>
    /// <param name="stream">Stream.</param>
    /// <param name="addr">Address to write to.</param>
    /// <param name="value">Value.</param>
    /// <param name="flags">Flags.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, uint value, uint flags);

    /// <summary>
    /// Write a value to memory (64-bit).
    /// </summary>
    /// <param name="stream">Stream.</param>
    /// <param name="addr">Address to write to.</param>
    /// <param name="value">Value.</param>
    /// <param name="flags">Flags.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuStreamWriteValue64(CUstream stream, CUdeviceptr addr, ulong value, uint flags);

    /// <summary>
    /// Batch memory operations.
    /// </summary>
    /// <param name="stream">Stream.</param>
    /// <param name="count">Count.</param>
    /// <param name="paramArray">Operations.</param>
    /// <param name="flags">Flags.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuStreamBatchMemOp(
        CUstream stream, uint count,
        ReadOnlySpan<CUstreamBatchMemOpParams> paramArray,
        uint flags);

    // Event management

    /// <summary>Creates a CUDA event.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuEventCreate(out CUevent phEvent, uint Flags);

    /// <summary>Destroys a CUDA event.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuEventDestroy(CUevent hEvent);

    /// <summary>Records a CUDA event in a stream.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuEventRecord(CUevent hEvent, CUstream hStream);

    /// <summary>Records an event in a stream with record flags.</summary>
    /// <param name="eventHandle">Event.</param>
    /// <param name="stream">Stream.</param>
    /// <param name="flags">Record flags.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuEventRecordWithFlags(
        CUevent eventHandle, CUstream stream, uint flags);

    /// <summary>Queries whether an event has completed.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuEventQuery(CUevent hEvent);

    /// <summary>Waits for an event to complete.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuEventSynchronize(CUevent hEvent);

    /// <summary>Returns elapsed time in milliseconds between two events.</summary>
    [LibraryImport(LibName)]
    public static partial CUresult cuEventElapsedTime(out float pMilliseconds, CUevent hStart, CUevent hEnd);

    // Graph management

    /// <summary>
    /// Creates a graph.
    /// </summary>
    /// <param name="phGraph">Returned graph.</param>
    /// <param name="flags">Flags.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuGraphCreate(out CUgraph phGraph, uint flags);

    /// <summary>
    /// Destroys a graph.
    /// </summary>
    /// <param name="hGraph">Graph to destroy.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuGraphDestroy(CUgraph hGraph);

    /// <summary>
    /// Adds a kernel node to a graph.
    /// </summary>
    /// <param name="phGraphNode">Returned node.</param>
    /// <param name="hGraph">Graph.</param>
    /// <param name="dependencies">Dependencies.</param>
    /// <param name="numDependencies">Number of dependencies.</param>
    /// <param name="nodeParams">Kernel parameters.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuGraphAddKernelNode(
        out CUgraphNode phGraphNode,
        CUgraph hGraph, ReadOnlySpan<CUgraphNode> dependencies,
        nuint numDependencies,
        in CUDA_KERNEL_NODE_PARAMS nodeParams);

    /// <summary>
    /// Instantiates a graph.
    /// </summary>
    /// <param name="phGraphExec">Returned executable graph.</param>
    /// <param name="hGraph">Graph to instantiate.</param>
    /// <param name="phErrorNode">Error node if failure.</param>
    /// <param name="logBuffer">Log buffer.</param>
    /// <param name="bufferSize">Buffer size.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuGraphInstantiate(
        out CUgraphExec phGraphExec,
        CUgraph hGraph, out CUgraphNode phErrorNode,
        Span<byte> logBuffer, nuint bufferSize);

    /// <summary>
    /// Instantiates a graph with explicit instantiation parameters.
    /// </summary>
    /// <param name="phGraphExec">Returned executable graph.</param>
    /// <param name="hGraph">Graph to instantiate.</param>
    /// <param name="instantiateParams">Instantiation parameters and result details.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuGraphInstantiateWithParams(
        out CUgraphExec phGraphExec,
        CUgraph hGraph,
        ref CUDA_GRAPH_INSTANTIATE_PARAMS instantiateParams);

    /// <summary>
    /// Sets the parameters of a kernel node in an executable graph.
    /// </summary>
    /// <param name="hGraphExec">Executable graph containing the node.</param>
    /// <param name="hNode">Kernel node from the source graph.</param>
    /// <param name="nodeParams">Updated kernel node parameters.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuGraphExecKernelNodeSetParams(
        CUgraphExec hGraphExec,
        CUgraphNode hNode,
        in CUDA_KERNEL_NODE_PARAMS nodeParams);

    /// <summary>
    /// Destroys an executable graph.
    /// </summary>
    /// <param name="hGraphExec">Executable graph to destroy.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuGraphExecDestroy(CUgraphExec hGraphExec);

    /// <summary>
    /// Uploads an executable graph to a stream.
    /// </summary>
    /// <param name="hGraphExec">Executable graph.</param>
    /// <param name="hStream">Stream used for the upload.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream);

    /// <summary>
    /// Launches an executable graph.
    /// </summary>
    /// <param name="hGraphExec">Executable graph.</param>
    /// <param name="hStream">Stream.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream);
}
