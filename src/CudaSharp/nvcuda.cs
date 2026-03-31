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
    /// Returns an identifer string for the device.
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

    /// <summary>
    /// Create a CUDA context.
    /// </summary>
    /// <param name="pctx">Returned context handle.</param>
    /// <param name="flags">Context creation flags.</param>
    /// <param name="dev">Device to create context on.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuCtxCreate(out CUcontext pctx, CUctx_flags flags, CUdevice dev);

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

    /// <summary>
    /// Returns a function handle.
    /// </summary>
    /// <param name="hfunc">Returned function handle.</param>
    /// <param name="hmod">Module to retrieve function from.</param>
    /// <param name="name">Name of function to retrieve.</param>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial CUresult cuModuleGetFunction(out CUfunction hfunc, CUmodule hmod, string name);

    /// <summary>
    /// Unloads a module.
    /// </summary>
    /// <param name="hmod">Module to unload.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuModuleUnload(CUmodule hmod);

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

    [LibraryImport(LibName)]
    public unsafe static partial CUresult cuLaunchKernel(CUfunction f,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes, CUstream hStream,
        void** kernelParams, void** extra);

    [LibraryImport(LibName)]
    public unsafe static partial CUresult cuLaunchKernelEx(
        in CUlaunchConfig config,
        CUfunction f,
        void** kernelParams,
        void** extra);

    [LibraryImport(LibName)]
    public static partial CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(
        out int numBlocks, CUfunction func,
        int blockSize, nuint dynamicSMemSize);

    [LibraryImport(LibName)]
    public static partial CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        out int numBlocks, CUfunction func,
        int blockSize, nuint dynamicSMemSize, uint flags);

    [LibraryImport(LibName)]
    public static partial CUresult cuMemAlloc(out CUdeviceptr dptr, nuint bytesize);

    [LibraryImport(LibName)]
    public static partial CUresult cuMemFree(CUdeviceptr dptr);

    [LibraryImport(LibName)]
    public static partial CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, IntPtr srcHost, nuint bytesize);

    [LibraryImport(LibName)]
    public static partial CUresult cuMemcpyDtoH(IntPtr dstHost, CUdeviceptr srcDevice, nuint bytesize);

    [LibraryImport(LibName)]
    public static partial CUresult cuMemcpyHtoDAsync(
        CUdeviceptr dstDevice, IntPtr srcHost,
        nuint bytesize, CUstream hStream);

    [LibraryImport(LibName)]
    public static partial CUresult cuMemcpyDtoHAsync(
        IntPtr dstHost, CUdeviceptr srcDevice,
        nuint bytesize, CUstream hStream);

    [LibraryImport(LibName)]
    public static partial CUresult cuMemcpyDtoDAsync(
        CUdeviceptr dstDevice, CUdeviceptr srcDevice,
        nuint bytesize, CUstream hStream);

    [LibraryImport(LibName)]
    public static partial CUresult cuMemAllocPitch(
        out CUdeviceptr dptr, out nuint pPitch,
        nuint WidthInBytes, nuint Height, uint ElementSizeBytes);

    [LibraryImport(LibName)]
    public static partial CUresult cuMemFreeHost(IntPtr p);

    [LibraryImport(LibName)]
    public static partial CUresult cuMemHostAlloc(out IntPtr pp, nuint bytesize, uint Flags);

    [LibraryImport(LibName)]
    public static partial CUresult cuMemGetInfo(out nuint free, out nuint total);

    [LibraryImport(LibName)]
    public static partial CUresult cuMemcpy2D(in CUDA_MEMCPY2D pCopy);

    [LibraryImport(LibName)]
    public static partial CUresult cuMemcpy3D(in CUDA_MEMCPY3D pCopy);

    [LibraryImport(LibName)]
    public static partial CUresult cuMemcpy2DAsync(in CUDA_MEMCPY2D pCopy, CUstream hStream);

    [LibraryImport(LibName)]
    public static partial CUresult cuMemcpy3DAsync(in CUDA_MEMCPY3D pCopy, CUstream hStream);

    [LibraryImport(LibName)]
    public static partial CUresult cuMemsetD8(CUdeviceptr dstDevice, byte uc, nuint N);

    [LibraryImport(LibName)]
    public static partial CUresult cuMemsetD16(CUdeviceptr dstDevice, ushort us, nuint N);

    [LibraryImport(LibName)]
    public static partial CUresult cuMemsetD32(CUdeviceptr dstDevice, uint ui, nuint N);

    [LibraryImport(LibName)]
    public static partial CUresult cuMemsetD2D8(
        CUdeviceptr dstDevice, nuint dstPitch,
        byte uc, nuint Width, nuint Height);

    [LibraryImport(LibName)]
    public static partial CUresult cuMemsetD2D16(
        CUdeviceptr dstDevice, nuint dstPitch,
        ushort us, nuint Width, nuint Height);

    [LibraryImport(LibName)]
    public static partial CUresult cuMemsetD2D32(
        CUdeviceptr dstDevice, nuint dstPitch,
        uint ui, nuint Width, nuint Height);

    [LibraryImport(LibName)]
    public static partial CUresult cuMemsetD8Async(CUdeviceptr dstDevice, byte uc, nuint N, CUstream hStream);

    [LibraryImport(LibName)]
    public static partial CUresult cuMemsetD16Async(CUdeviceptr dstDevice, ushort us, nuint N, CUstream hStream);

    [LibraryImport(LibName)]
    public static partial CUresult cuMemsetD32Async(CUdeviceptr dstDevice, uint ui, nuint N, CUstream hStream);

    [LibraryImport(LibName)]
    public static partial CUresult cuMemsetD2D8Async(
        CUdeviceptr dstDevice, nuint dstPitch,
        byte uc, nuint Width, nuint Height,
        CUstream hStream);

    [LibraryImport(LibName)]
    public static partial CUresult cuMemsetD2D16Async(
        CUdeviceptr dstDevice, nuint dstPitch,
        ushort us, nuint Width, nuint Height,
        CUstream hStream);

    [LibraryImport(LibName)]
    public static partial CUresult cuMemsetD2D32Async(
        CUdeviceptr dstDevice, nuint dstPitch,
        uint ui, nuint Width, nuint Height,
        CUstream hStream);

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
    /// Launches an executable graph.
    /// </summary>
    /// <param name="hGraphExec">Executable graph.</param>
    /// <param name="hStream">Stream.</param>
    [LibraryImport(LibName)]
    public static partial CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream);

    [LibraryImport(LibName)]
    public static partial CUresult cuCtxSynchronize();

    [LibraryImport(LibName)]
    public static partial CUresult cuStreamCreate(out CUstream pStream, uint Flags);

    [LibraryImport(LibName)]
    public static partial CUresult cuStreamSynchronize(CUstream hStream);

    [LibraryImport(LibName)]
    public static partial CUresult cuStreamDestroy(CUstream hStream);

    [LibraryImport(LibName)]
    public static partial CUresult cuEventCreate(out CUevent phEvent, uint Flags);

    [LibraryImport(LibName)]
    public static partial CUresult cuEventDestroy(CUevent hEvent);

    [LibraryImport(LibName)]
    public static partial CUresult cuEventRecord(CUevent hEvent, CUstream hStream);

    [LibraryImport(LibName)]
    public static partial CUresult cuEventQuery(CUevent hEvent);

    [LibraryImport(LibName)]
    public static partial CUresult cuEventSynchronize(CUevent hEvent);

    [LibraryImport(LibName)]
    public static partial CUresult cuEventElapsedTime(out float pMilliseconds, CUevent hStart, CUevent hEnd);
}
