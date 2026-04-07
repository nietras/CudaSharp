namespace CudaSharp;

/// <summary>
/// NVIDIA JIT Link API.
/// </summary>
/// <seealso href="https://docs.nvidia.com/cuda/nvjitlink/index.html"/>
#pragma warning disable IDE1006 // Naming Styles
public static partial class nvJitLink
#pragma warning restore IDE1006 // Naming Styles
{
    static nvJitLink()
    {
        DllResolver.Register();
    }

    const string LibName = nameof(nvJitLink);

    /// <summary>
    /// Creates an nvJitLink handle.
    /// </summary>
    /// <param name="handle">Returned linker handle.</param>
    /// <param name="numOptions">Number of linker options.</param>
    /// <param name="options">Linker options.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/nvjitlink/index.html#linking"/>
    [LibraryImport(LibName, EntryPoint = "__nvJitLinkCreate_13_1")]
    public static unsafe partial nvJitLinkResult nvJitLinkCreate(
        out nvJitLinkHandle handle,
        uint numOptions,
        byte** options);

    /// <summary>
    /// Destroys an nvJitLink handle.
    /// </summary>
    /// <param name="handle">Handle to destroy.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/nvjitlink/index.html#linking"/>
    [LibraryImport(LibName, EntryPoint = "__nvJitLinkDestroy_13_1")]
    public static partial nvJitLinkResult nvJitLinkDestroy(ref nvJitLinkHandle handle);

    /// <summary>
    /// Adds an in-memory input to an nvJitLink handle.
    /// </summary>
    /// <param name="handle">Linker handle.</param>
    /// <param name="inputType">Input kind.</param>
    /// <param name="data">Input buffer.</param>
    /// <param name="size">Input buffer size in bytes.</param>
    /// <param name="name">Optional input name.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/nvjitlink/index.html#linking"/>
    [LibraryImport(LibName, EntryPoint = "__nvJitLinkAddData_13_1", StringMarshalling = StringMarshalling.Utf8)]
    public static unsafe partial nvJitLinkResult nvJitLinkAddData(
        nvJitLinkHandle handle,
        nvJitLinkInputType inputType,
        void* data,
        nuint size,
        string? name);

    /// <summary>
    /// Adds a file input to an nvJitLink handle.
    /// </summary>
    /// <param name="handle">Linker handle.</param>
    /// <param name="inputType">Input kind.</param>
    /// <param name="fileName">Path to the input file.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/nvjitlink/index.html#linking"/>
    [LibraryImport(LibName, EntryPoint = "__nvJitLinkAddFile_13_1", StringMarshalling = StringMarshalling.Utf8)]
    public static partial nvJitLinkResult nvJitLinkAddFile(
        nvJitLinkHandle handle,
        nvJitLinkInputType inputType,
        string fileName);

    /// <summary>
    /// Completes linking.
    /// </summary>
    /// <param name="handle">Linker handle.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/nvjitlink/index.html#linking"/>
    [LibraryImport(LibName, EntryPoint = "__nvJitLinkComplete_13_1")]
    public static partial nvJitLinkResult nvJitLinkComplete(nvJitLinkHandle handle);

    /// <summary>
    /// Gets the size of the linked cubin.
    /// </summary>
    /// <param name="handle">Linker handle.</param>
    /// <param name="size">Returned cubin size.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/nvjitlink/index.html#linking"/>
    [LibraryImport(LibName, EntryPoint = "__nvJitLinkGetLinkedCubinSize_13_1")]
    public static partial nvJitLinkResult nvJitLinkGetLinkedCubinSize(
        nvJitLinkHandle handle,
        out nuint size);

    /// <summary>
    /// Gets the linked cubin.
    /// </summary>
    /// <param name="handle">Linker handle.</param>
    /// <param name="cubin">Destination cubin buffer.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/nvjitlink/index.html#linking"/>
    [LibraryImport(LibName, EntryPoint = "__nvJitLinkGetLinkedCubin_13_1")]
    public static partial nvJitLinkResult nvJitLinkGetLinkedCubin(
        nvJitLinkHandle handle,
        Span<byte> cubin);

    /// <summary>
    /// Gets the size of the error log.
    /// </summary>
    /// <param name="handle">Linker handle.</param>
    /// <param name="size">Returned error log size.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/nvjitlink/index.html#linking"/>
    [LibraryImport(LibName, EntryPoint = "__nvJitLinkGetErrorLogSize_13_1")]
    public static partial nvJitLinkResult nvJitLinkGetErrorLogSize(
        nvJitLinkHandle handle,
        out nuint size);

    /// <summary>
    /// Gets the error log.
    /// </summary>
    /// <param name="handle">Linker handle.</param>
    /// <param name="log">Destination error log buffer.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/nvjitlink/index.html#linking"/>
    [LibraryImport(LibName, EntryPoint = "__nvJitLinkGetErrorLog_13_1")]
    public static unsafe partial nvJitLinkResult nvJitLinkGetErrorLog(
        nvJitLinkHandle handle,
        byte* log);

    /// <summary>
    /// Gets the size of the info log.
    /// </summary>
    /// <param name="handle">Linker handle.</param>
    /// <param name="size">Returned info log size.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/nvjitlink/index.html#linking"/>
    [LibraryImport(LibName, EntryPoint = "__nvJitLinkGetInfoLogSize_13_1")]
    public static partial nvJitLinkResult nvJitLinkGetInfoLogSize(
        nvJitLinkHandle handle,
        out nuint size);

    /// <summary>
    /// Gets the info log.
    /// </summary>
    /// <param name="handle">Linker handle.</param>
    /// <param name="log">Destination info log buffer.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/nvjitlink/index.html#linking"/>
    [LibraryImport(LibName, EntryPoint = "__nvJitLinkGetInfoLog_13_1")]
    public static unsafe partial nvJitLinkResult nvJitLinkGetInfoLog(
        nvJitLinkHandle handle,
        byte* log);

    /// <summary>
    /// Gets the nvJitLink version.
    /// </summary>
    /// <param name="major">Returned major version.</param>
    /// <param name="minor">Returned minor version.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/nvjitlink/index.html#linking"/>
    [LibraryImport(LibName)]
    public static partial nvJitLinkResult nvJitLinkVersion(out uint major, out uint minor);
}