namespace CudaSharp;

public static partial class nvrtc
{
    /// <summary>Result codes returned by NVRTC API functions.</summary>
    public enum nvrtcResult
    {
        /// <summary>The API call completed successfully.</summary>
        NVRTC_SUCCESS = 0,
        /// <summary>NVRTC could not allocate enough host memory.</summary>
        NVRTC_ERROR_OUT_OF_MEMORY = 1,
        /// <summary>NVRTC could not create the program.</summary>
        NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
        /// <summary>An input argument is invalid.</summary>
        NVRTC_ERROR_INVALID_INPUT = 3,
        /// <summary>The program handle is invalid.</summary>
        NVRTC_ERROR_INVALID_PROGRAM = 4,
        /// <summary>A compiler option is invalid.</summary>
        NVRTC_ERROR_INVALID_OPTION = 5,
        /// <summary>Compilation failed; retrieve the program log for details.</summary>
        NVRTC_ERROR_COMPILATION = 6,
        /// <summary>An operation involving the built-in headers failed.</summary>
        NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
        /// <summary>A name expression was added after the program had been compiled.</summary>
        NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
        /// <summary>A lowered name was requested before the program had been compiled.</summary>
        NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
        /// <summary>The supplied name expression is invalid.</summary>
        NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
        /// <summary>An internal NVRTC error occurred.</summary>
        NVRTC_ERROR_INTERNAL_ERROR = 11,
        /// <summary>NVRTC could not write a requested compilation-time output file.</summary>
        NVRTC_ERROR_TIME_FILE_WRITE_FAILED = 12,
        /// <summary>No precompiled-header creation was attempted for the program.</summary>
        NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED = 13,
        /// <summary>The precompiled-header heap was too small to create the PCH.</summary>
        NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED = 14,
        /// <summary>An error prevented creation of the precompiled header.</summary>
        NVRTC_ERROR_PCH_CREATE = 15,
        /// <summary>Compilation was cancelled by the registered flow callback.</summary>
        NVRTC_ERROR_CANCELLED = 16,
        /// <summary>NVRTC could not write a requested time-trace output file.</summary>
        NVRTC_ERROR_TIME_TRACE_FILE_WRITE_FAILED = 17,
        /// <summary>The program is already being compiled or used by another operation.</summary>
        NVRTC_ERROR_BUSY = 18,
    }

    /// <summary>Controls installation of the CUDA headers bundled with NVRTC.</summary>
    [Flags]
    public enum nvrtcInstallHeadersFlags : uint
    {
        /// <summary>Skips installation when matching bundled headers already exist.</summary>
        NVRTC_INSTALL_HEADERS_SKIP_IF_EXISTS = 0,
        /// <summary>Clears existing directory contents before installing the bundled headers.</summary>
        NVRTC_INSTALL_HEADERS_FORCE_OVERWRITE = 1,
        /// <summary>Returns <see cref="nvrtcResult.NVRTC_ERROR_BUSY"/> instead of waiting for another installation.</summary>
        NVRTC_INSTALL_HEADERS_NO_WAIT = 2,
    }

    /// <summary>Opaque NVRTC program handle.</summary>
    public readonly record struct nvrtcProgram(IntPtr Value);

    /// <summary>Describes the built-in headers bundled with NVRTC.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct nvrtcBundledHeadersInfo
    {
        /// <summary>Indicates whether bundled headers are available.</summary>
        public int available;
        /// <summary>Compressed size of the bundled headers in bytes.</summary>
        public nuint compressedSize;
        /// <summary>Estimated uncompressed size of the bundled headers in bytes.</summary>
        public nuint uncompressedSize;
        /// <summary>Major CUDA version of the bundled headers.</summary>
        public int cudaVersionMajor;
        /// <summary>Minor CUDA version of the bundled headers.</summary>
        public int cudaVersionMinor;
        /// <summary>Number of files in the bundled-header archive.</summary>
        public uint numFiles;
    }

    /// <summary>Callback invoked by NVRTC during compilation.</summary>
    /// <param name="payload">Compiler-provided callback payload.</param>
    /// <param name="userData">User data registered with <see cref="nvrtcSetFlowCallback"/>.</param>
    /// <returns>Zero to continue compilation; nonzero to cancel it.</returns>
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate int nvrtcFlowCallback(IntPtr payload, IntPtr userData);
}
