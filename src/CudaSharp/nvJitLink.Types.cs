using System.Runtime.InteropServices;

namespace CudaSharp;

/// <summary>
/// NVIDIA JIT Link API.
/// </summary>
/// <seealso href="https://docs.nvidia.com/cuda/nvjitlink/index.html"/>
#pragma warning disable IDE1006 // Naming Styles
public static partial class nvJitLink
#pragma warning restore IDE1006 // Naming Styles
{
    public readonly record struct nvJitLinkHandle(IntPtr Value);

    /// <summary>
    /// nvJitLink result codes.
    /// </summary>
    /// <seealso href="https://docs.nvidia.com/cuda/nvjitlink/index.html#error-codes"/>
    public enum nvJitLinkResult
    {
        /// <summary>The operation completed successfully.</summary>
        NVJITLINK_SUCCESS = 0,
        /// <summary>A linker option was not recognized.</summary>
        NVJITLINK_ERROR_UNRECOGNIZED_OPTION = 1,
        /// <summary>No target architecture option was provided.</summary>
        NVJITLINK_ERROR_MISSING_ARCH = 2,
        /// <summary>An input argument or linker handle is invalid.</summary>
        NVJITLINK_ERROR_INVALID_INPUT = 3,
        /// <summary>PTX compilation failed.</summary>
        NVJITLINK_ERROR_PTX_COMPILE = 4,
        /// <summary>NVVM compilation failed.</summary>
        NVJITLINK_ERROR_NVVM_COMPILE = 5,
        /// <summary>An internal linker error occurred.</summary>
        NVJITLINK_ERROR_INTERNAL = 6,
        /// <summary>The linker thread pool could not be created.</summary>
        NVJITLINK_ERROR_THREADPOOL = 7,
        /// <summary>The input data format was not recognized.</summary>
        NVJITLINK_ERROR_UNRECOGNIZED_INPUT = 8,
        /// <summary>Final output generation failed.</summary>
        NVJITLINK_ERROR_FINALIZE = 9,
        /// <summary>A required input pointer is null.</summary>
        NVJITLINK_ERROR_NULL_INPUT = 10,
        /// <summary>The specified linker options are incompatible.</summary>
        NVJITLINK_ERROR_INCOMPATIBLE_OPTIONS = 11,
        /// <summary>The declared input type does not match the input data.</summary>
        NVJITLINK_ERROR_INCORRECT_INPUT_TYPE = 12,
        /// <summary>An input targets an architecture incompatible with the link target.</summary>
        NVJITLINK_ERROR_ARCH_MISMATCH = 13,
        /// <summary>An input was produced by a newer, incompatible toolkit library.</summary>
        NVJITLINK_ERROR_OUTDATED_LIBRARY = 14,
        /// <summary>A required fat binary is missing.</summary>
        NVJITLINK_ERROR_MISSING_FATBIN = 15,
        /// <summary>The target architecture was not recognized.</summary>
        NVJITLINK_ERROR_UNRECOGNIZED_ARCH = 16,
        /// <summary>The target architecture is recognized but unsupported.</summary>
        NVJITLINK_ERROR_UNSUPPORTED_ARCH = 17,
        /// <summary>An LTO-only operation was requested without enabling LTO.</summary>
        NVJITLINK_ERROR_LTO_NOT_ENABLED = 18,
    }

    /// <summary>
    /// nvJitLink input types.
    /// </summary>
    /// <seealso href="https://docs.nvidia.com/cuda/nvjitlink/index.html#linking"/>
    public enum nvJitLinkInputType
    {
        /// <summary>Invalid input type.</summary>
        NVJITLINK_INPUT_NONE = 0,
        /// <summary>CUDA binary input.</summary>
        NVJITLINK_INPUT_CUBIN = 1,
        /// <summary>PTX input.</summary>
        NVJITLINK_INPUT_PTX = 2,
        /// <summary>LTO-IR container input.</summary>
        NVJITLINK_INPUT_LTOIR = 3,
        /// <summary>CUDA fat binary input.</summary>
        NVJITLINK_INPUT_FATBIN = 4,
        /// <summary>Host object input.</summary>
        NVJITLINK_INPUT_OBJECT = 5,
        /// <summary>Host library input.</summary>
        NVJITLINK_INPUT_LIBRARY = 6,
        /// <summary>Archive index input.</summary>
        NVJITLINK_INPUT_INDEX = 7,
        /// <summary>Detects any supported input type from its contents.</summary>
        NVJITLINK_INPUT_ANY = 10,
    }
}
