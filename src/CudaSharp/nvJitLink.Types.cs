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
    /// <summary>
    /// Opaque nvJitLink handle.
    /// </summary>
    /// <seealso href="https://docs.nvidia.com/cuda/nvjitlink/index.html#linking"/>
    public readonly record struct nvJitLinkHandle(IntPtr Value);

    /// <summary>
    /// nvJitLink result codes.
    /// </summary>
    /// <seealso href="https://docs.nvidia.com/cuda/nvjitlink/index.html#error-codes"/>
    public enum nvJitLinkResult
    {
        NVJITLINK_SUCCESS = 0,
        NVJITLINK_ERROR_UNRECOGNIZED_OPTION = 1,
        NVJITLINK_ERROR_MISSING_ARCH = 2,
        NVJITLINK_ERROR_INVALID_INPUT = 3,
        NVJITLINK_ERROR_PTX_COMPILE = 4,
        NVJITLINK_ERROR_NVVM_COMPILE = 5,
        NVJITLINK_ERROR_INTERNAL = 6,
        NVJITLINK_ERROR_THREADPOOL = 7,
        NVJITLINK_ERROR_UNRECOGNIZED_INPUT = 8,
        NVJITLINK_ERROR_FINALIZE = 9,
        NVJITLINK_ERROR_NULL_INPUT = 10,
        NVJITLINK_ERROR_INCOMPATIBLE_OPTIONS = 11,
        NVJITLINK_ERROR_INCORRECT_INPUT_TYPE = 12,
        NVJITLINK_ERROR_ARCH_MISMATCH = 13,
        NVJITLINK_ERROR_OUTDATED_LIBRARY = 14,
        NVJITLINK_ERROR_MISSING_FATBIN = 15,
        NVJITLINK_ERROR_UNRECOGNIZED_ARCH = 16,
        NVJITLINK_ERROR_UNSUPPORTED_ARCH = 17,
        NVJITLINK_ERROR_LTO_NOT_ENABLED = 18,
    }

    /// <summary>
    /// nvJitLink input types.
    /// </summary>
    /// <seealso href="https://docs.nvidia.com/cuda/nvjitlink/index.html#linking"/>
    public enum nvJitLinkInputType
    {
        NVJITLINK_INPUT_NONE = 0,
        NVJITLINK_INPUT_CUBIN = 1,
        NVJITLINK_INPUT_PTX = 2,
        NVJITLINK_INPUT_LTOIR = 3,
        NVJITLINK_INPUT_FATBIN = 4,
        NVJITLINK_INPUT_OBJECT = 5,
        NVJITLINK_INPUT_LIBRARY = 6,
        NVJITLINK_INPUT_INDEX = 7,
        NVJITLINK_INPUT_ANY = 10,
    }
}