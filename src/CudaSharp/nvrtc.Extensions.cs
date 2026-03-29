using System.Text;

namespace CudaSharp;

public static partial class nvrtc
{
    public unsafe static ReadOnlySpan<byte> nvrtcGetErrorStringSpan(nvrtcResult result)
    {
        var ptr = (byte*)nvrtcGetErrorString(result);
        return MemoryMarshal.CreateReadOnlySpanFromNullTerminated(ptr);
    }
    internal static string nvrtcGetErrorStringString(nvrtcResult result)
    {
        var span = nvrtcGetErrorStringSpan(result);
        return Encoding.UTF8.GetString(span);
    }

    extension(nvrtcResult result)
    {
        public void Ok()
        {
            if (result != nvrtcResult.NVRTC_SUCCESS)
            {
                Throws.Throw(result, result.ToStringFast());
            }
        }
        public bool IsOk() => result == nvrtcResult.NVRTC_SUCCESS;
        public bool IsError() => result != nvrtcResult.NVRTC_SUCCESS;

        public string ToStringFast() => result switch
        {
            nvrtcResult.NVRTC_SUCCESS => nameof(nvrtcResult.NVRTC_SUCCESS),
            nvrtcResult.NVRTC_ERROR_OUT_OF_MEMORY => nameof(nvrtcResult.NVRTC_ERROR_OUT_OF_MEMORY),
            nvrtcResult.NVRTC_ERROR_PROGRAM_CREATION_FAILURE => nameof(nvrtcResult.NVRTC_ERROR_PROGRAM_CREATION_FAILURE),
            nvrtcResult.NVRTC_ERROR_INVALID_INPUT => nameof(nvrtcResult.NVRTC_ERROR_INVALID_INPUT),
            nvrtcResult.NVRTC_ERROR_INVALID_PROGRAM => nameof(nvrtcResult.NVRTC_ERROR_INVALID_PROGRAM),
            nvrtcResult.NVRTC_ERROR_INVALID_OPTION => nameof(nvrtcResult.NVRTC_ERROR_INVALID_OPTION),
            nvrtcResult.NVRTC_ERROR_COMPILATION => nameof(nvrtcResult.NVRTC_ERROR_COMPILATION),
            nvrtcResult.NVRTC_ERROR_BUILTIN_OPERATION_FAILURE => nameof(nvrtcResult.NVRTC_ERROR_BUILTIN_OPERATION_FAILURE),
            nvrtcResult.NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION => nameof(nvrtcResult.NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION),
            nvrtcResult.NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION => nameof(nvrtcResult.NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION),
            nvrtcResult.NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID => nameof(nvrtcResult.NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID),
            nvrtcResult.NVRTC_ERROR_INTERNAL_ERROR => nameof(nvrtcResult.NVRTC_ERROR_INTERNAL_ERROR),
            nvrtcResult.NVRTC_ERROR_TIME_FILE_WRITE_FAILED => nameof(nvrtcResult.NVRTC_ERROR_TIME_FILE_WRITE_FAILED),
            _ => $"NVRTC_ERROR_UNKNOWN:{result}",
        };
    }
}
