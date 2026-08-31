using System.Text;
using System.Runtime.InteropServices;

namespace CudaSharp;

public static partial class nvrtc
{
    /// <summary>Returns the null-terminated NVRTC error text as UTF-8 bytes.</summary>
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

    /// <summary>Returns the virtual architectures supported by NVRTC.</summary>
    public static int[] nvrtcGetSupportedArchs()
    {
        nvrtcGetNumSupportedArchs(out var count).Ok();
        var architectures = GC.AllocateUninitializedArray<int>(count);
        nvrtcGetSupportedArchs(architectures).Ok();
        return architectures;
    }

    /// <summary>Compiles a program using managed compiler-option strings.</summary>
    public static unsafe nvrtcResult nvrtcCompileProgram(
        nvrtcProgram program,
        int numOptions,
        string[] options)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentOutOfRangeException.ThrowIfNegative(numOptions);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(numOptions, options.Length);

        var optionPointers = stackalloc byte*[numOptions];
        try
        {
            for (var i = 0; i < numOptions; i++)
                optionPointers[i] = (byte*)Marshal.StringToCoTaskMemUTF8(options[i]);

            return nvrtcCompileProgram(program, numOptions, optionPointers);
        }
        finally
        {
            for (var i = 0; i < numOptions; i++)
                Marshal.FreeCoTaskMem((IntPtr)optionPointers[i]);
        }
    }

    /// <summary>Returns the PTX generated for a compiled program.</summary>
    public static byte[] nvrtcGetPTX(nvrtcProgram program)
    {
        nvrtcGetPTXSize(program, out var size).Ok();
        var output = AllocateOutput(size);
        nvrtcGetPTX(program, output).Ok();
        return output;
    }

    /// <summary>Returns the CUBIN generated for a compiled program.</summary>
    public static byte[] nvrtcGetCUBIN(nvrtcProgram program)
    {
        nvrtcGetCUBINSize(program, out var size).Ok();
        var output = AllocateOutput(size);
        nvrtcGetCUBIN(program, output).Ok();
        return output;
    }

    /// <summary>Returns the LTO IR generated for a compiled program.</summary>
    public static byte[] nvrtcGetLTOIR(nvrtcProgram program)
    {
        nvrtcGetLTOIRSize(program, out var size).Ok();
        var output = AllocateOutput(size);
        nvrtcGetLTOIR(program, output).Ok();
        return output;
    }

    /// <summary>Returns the OptiX IR generated for a compiled program.</summary>
    public static byte[] nvrtcGetOptiXIR(nvrtcProgram program)
    {
        nvrtcGetOptiXIRSize(program, out var size).Ok();
        var output = AllocateOutput(size);
        nvrtcGetOptiXIR(program, output).Ok();
        return output;
    }

    /// <summary>Returns the TileIR generated for a CUDA Tile C++ program.</summary>
    public static byte[] nvrtcGetTileIR(nvrtcProgram program)
    {
        nvrtcGetTileIRSize(program, out var size).Ok();
        var output = AllocateOutput(size);
        nvrtcGetTileIR(program, output).Ok();
        return output;
    }

    /// <summary>Returns the compilation log for a program.</summary>
    public static string nvrtcGetProgramLogString(nvrtcProgram program)
    {
        nvrtcGetProgramLogSize(program, out var size).Ok();
        var output = AllocateOutput(size);
        nvrtcGetProgramLog(program, output).Ok();
        return Encoding.UTF8.GetString(output).TrimEnd('\0');
    }

    /// <summary>Returns the lowered name for a previously added name expression.</summary>
    public static unsafe string nvrtcGetLoweredNameString(nvrtcProgram program, string nameExpression)
    {
        nvrtcGetLoweredName(program, nameExpression, out var loweredName).Ok();
        return Encoding.UTF8.GetString(MemoryMarshal.CreateReadOnlySpanFromNullTerminated((byte*)loweredName));
    }

    static byte[] AllocateOutput(nuint size) => GC.AllocateUninitializedArray<byte>(checked((int)size));

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
            nvrtcResult.NVRTC_ERROR_TIME_TRACE_FILE_WRITE_FAILED => nameof(nvrtcResult.NVRTC_ERROR_TIME_TRACE_FILE_WRITE_FAILED),
            nvrtcResult.NVRTC_ERROR_PCH_CREATE => nameof(nvrtcResult.NVRTC_ERROR_PCH_CREATE),
            nvrtcResult.NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED => nameof(nvrtcResult.NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED),
            nvrtcResult.NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED => nameof(nvrtcResult.NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED),
            nvrtcResult.NVRTC_ERROR_CANCELLED => nameof(nvrtcResult.NVRTC_ERROR_CANCELLED),
            nvrtcResult.NVRTC_ERROR_BUSY => nameof(nvrtcResult.NVRTC_ERROR_BUSY),
            _ => $"NVRTC_ERROR_UNKNOWN:{result}",
        };
    }
}
