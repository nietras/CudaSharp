using System.Buffers;
using System.Text;

namespace CudaSharp;

public static partial class nvJitLink
{
    public unsafe static nvJitLinkResult nvJitLinkCreate(
        out nvJitLinkHandle handle,
        ReadOnlySpan<string> options)
    {
        var optionPointers = stackalloc byte*[options.Length];
        var byteCount = 0;
        for (var i = 0; i < options.Length; i++)
        {
            byteCount = checked(byteCount + Encoding.UTF8.GetByteCount(options[i]) + 1);
        }

        byte[]? pooledBuffer = null;
        Span<byte> buffer = byteCount <= 4096
            ? stackalloc byte[byteCount]
            : (pooledBuffer = ArrayPool<byte>.Shared.Rent(byteCount)).AsSpan(0, byteCount);

        try
        {
            fixed (byte* bufferPointer = buffer)
            {
                var offset = 0;
                for (var i = 0; i < options.Length; i++)
                {
                    optionPointers[i] = bufferPointer + offset;
                    offset += Encoding.UTF8.GetBytes(options[i], buffer[offset..]);
                    buffer[offset++] = 0;
                }

                return nvJitLinkCreate(out handle, (uint)options.Length, optionPointers);
            }
        }
        finally
        {
            if (pooledBuffer is not null)
            {
                ArrayPool<byte>.Shared.Return(pooledBuffer);
            }
        }
    }

    public static unsafe nvJitLinkResult nvJitLinkAddData(
        nvJitLinkHandle handle,
        nvJitLinkInputType inputType,
        ReadOnlySpan<byte> data,
        string? name)
    {
        fixed (byte* dataPointer = data)
        {
            return nvJitLinkAddData(handle, inputType, dataPointer, (nuint)data.Length, name);
        }
    }

    public static byte[] nvJitLinkGetLinkedCubin(nvJitLinkHandle handle) =>
        AllocateOutput(handle, nvJitLinkGetLinkedCubinSize, nvJitLinkGetLinkedCubin);

    public static byte[] nvJitLinkGetLinkedLTOIR(nvJitLinkHandle handle) =>
        AllocateOutput(handle, nvJitLinkGetLinkedLTOIRSize, nvJitLinkGetLinkedLTOIR);

    public static string nvJitLinkGetLinkedPtxString(nvJitLinkHandle handle) =>
        GetUtf8Output(handle, nvJitLinkGetLinkedPtxSize, nvJitLinkGetLinkedPtx);

    public static string nvJitLinkGetErrorLogString(nvJitLinkHandle handle) =>
        GetUtf8Output(handle, nvJitLinkGetErrorLogSize, nvJitLinkGetErrorLog);

    public static string nvJitLinkGetInfoLogString(nvJitLinkHandle handle) =>
        GetUtf8Output(handle, nvJitLinkGetInfoLogSize, nvJitLinkGetInfoLog);

    static byte[] AllocateOutput(
        nvJitLinkHandle handle,
        GetOutputSize getSize,
        GetOutput getOutput)
    {
        getSize(handle, out var size).Ok();
        var output = new byte[checked((int)size)];
        getOutput(handle, output).Ok();
        return output;
    }

    static string GetUtf8Output(
        nvJitLinkHandle handle,
        GetOutputSize getSize,
        GetOutput getOutput)
    {
        getSize(handle, out var size).Ok();
        var length = checked((int)size);
        byte[]? pooledBuffer = null;
        Span<byte> output = length <= 4096
            ? stackalloc byte[length]
            : (pooledBuffer = ArrayPool<byte>.Shared.Rent(length)).AsSpan(0, length);

        try
        {
            getOutput(handle, output).Ok();
            var nullIndex = output.IndexOf((byte)0);
            if (nullIndex >= 0)
            {
                output = output[..nullIndex];
            }

            return Encoding.UTF8.GetString(output);
        }
        finally
        {
            if (pooledBuffer is not null)
            {
                ArrayPool<byte>.Shared.Return(pooledBuffer);
            }
        }
    }

    delegate nvJitLinkResult GetOutputSize(nvJitLinkHandle handle, out nuint size);

    delegate nvJitLinkResult GetOutput(nvJitLinkHandle handle, Span<byte> output);

    extension(nvJitLinkResult result)
    {
        public void Ok()
        {
            if (result != nvJitLinkResult.NVJITLINK_SUCCESS)
            {
                Throws.Throw(result, result.ToStringFast());
            }
        }

        public bool IsOk() => result == nvJitLinkResult.NVJITLINK_SUCCESS;

        public bool IsError() => result != nvJitLinkResult.NVJITLINK_SUCCESS;

        public string ToStringFast() => result switch
        {
            nvJitLinkResult.NVJITLINK_SUCCESS => nameof(nvJitLinkResult.NVJITLINK_SUCCESS),
            nvJitLinkResult.NVJITLINK_ERROR_UNRECOGNIZED_OPTION => nameof(nvJitLinkResult.NVJITLINK_ERROR_UNRECOGNIZED_OPTION),
            nvJitLinkResult.NVJITLINK_ERROR_MISSING_ARCH => nameof(nvJitLinkResult.NVJITLINK_ERROR_MISSING_ARCH),
            nvJitLinkResult.NVJITLINK_ERROR_INVALID_INPUT => nameof(nvJitLinkResult.NVJITLINK_ERROR_INVALID_INPUT),
            nvJitLinkResult.NVJITLINK_ERROR_PTX_COMPILE => nameof(nvJitLinkResult.NVJITLINK_ERROR_PTX_COMPILE),
            nvJitLinkResult.NVJITLINK_ERROR_NVVM_COMPILE => nameof(nvJitLinkResult.NVJITLINK_ERROR_NVVM_COMPILE),
            nvJitLinkResult.NVJITLINK_ERROR_INTERNAL => nameof(nvJitLinkResult.NVJITLINK_ERROR_INTERNAL),
            nvJitLinkResult.NVJITLINK_ERROR_THREADPOOL => nameof(nvJitLinkResult.NVJITLINK_ERROR_THREADPOOL),
            nvJitLinkResult.NVJITLINK_ERROR_UNRECOGNIZED_INPUT => nameof(nvJitLinkResult.NVJITLINK_ERROR_UNRECOGNIZED_INPUT),
            nvJitLinkResult.NVJITLINK_ERROR_FINALIZE => nameof(nvJitLinkResult.NVJITLINK_ERROR_FINALIZE),
            nvJitLinkResult.NVJITLINK_ERROR_NULL_INPUT => nameof(nvJitLinkResult.NVJITLINK_ERROR_NULL_INPUT),
            nvJitLinkResult.NVJITLINK_ERROR_INCOMPATIBLE_OPTIONS => nameof(nvJitLinkResult.NVJITLINK_ERROR_INCOMPATIBLE_OPTIONS),
            nvJitLinkResult.NVJITLINK_ERROR_INCORRECT_INPUT_TYPE => nameof(nvJitLinkResult.NVJITLINK_ERROR_INCORRECT_INPUT_TYPE),
            nvJitLinkResult.NVJITLINK_ERROR_ARCH_MISMATCH => nameof(nvJitLinkResult.NVJITLINK_ERROR_ARCH_MISMATCH),
            nvJitLinkResult.NVJITLINK_ERROR_OUTDATED_LIBRARY => nameof(nvJitLinkResult.NVJITLINK_ERROR_OUTDATED_LIBRARY),
            nvJitLinkResult.NVJITLINK_ERROR_MISSING_FATBIN => nameof(nvJitLinkResult.NVJITLINK_ERROR_MISSING_FATBIN),
            nvJitLinkResult.NVJITLINK_ERROR_UNRECOGNIZED_ARCH => nameof(nvJitLinkResult.NVJITLINK_ERROR_UNRECOGNIZED_ARCH),
            nvJitLinkResult.NVJITLINK_ERROR_UNSUPPORTED_ARCH => nameof(nvJitLinkResult.NVJITLINK_ERROR_UNSUPPORTED_ARCH),
            nvJitLinkResult.NVJITLINK_ERROR_LTO_NOT_ENABLED => nameof(nvJitLinkResult.NVJITLINK_ERROR_LTO_NOT_ENABLED),
            _ => $"NVJITLINK_ERROR_UNKNOWN:{result}",
        };
    }
}
