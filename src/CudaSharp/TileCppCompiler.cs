using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using static CudaSharp.nvcuda;
using static CudaSharp.nvrtc;

namespace CudaSharp.Tile;

/// <summary>Contains a CUDA Tile C++ virtual header supplied to NVRTC.</summary>
/// <seealso href="https://docs.nvidia.com/cuda/nvrtc/index.html#group__compilation" />
public sealed record TileCppHeader
{
    /// <summary>Creates a virtual CUDA Tile C++ header.</summary>
    /// <param name="name">Include name used by CUDA source.</param>
    /// <param name="source">Header source text.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/nvrtc/index.html#group__compilation" />
    public TileCppHeader(string name, string source)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(source);
        Name = name;
        Source = source;
    }

    /// <summary>Gets the virtual include name.</summary>
    public string Name { get; }

    /// <summary>Gets the header source text.</summary>
    public string Source { get; }
}

/// <summary>Contains NVRTC-compiled CUDA TileIR and its lowered kernel entry point.</summary>
/// <seealso href="https://docs.nvidia.com/cuda/nvrtc/index.html#group__compilation" />
public sealed record TileCppCompilation(byte[] TileIr, string EntryPoint);

/// <summary>Compiles CUDA Tile C++ source directly to TileIR or CUBIN with NVRTC.</summary>
/// <remarks>
/// NVRTC and its bundled CUDA headers can be distributed as application dependencies. This type does not inspect or
/// require a machine-wide CUDA Toolkit installation.
/// </remarks>
/// <seealso href="https://docs.nvidia.com/cuda/nvrtc/index.html" />
public sealed class TileCppCompiler
{
    readonly string? _bundledHeadersPath;

    /// <summary>Creates a CUDA Tile C++ compiler for a target GPU architecture.</summary>
    /// <param name="architecture">Target SM architecture encoded as major times ten plus minor.</param>
    /// <param name="installBundledHeaders">
    /// Whether to install and use the CUDA headers embedded in the NVRTC redistributable.
    /// </param>
    /// <param name="bundledHeadersPath">Optional extraction directory for NVRTC's bundled CUDA headers.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/nvrtc/index.html" />
    public TileCppCompiler(int architecture,
        bool installBundledHeaders = true, string? bundledHeadersPath = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(architecture);
        if (!installBundledHeaders && bundledHeadersPath is not null)
            throw new ArgumentException("A bundled-header path requires bundled-header installation.", nameof(bundledHeadersPath));

        Architecture = architecture;
        _bundledHeadersPath = installBundledHeaders
            ? bundledHeadersPath ?? GetDefaultBundledHeadersPath()
            : null;
    }

    /// <summary>Gets the target SM architecture encoded as major times ten plus minor.</summary>
    public int Architecture { get; }

    /// <summary>Queries a CUDA device's target architecture.</summary>
    /// <param name="device">CUDA device handle.</param>
    /// <returns>The SM architecture encoded as major times ten plus minor.</returns>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html" />
    public static int GetArchitecture(CUdevice device)
    {
        cuDeviceGetAttribute(out var major,
            CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device).Ok();
        cuDeviceGetAttribute(out var minor,
            CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device).Ok();
        return checked(major * 10 + minor);
    }

    /// <summary>Compiles CUDA Tile C++ source to TileIR using NVRTC 13.3 or later.</summary>
    /// <param name="source">CUDA Tile C++ source.</param>
    /// <param name="sourceName">Diagnostic source name.</param>
    /// <param name="config">Compile-time kernel parameters and compiler hints.</param>
    /// <param name="headers">Optional virtual headers.</param>
    /// <param name="additionalOptions">Optional additional NVRTC command-line options.</param>
    /// <returns>CUDA TileIR bytecode.</returns>
    /// <seealso href="https://docs.nvidia.com/cuda/nvrtc/index.html" />
    public byte[] CompileToTileIr(string source, string sourceName, TileCppConfig config,
        IReadOnlyList<TileCppHeader>? headers = null, IReadOnlyList<string>? additionalOptions = null)
        => CompileOutput(source, sourceName, config, headers, additionalOptions, static program => nvrtcGetTileIR(program));

    byte[] CompileOutput(string source, string sourceName, TileCppConfig config,
        IReadOnlyList<TileCppHeader>? headers, IReadOnlyList<string>? additionalOptions,
        Func<nvrtcProgram, byte[]> getOutput)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourceName);
        ArgumentNullException.ThrowIfNull(config);

        var headerSources = headers is null ? [] : headers.Select(static header => header.Source).ToArray();
        var headerNames = headers is null ? [] : headers.Select(static header => header.Name).ToArray();
        nvrtcCreateProgram(out var program, source, sourceName, headerSources.Length, headerSources, headerNames).Ok();
        try
        {
            var options = CreateNvrtcOptions(config, additionalOptions);
            var result = nvrtcCompileProgram(program, options.Length, options);
            if (result != nvrtcResult.NVRTC_SUCCESS)
            {
                var log = nvrtcGetProgramLogString(program);
                throw new CudaException<nvrtcResult>(result,
                    $"NVRTC CUDA Tile C++ compilation failed with {result.ToStringFast()}:\n{log}");
            }

            return getOutput(program);
        }
        finally
        {
            nvrtcDestroyProgram(ref program).Ok();
        }
    }

    /// <summary>Compiles CUDA Tile C++ source to TileIR that the CUDA driver can JIT load.</summary>
    /// <param name="source">CUDA Tile C++ source.</param>
    /// <param name="sourceName">Diagnostic source name.</param>
    /// <param name="config">Compile-time kernel parameters and compiler hints.</param>
    /// <param name="headers">Optional virtual headers.</param>
    /// <param name="additionalOptions">Optional additional NVRTC command-line options.</param>
    /// <returns>TileIR bytecode for <see cref="nvcuda.cuModuleLoadData" />.</returns>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html" />
    public byte[] Compile(string source, string sourceName, TileCppConfig config,
        IReadOnlyList<TileCppHeader>? headers = null, IReadOnlyList<string>? additionalOptions = null)
        => CompileToTileIr(source, sourceName, config, headers, additionalOptions);

    /// <summary>Compiles a templated CUDA Tile C++ kernel and returns its lowered TileIR entry point.</summary>
    /// <param name="source">CUDA Tile C++ source including any required explicit template instantiation.</param>
    /// <param name="sourceName">Diagnostic source name.</param>
    /// <param name="nameExpression">NVRTC name expression such as <c>&amp;kernel&lt;float, 64&gt;</c>.</param>
    /// <param name="config">Compile-time kernel parameters and compiler hints.</param>
    /// <param name="headers">Optional virtual headers.</param>
    /// <param name="additionalOptions">Optional additional NVRTC command-line options.</param>
    /// <returns>The TileIR bytecode and lowered kernel entry point.</returns>
    /// <seealso href="https://docs.nvidia.com/cuda/nvrtc/index.html#group__compilation" />
    public TileCppCompilation CompileKernel(string source, string sourceName, string nameExpression,
        TileCppConfig config, IReadOnlyList<TileCppHeader>? headers = null,
        IReadOnlyList<string>? additionalOptions = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(nameExpression);
        ArgumentNullException.ThrowIfNull(source);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourceName);
        ArgumentNullException.ThrowIfNull(config);

        var headerSources = headers is null ? [] : headers.Select(static header => header.Source).ToArray();
        var headerNames = headers is null ? [] : headers.Select(static header => header.Name).ToArray();
        nvrtcCreateProgram(out var program, source, sourceName, headerSources.Length, headerSources, headerNames).Ok();
        try
        {
            nvrtcAddNameExpression(program, nameExpression).Ok();
            var options = CreateNvrtcOptions(config, additionalOptions);
            var result = nvrtcCompileProgram(program, options.Length, options);
            if (result != nvrtcResult.NVRTC_SUCCESS)
                throw new CudaException<nvrtcResult>(result,
                    $"NVRTC CUDA Tile C++ compilation failed with {result.ToStringFast()}:\n{nvrtcGetProgramLogString(program)}");

            return new TileCppCompilation(nvrtcGetTileIR(program), nvrtcGetLoweredNameString(program, nameExpression));
        }
        finally
        {
            nvrtcDestroyProgram(ref program).Ok();
        }
    }

    string[] CreateNvrtcOptions(TileCppConfig config, IReadOnlyList<string>? additionalOptions)
    {
        if (_bundledHeadersPath is not null)
            InstallBundledHeaders(_bundledHeadersPath);

        var options = new List<string>(4 + config.Parameters.Count + (additionalOptions?.Count ?? 0))
        {
            $"--gpu-architecture=sm_{Architecture}",
            "--std=c++20",
            "-enable-tile",
        };
        if (_bundledHeadersPath is not null)
        {
            options.Add($"--include-path={_bundledHeadersPath}");
            options.Add($"--include-path={Path.Combine(_bundledHeadersPath, "cccl")}");
        }
        foreach (var parameter in config.Parameters)
            options.Add($"-D{parameter.Key}={parameter.Value}");
        if (additionalOptions is not null)
            options.AddRange(additionalOptions);
        return [.. options];
    }

    static void InstallBundledHeaders(string path)
    {
        var result = nvrtcInstallBundledHeaders(path,
            nvrtcInstallHeadersFlags.NVRTC_INSTALL_HEADERS_SKIP_IF_EXISTS, out var errorLog);
        if (result != nvrtcResult.NVRTC_SUCCESS)
        {
            var message = errorLog == IntPtr.Zero ? string.Empty : Marshal.PtrToStringUTF8(errorLog);
            throw new CudaException<nvrtcResult>(result,
                $"Installing NVRTC bundled headers failed with {result.ToStringFast()}: {message}");
        }
    }

    static string GetDefaultBundledHeadersPath()
    {
        nvrtcGetBundledHeadersInfo(out var info, out var errorLog).Ok();
        if (info.available == 0)
        {
            var message = errorLog == IntPtr.Zero ? string.Empty : Marshal.PtrToStringUTF8(errorLog);
            throw new CudaException($"The NVRTC redistributable does not contain bundled CUDA headers. {message}");
        }

        var root = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        return Path.Combine(root, "CudaSharp", "nvrtc-headers", $"{info.cudaVersionMajor}.{info.cudaVersionMinor}");
    }
}
