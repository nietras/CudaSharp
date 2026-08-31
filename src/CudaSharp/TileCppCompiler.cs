using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using static CudaSharp.nvcuda;
using static CudaSharp.nvrtc;

namespace CudaSharp.Tile;

/// <summary>Converts CUDA TileIR bytecode into a loadable CUBIN image.</summary>
/// <remarks>
/// NVIDIA's CUDA Tile compiler uses the <c>tileiras</c> component for this stage. Implementations can bind a
/// NuGet-distributed compiler component without requiring a CUDA Toolkit installation.
/// </remarks>
/// <seealso href="https://docs.nvidia.com/cuda/cutile-python/debugging.html" />
public interface ITileIrAssembler
{
    /// <summary>Compiles TileIR bytecode for a target architecture.</summary>
    /// <param name="tileIr">TileIR bytecode emitted by NVRTC.</param>
    /// <param name="options">Target and compiler-hint options.</param>
    /// <returns>A loadable CUBIN image.</returns>
    /// <seealso href="https://docs.nvidia.com/cuda/cutile-python/compilation.html" />
    byte[] Assemble(ReadOnlySpan<byte> tileIr, TileCppAssemblerOptions options);
}

/// <summary>Specifies options for compiling CUDA TileIR to CUBIN.</summary>
/// <seealso href="https://docs.nvidia.com/cuda/cutile-python/execution.html#cuda.tile.kernel" />
public sealed record TileCppAssemblerOptions
{
    /// <summary>Creates TileIR assembler options from a kernel configuration.</summary>
    /// <param name="architecture">Target SM architecture encoded as major times ten plus minor.</param>
    /// <param name="config">Kernel configuration containing Tile compiler hints.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/cutile-python/execution.html#cuda.tile.kernel" />
    public TileCppAssemblerOptions(int architecture, TileCppConfig config)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(architecture);
        ArgumentNullException.ThrowIfNull(config);
        Architecture = architecture;
        NumCtas = config.NumCtas;
        Occupancy = config.Occupancy;
        OptimizationLevel = config.OptimizationLevel;
        NumWorkerWarps = config.NumWorkerWarps;
    }

    /// <summary>Gets the target SM architecture encoded as major times ten plus minor.</summary>
    public int Architecture { get; }

    /// <summary>Gets the number of CTAs in a cluster, or <see langword="null" /> for automatic selection.</summary>
    public int? NumCtas { get; }

    /// <summary>Gets the expected active CTAs per SM, or <see langword="null" /> for automatic selection.</summary>
    public int? Occupancy { get; }

    /// <summary>Gets the Tile compiler optimization level.</summary>
    public int OptimizationLevel { get; }

    /// <summary>Gets the CUDA core worker warp count, or <see langword="null" /> for automatic selection.</summary>
    public int? NumWorkerWarps { get; }
}

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

/// <summary>Compiles CUDA Tile C++ source with NVRTC and a caller-provided TileIR assembler.</summary>
/// <remarks>
/// NVRTC and its bundled CUDA headers can be distributed as application dependencies. This type does not inspect or
/// require a machine-wide CUDA Toolkit installation.
/// </remarks>
/// <seealso href="https://docs.nvidia.com/cuda/nvrtc/index.html" />
public sealed class TileCppCompiler
{
    readonly ITileIrAssembler _assembler;
    readonly string? _bundledHeadersPath;

    /// <summary>Creates a CUDA Tile C++ compiler for a target GPU architecture.</summary>
    /// <param name="assembler">TileIR-to-CUBIN compiler backend supplied by the application.</param>
    /// <param name="architecture">Target SM architecture encoded as major times ten plus minor.</param>
    /// <param name="installBundledHeaders">
    /// Whether to install and use the CUDA headers embedded in the NVRTC redistributable.
    /// </param>
    /// <param name="bundledHeadersPath">Optional extraction directory for NVRTC's bundled CUDA headers.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/nvrtc/index.html" />
    public TileCppCompiler(ITileIrAssembler assembler, int architecture,
        bool installBundledHeaders = true, string? bundledHeadersPath = null)
    {
        ArgumentNullException.ThrowIfNull(assembler);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(architecture);
        if (!installBundledHeaders && bundledHeadersPath is not null)
            throw new ArgumentException("A bundled-header path requires bundled-header installation.", nameof(bundledHeadersPath));

        _assembler = assembler;
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

            return nvrtcGetTileIR(program);
        }
        finally
        {
            nvrtcDestroyProgram(ref program).Ok();
        }
    }

    /// <summary>Compiles CUDA Tile C++ source to a loadable CUBIN image.</summary>
    /// <param name="source">CUDA Tile C++ source.</param>
    /// <param name="sourceName">Diagnostic source name.</param>
    /// <param name="config">Compile-time kernel parameters and compiler hints.</param>
    /// <param name="headers">Optional virtual headers.</param>
    /// <param name="additionalOptions">Optional additional NVRTC command-line options.</param>
    /// <returns>A CUBIN image for <see cref="nvcuda.cuModuleLoadData" />.</returns>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html" />
    public byte[] Compile(string source, string sourceName, TileCppConfig config,
        IReadOnlyList<TileCppHeader>? headers = null, IReadOnlyList<string>? additionalOptions = null)
    {
        var tileIr = CompileToTileIr(source, sourceName, config, headers, additionalOptions);
        return _assembler.Assemble(tileIr, new TileCppAssemblerOptions(Architecture, config));
    }

    string[] CreateNvrtcOptions(TileCppConfig config, IReadOnlyList<string>? additionalOptions)
    {
        if (_bundledHeadersPath is not null)
            InstallBundledHeaders(_bundledHeadersPath);

        var options = new List<string>(4 + config.Parameters.Count + (additionalOptions?.Count ?? 0))
        {
            $"--gpu-architecture=compute_{Architecture}",
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
