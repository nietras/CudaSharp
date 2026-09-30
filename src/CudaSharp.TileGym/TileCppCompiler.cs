using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using static CudaSharp.nvrtc;

namespace CudaSharp.TileGym;

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

/// <summary>Contains one compiled CUDA TileIR module and the exact lowered names of its kernel expressions.</summary>
/// <param name="TileIr">CUDA TileIR bytecode containing all requested specializations.</param>
/// <param name="EntryPoints">Maps each distinct NVRTC name expression to its copied lowered entry-point name.</param>
/// <seealso href="https://docs.nvidia.com/cuda/nvrtc/index.html#group__compilation" />
public sealed record TileCppCompilationBatch(byte[] TileIr, IReadOnlyDictionary<string, string> EntryPoints);

/// <summary>Compiles CUDA Tile C++ source directly to TileIR or CUBIN with NVRTC.</summary>
/// <remarks>
/// NVRTC and its bundled CUDA headers can be distributed as application dependencies. This type does not inspect or
/// require a machine-wide CUDA Toolkit installation. Header initialization must complete before concurrent compilation.
/// </remarks>
/// <seealso href="https://docs.nvidia.com/cuda/nvrtc/index.html" />
public sealed class TileCppCompiler
{
    readonly string? _bundledHeadersPath;
    bool _headersInstalled;

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
        {
            throw new ArgumentException("A bundled-header path requires bundled-header installation.", nameof(bundledHeadersPath));
        }

        Architecture = architecture;
        _bundledHeadersPath = installBundledHeaders
            ? bundledHeadersPath ?? GetDefaultBundledHeadersPath()
            : null;
    }

    /// <summary>Gets the target SM architecture encoded as major times ten plus minor.</summary>
    public int Architecture { get; }

    /// <summary>Installs bundled CUDA headers before starting concurrent compilations.</summary>
    public void PrepareBundledHeaders()
    {
        using var timing = new TileCppCompilationTiming(nameof(PrepareBundledHeaders));
        timing.Step(nameof(_headersInstalled));
        if (_headersInstalled)
        {
            return;
        }
        if (_bundledHeadersPath is not null)
        {
            timing.Step(nameof(InstallBundledHeaders));
            InstallBundledHeaders(_bundledHeadersPath);
        }
        _headersInstalled = true;
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
        => CompileProgram(source, sourceName, [], config, headers, additionalOptions,
            $"CompileOutput[{sourceName}]", (program, _) => getOutput(program));

    T CompileProgram<T>(string source, string sourceName, string[] nameExpressions, TileCppConfig config,
        IReadOnlyList<TileCppHeader>? headers, IReadOnlyList<string>? additionalOptions,
        string timingLabel, Func<nvrtcProgram, TileCppCompilationTiming, T> getOutput)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourceName);
        ArgumentNullException.ThrowIfNull(config);

        using var timing = new TileCppCompilationTiming(timingLabel);
        timing.Step("Build virtual header buffers");
        var headerSnapshot = headers is null ? [] : headers.ToArray();
        var headerSources = headerSnapshot.Select(static header => header.Source).ToArray();
        var headerNames = headerSnapshot.Select(static header => header.Name).ToArray();
        var optionsSnapshot = additionalOptions?.ToArray();
        timing.Step(nameof(nvrtcCreateProgram));
        nvrtcCreateProgram(out var program, source, sourceName, headerSources.Length, headerSources, headerNames).Ok();
        try
        {
            foreach (var nameExpression in nameExpressions)
            {
                timing.Step(nameof(nvrtcAddNameExpression));
                nvrtcAddNameExpression(program, nameExpression).Ok();
            }
            timing.Step(nameof(CreateNvrtcOptions));
            var options = CreateNvrtcOptions(config, optionsSnapshot);
            timing.Step($"nvrtcCompileProgram[program=0x{program.Value:X}]");
            var result = nvrtcCompileProgram(program, options.Length, options);
            timing.Step("Check compilation result");
            if (result != nvrtcResult.NVRTC_SUCCESS)
            {
                timing.Step(nameof(nvrtcGetProgramLogString));
                var log = nvrtcGetProgramLogString(program);
                var resultName = result.ToStringFast();
                throw new CudaException<nvrtcResult>(result,
                    $"NVRTC CUDA Tile C++ compilation failed with {resultName}:\n{log}");
            }

            timing.Step(nameof(nvrtcGetTileIR));
            return getOutput(program, timing);
        }
        finally
        {
            timing.Step(nameof(nvrtcDestroyProgram));
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
        var compilation = CompileKernelsCore(source, sourceName, [nameExpression], config, headers,
            additionalOptions, $"CompileKernel[{sourceName}, {nameExpression}]");
        return new TileCppCompilation(compilation.TileIr, compilation.EntryPoints[nameExpression]);
    }

    /// <summary>Compiles multiple templated kernel specializations into one CUDA TileIR module.</summary>
    /// <param name="source">CUDA Tile C++ source including any required explicit template instantiations.</param>
    /// <param name="sourceName">Diagnostic source name.</param>
    /// <param name="nameExpressions">Nonempty NVRTC name expressions; exact duplicates are registered only once.</param>
    /// <param name="config">Compile-time kernel parameters and compiler hints shared by the batch.</param>
    /// <param name="headers">Optional virtual headers.</param>
    /// <param name="additionalOptions">Optional additional NVRTC command-line options.</param>
    /// <returns>One TileIR blob and an ordinal map from each distinct expression to its exact lowered name.</returns>
    /// <remarks>All expressions are registered before compilation and all lowered names are copied before destruction.</remarks>
    /// <seealso href="https://docs.nvidia.com/cuda/nvrtc/index.html#group__compilation" />
    public TileCppCompilationBatch CompileKernels(string source, string sourceName,
        ReadOnlySpan<string> nameExpressions, TileCppConfig config,
        IReadOnlyList<TileCppHeader>? headers = null, IReadOnlyList<string>? additionalOptions = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourceName);
        ArgumentNullException.ThrowIfNull(config);
        if (nameExpressions.IsEmpty)
        {
            throw new ArgumentException("At least one name expression is required.", nameof(nameExpressions));
        }

        var expressions = new List<string>(nameExpressions.Length);
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var expression in nameExpressions)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(expression, nameof(nameExpressions));
            if (expression.Contains('\0'))
            {
                throw new ArgumentException("Name expressions cannot contain a null character.", nameof(nameExpressions));
            }
            if (seen.Add(expression))
            {
                expressions.Add(expression);
            }
        }
        var snapshot = expressions.ToArray();
        return CompileKernelsCore(source, sourceName, snapshot, config, headers, additionalOptions,
            $"CompileKernels[{sourceName}, {snapshot.Length} specializations]");
    }

    TileCppCompilationBatch CompileKernelsCore(string source, string sourceName, string[] nameExpressions,
        TileCppConfig config, IReadOnlyList<TileCppHeader>? headers, IReadOnlyList<string>? additionalOptions,
        string timingLabel)
        => CompileProgram(source, sourceName, nameExpressions, config, headers, additionalOptions,
            timingLabel, (program, timing) =>
            {
                var tileIr = nvrtcGetTileIR(program);
                var entryPoints = new Dictionary<string, string>(nameExpressions.Length, StringComparer.Ordinal);
                foreach (var expression in nameExpressions)
                {
                    timing.Step(nameof(nvrtcGetLoweredNameString));
                    var loweredName = nvrtcGetLoweredNameString(program, expression);
                    entryPoints.Add(expression, loweredName);
                }
                timing.Step("Build compilation result");
                var readOnlyEntryPoints = new ReadOnlyDictionary<string, string>(entryPoints);
                return new TileCppCompilationBatch(tileIr, readOnlyEntryPoints);
            });

    string[] CreateNvrtcOptions(TileCppConfig config, IReadOnlyList<string>? additionalOptions)
    {
        PrepareBundledHeaders();

        var options = new List<string>(4 + config.Parameters.Count + (additionalOptions?.Count ?? 0))
        {
            $"--gpu-architecture=sm_{Architecture}",
            "--std=c++20",
            "-enable-tile",
        };
        if (_bundledHeadersPath is not null)
        {
            options.Add($"--include-path={_bundledHeadersPath}");
            var ccclHeadersPath = Path.Combine(_bundledHeadersPath, "cccl");
            options.Add($"--include-path={ccclHeadersPath}");
        }
        foreach (var parameter in config.Parameters)
        {
            options.Add($"-D{parameter.Key}={parameter.Value}");
        }
        if (additionalOptions is not null)
        {
            options.AddRange(additionalOptions);
        }
        return [.. options];
    }

    static void InstallBundledHeaders(string path)
    {
        var result = nvrtcInstallBundledHeaders(path,
            nvrtcInstallHeadersFlags.NVRTC_INSTALL_HEADERS_SKIP_IF_EXISTS, out var errorLog);
        if (result != nvrtcResult.NVRTC_SUCCESS)
        {
            var message = errorLog == IntPtr.Zero ? string.Empty : Marshal.PtrToStringUTF8(errorLog);
            var resultName = result.ToStringFast();
            throw new CudaException<nvrtcResult>(result,
                $"Installing NVRTC bundled headers failed with {resultName}: {message}");
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
