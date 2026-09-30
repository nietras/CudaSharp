using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using static CudaSharp.nvcuda;

namespace CudaSharp.TileGym;

/// <summary>Compiles and loads variants of a CUDA Tile C++ kernel.</summary>
/// <remarks>
/// Loaded modules are cached per CUDA context. Dispose this object before destroying any context in which a variant
/// was loaded. CUDA Tile kernels are launched with one thread per tile block as required by NVIDIA.
/// </remarks>
/// <seealso href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#launching-kernels" />
public sealed class TileCppKernel : IDisposable
{
    readonly ConcurrentDictionary<string, Lazy<TileCppCompilation>> _compilations = new(StringComparer.Ordinal);
    readonly Dictionary<LoadedKey, LoadedKernel> _loadedKernels = [];
    readonly Lock _lock = new();
    readonly TileCppCompiler _compiler;
    readonly string _source;
    readonly string _sourceName;
    readonly string _kernelName;
    readonly string? _nameExpression;
    readonly IReadOnlyList<TileCppHeader>? _headers;
    readonly IReadOnlyList<string>? _additionalOptions;
    volatile bool _disposed;

    /// <summary>Creates a reusable CUDA Tile C++ kernel definition.</summary>
    /// <param name="compiler">CUDA Tile C++ compiler.</param>
    /// <param name="source">CUDA Tile C++ source text.</param>
    /// <param name="sourceName">Diagnostic source name.</param>
    /// <param name="kernelName">Unmangled kernel entry-point name.</param>
    /// <param name="headers">Optional virtual headers.</param>
    /// <param name="additionalOptions">Optional additional NVRTC command-line options.</param>
    /// <param name="nameExpression">Optional NVRTC name expression for a templated kernel specialization.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#writing-tile-kernels" />
    public TileCppKernel(TileCppCompiler compiler, string source, string sourceName, string kernelName,
        IReadOnlyList<TileCppHeader>? headers = null, IReadOnlyList<string>? additionalOptions = null,
        string? nameExpression = null)
    {
        ArgumentNullException.ThrowIfNull(compiler);
        ArgumentNullException.ThrowIfNull(source);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourceName);
        ArgumentException.ThrowIfNullOrWhiteSpace(kernelName);
        _compiler = compiler;
        _source = source;
        _sourceName = sourceName;
        _kernelName = kernelName;
        _nameExpression = nameExpression;
        _headers = headers;
        _additionalOptions = additionalOptions;
    }

    /// <summary>Compiles a configuration to TileIR without requiring a CUDA context or loading a module.</summary>
    /// <returns>The TileIR size in bytes.</returns>
    public int Compile(TileCppConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        ObjectDisposedException.ThrowIf(_disposed, this);
        using var timing = new TileCppCompilationTiming($"TileCppKernel.Compile[{_sourceName}, {_nameExpression ?? _kernelName}]");
        timing.Step(nameof(GetConfigKey));
        var configKey = GetConfigKey(config);
        timing.Step($"GetOrCompile[{configKey}]");
        var compilation = GetOrCompile(config, configKey);
        timing.Step("Result validation");
        ObjectDisposedException.ThrowIf(_disposed, this);
        return compilation.TileIr.Length;
    }

    /// <summary>Loads a previously compiled configuration in the current CUDA context.</summary>
    public CUfunction LoadFunction(TileCppConfig config)
        => GetFunctionCore(config, compileIfNeeded: false);

    /// <summary>Gets the loaded CUDA function for a configuration in the current CUDA context.</summary>
    /// <param name="config">Compile-time kernel configuration.</param>
    /// <returns>A CUDA function handle for the current context.</returns>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html" />
    public CUfunction GetFunction(TileCppConfig config)
        => GetFunctionCore(config, compileIfNeeded: true);

    CUfunction GetFunctionCore(TileCppConfig config, bool compileIfNeeded)
    {
        ArgumentNullException.ThrowIfNull(config);
        ObjectDisposedException.ThrowIf(_disposed, this);
        cuCtxGetCurrent(out var context).Ok();
        if (context.Value == IntPtr.Zero)
        {
            throw new InvalidOperationException("A CUDA context must be current before loading a CUDA Tile C++ kernel.");
        }

        var configKey = GetConfigKey(config);
        var loadedKey = new LoadedKey(configKey, context);
        lock (_lock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_loadedKernels.TryGetValue(loadedKey, out var loaded))
            {
                return loaded.Function;
            }

            TileCppCompilation compilation;
            if (compileIfNeeded)
            {
                compilation = GetOrCompile(config, configKey);
            }
            else
            {
                if (!_compilations.TryGetValue(configKey, out var completed) || !completed.IsValueCreated)
                {
                    throw new InvalidOperationException("Compile the configuration before loading its CUDA function.");
                }
                compilation = completed.Value;
            }

            cuModuleLoadData(out var module, compilation.TileIr).Ok();
            try
            {
                var lookup = cuModuleGetFunction(out var function, module, compilation.EntryPoint);
                if (lookup == CUresult.CUDA_ERROR_NOT_FOUND)
                {
                    function = FindFunction(module, _kernelName);
                }
                else
                {
                    lookup.Ok();
                }
                _loadedKernels.Add(loadedKey, new LoadedKernel(module, function));
                return function;
            }
            catch
            {
                cuModuleUnload(module).Ok();
                throw;
            }
        }
    }

    TileCppCompilation GetOrCompile(TileCppConfig config, string configKey)
    {
        using var timing = new TileCppCompilationTiming($"GetOrCompile[{_sourceName}, {configKey}]");
        timing.Step("_compilations.GetOrAdd");
        var lazy = _compilations.GetOrAdd(configKey, _ => new Lazy<TileCppCompilation>(() =>
        {
            TileCppCompilation compilation;
            if (_nameExpression is null)
            {
                var tileIr = _compiler.Compile(_source, _sourceName, config, _headers, _additionalOptions);
                compilation = new TileCppCompilation(tileIr, _kernelName);
            }
            else
            {
                compilation = _compiler.CompileKernel(
                    _source, _sourceName, _nameExpression, config, _headers, _additionalOptions);
            }
            if (compilation.TileIr.Length == 0)
            {
                throw new InvalidOperationException("NVRTC returned empty CUDA TileIR.");
            }
            return compilation;
        }, LazyThreadSafetyMode.ExecutionAndPublication));
        timing.Step("Lazy.Value (inclusive of compiler stages)");
        try
        {
            return lazy.Value;
        }
        catch
        {
            timing.Step("Remove failed compilation");
            ((ICollection<KeyValuePair<string, Lazy<TileCppCompilation>>>)_compilations)
                .Remove(new(configKey, lazy));
            throw;
        }
    }

    static CUfunction FindFunction(CUmodule module, string expectedName)
    {
        cuModuleGetFunctionCount(out var count, module).Ok();
        var functions = new CUfunction[count];
        cuModuleEnumerateFunctions(functions, count, module).Ok();
        var names = new List<string>(functions.Length);
        foreach (var function in functions)
        {
            cuFuncGetName(out var namePointer, function).Ok();
            var name = Marshal.PtrToStringUTF8(namePointer);
            if (name is not null)
            {
                names.Add(name);
            }
            if (name?.Contains(expectedName, StringComparison.Ordinal) is true)
            {
                return function;
            }
        }
        var functionNames = string.Join(", ", names);
        throw new InvalidOperationException(
            $"CUDA Tile C++ kernel '{expectedName}' was not found in the driver-loaded TileIR module. Functions: {functionNames}");
    }

    /// <summary>Unloads all context-specific CUDA modules owned by this kernel.</summary>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html" />
    public void Dispose()
    {
        lock (_lock)
        {
            if (_disposed)
            {
                return;
            }

            foreach (var entry in _loadedKernels)
            {
                cuCtxGetCurrent(out var previousContext).Ok();
                if (previousContext != entry.Key.Context)
                {
                    cuCtxSetCurrent(entry.Key.Context).Ok();
                }
                try
                {
                    cuModuleUnload(entry.Value.Module).Ok();
                }
                finally
                {
                    if (previousContext != entry.Key.Context)
                    {
                        cuCtxSetCurrent(previousContext).Ok();
                    }
                }
            }

            _loadedKernels.Clear();
            _compilations.Clear();
            _disposed = true;
        }
    }

    static string GetConfigKey(TileCppConfig config)
    {
        var orderedParameters = config.Parameters.OrderBy(static pair => pair.Key, StringComparer.Ordinal);
        var parameterValues = orderedParameters.Select(static pair => $"{pair.Key}={pair.Value}");
        var parameters = string.Join(";", parameterValues);
        return $"{parameters}|{config.NumCtas}|{config.Occupancy}|{config.OptimizationLevel}|{config.NumWorkerWarps}";
    }

    readonly record struct LoadedKey(string ConfigKey, CUcontext Context);
    readonly record struct LoadedKernel(CUmodule Module, CUfunction Function);
}
