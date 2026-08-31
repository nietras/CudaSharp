using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using static CudaSharp.nvcuda;

namespace CudaSharp.Tile;

/// <summary>Compiles, caches, loads, and launches variants of a CUDA Tile C++ kernel.</summary>
/// <remarks>
/// Loaded modules are cached per CUDA context. Dispose this object before destroying any context in which a variant
/// was loaded. CUDA Tile kernels are launched with one thread per tile block as required by NVIDIA.
/// </remarks>
/// <seealso href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#launching-kernels" />
public sealed class TileCppKernel : IDisposable
{
    readonly Dictionary<string, byte[]> _cubins = new(StringComparer.Ordinal);
    readonly Dictionary<LoadedKey, LoadedKernel> _loadedKernels = [];
    readonly Lock _lock = new();
    readonly TileCppCompiler _compiler;
    readonly string _source;
    readonly string _sourceName;
    readonly string _kernelName;
    readonly IReadOnlyList<TileCppHeader>? _headers;
    readonly IReadOnlyList<string>? _additionalOptions;
    bool _disposed;

    /// <summary>Creates a reusable CUDA Tile C++ kernel definition.</summary>
    /// <param name="compiler">CUDA Tile C++ compiler.</param>
    /// <param name="source">CUDA Tile C++ source text.</param>
    /// <param name="sourceName">Diagnostic source name.</param>
    /// <param name="kernelName">Unmangled kernel entry-point name.</param>
    /// <param name="headers">Optional virtual headers.</param>
    /// <param name="additionalOptions">Optional additional NVRTC command-line options.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#writing-tile-kernels" />
    public TileCppKernel(TileCppCompiler compiler, string source, string sourceName, string kernelName,
        IReadOnlyList<TileCppHeader>? headers = null, IReadOnlyList<string>? additionalOptions = null)
    {
        ArgumentNullException.ThrowIfNull(compiler);
        ArgumentNullException.ThrowIfNull(source);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourceName);
        ArgumentException.ThrowIfNullOrWhiteSpace(kernelName);
        _compiler = compiler;
        _source = source;
        _sourceName = sourceName;
        _kernelName = kernelName;
        _headers = headers;
        _additionalOptions = additionalOptions;
    }

    /// <summary>Gets the loaded CUDA function for a configuration in the current CUDA context.</summary>
    /// <param name="config">Compile-time kernel configuration.</param>
    /// <returns>A CUDA function handle for the current context.</returns>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html" />
    public CUfunction GetFunction(TileCppConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        ObjectDisposedException.ThrowIf(_disposed, this);
        cuCtxGetCurrent(out var context).Ok();
        if (context.Value == IntPtr.Zero)
            throw new InvalidOperationException("A CUDA context must be current before loading a CUDA Tile C++ kernel.");

        var configKey = GetConfigKey(config);
        var loadedKey = new LoadedKey(configKey, context);
        lock (_lock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_loadedKernels.TryGetValue(loadedKey, out var loaded))
                return loaded.Function;

            if (!_cubins.TryGetValue(configKey, out var cubin))
            {
                cubin = _compiler.Compile(_source, _sourceName, config, _headers, _additionalOptions);
                if (cubin.Length == 0)
                    throw new InvalidOperationException("The TileIR assembler returned an empty CUBIN image.");
                _cubins.Add(configKey, cubin);
            }

            cuModuleLoadData(out var module, cubin).Ok();
            try
            {
                cuModuleGetFunction(out var function, module, _kernelName).Ok();
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

    /// <summary>Launches a CUDA Tile C++ kernel using pointers to argument storage.</summary>
    /// <param name="config">Compile-time kernel configuration.</param>
    /// <param name="grid">Tile-block grid dimensions.</param>
    /// <param name="stream">CUDA stream on which to enqueue the launch.</param>
    /// <param name="argumentPointers">Pointers to storage for each kernel argument.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#launching-kernels" />
    [SkipLocalsInit]
    public unsafe void Launch(TileCppConfig config, TileCppGrid grid, CUstream stream,
        ReadOnlySpan<IntPtr> argumentPointers)
    {
        var function = GetFunction(config);
        fixed (IntPtr* parameters = argumentPointers)
        {
            cuLaunchKernel(function,
                grid.X, grid.Y, grid.Z,
                1, 1, 1,
                0, stream,
                (void**)parameters, null).Ok();
        }
    }

    /// <summary>Launches a CUDA Tile C++ kernel with one unmanaged argument.</summary>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#launching-kernels" />
    [SkipLocalsInit]
    public unsafe void Launch<T1>(TileCppConfig config, TileCppGrid grid, CUstream stream, T1 arg1)
        where T1 : unmanaged
    {
        var function = GetFunction(config);
        var parameters = stackalloc void*[] { &arg1 };
        cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, stream, parameters, null).Ok();
    }

    /// <summary>Launches a CUDA Tile C++ kernel with two unmanaged arguments.</summary>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#launching-kernels" />
    public void Launch<T1, T2>(TileCppConfig config, TileCppGrid grid, CUstream stream, T1 arg1, T2 arg2)
        where T1 : unmanaged
        where T2 : unmanaged
    {
        var function = GetFunction(config);
        cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, stream, arg1, arg2).Ok();
    }

    /// <summary>Launches a CUDA Tile C++ kernel with three unmanaged arguments.</summary>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#launching-kernels" />
    public void Launch<T1, T2, T3>(TileCppConfig config, TileCppGrid grid, CUstream stream,
        T1 arg1, T2 arg2, T3 arg3)
        where T1 : unmanaged
        where T2 : unmanaged
        where T3 : unmanaged
    {
        var function = GetFunction(config);
        cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, stream, arg1, arg2, arg3).Ok();
    }

    /// <summary>Launches a CUDA Tile C++ kernel with four unmanaged arguments.</summary>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#launching-kernels" />
    public void Launch<T1, T2, T3, T4>(TileCppConfig config, TileCppGrid grid, CUstream stream,
        T1 arg1, T2 arg2, T3 arg3, T4 arg4)
        where T1 : unmanaged
        where T2 : unmanaged
        where T3 : unmanaged
        where T4 : unmanaged
    {
        var function = GetFunction(config);
        cuLaunchKernel(function, grid.X, grid.Y, grid.Z, 1, 1, 1, 0, stream, arg1, arg2, arg3, arg4).Ok();
    }

    /// <summary>Unloads all context-specific CUDA modules owned by this kernel.</summary>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html" />
    public void Dispose()
    {
        lock (_lock)
        {
            if (_disposed)
                return;

            foreach (var entry in _loadedKernels)
            {
                cuCtxPushCurrent(entry.Key.Context).Ok();
                try
                {
                    cuModuleUnload(entry.Value.Module).Ok();
                }
                finally
                {
                    cuCtxPopCurrent(out _).Ok();
                }
            }

            _loadedKernels.Clear();
            _cubins.Clear();
            _disposed = true;
        }
    }

    static string GetConfigKey(TileCppConfig config)
    {
        var parameters = string.Join(";", config.Parameters.OrderBy(static pair => pair.Key, StringComparer.Ordinal)
            .Select(static pair => $"{pair.Key}={pair.Value}"));
        return $"{parameters}|{config.NumCtas}|{config.Occupancy}|{config.OptimizationLevel}|{config.NumWorkerWarps}";
    }

    readonly record struct LoadedKey(string ConfigKey, CUcontext Context);
    readonly record struct LoadedKernel(CUmodule Module, CUfunction Function);
}
