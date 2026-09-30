using System.Collections.Generic;
using System.Collections.ObjectModel;
using static CudaSharp.nvcuda;

namespace CudaSharp.TileGym;

/// <summary>Owns one compiled CUDA TileIR module loaded independently in each current CUDA context.</summary>
/// <remarks>
/// This owner is not thread-safe. Dispose it before destroying any context in which its functions were loaded.
/// </remarks>
/// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html" />
public sealed class TileCppModule : IDisposable
{
    readonly byte[] _tileIr;
    readonly IReadOnlyDictionary<string, string> _entryPoints;
    readonly Dictionary<CUcontext, LoadedModule> _loadedModules = [];
    bool _disposed;

    /// <summary>Snapshots a compiled batch without requiring a CUDA context or calling the CUDA driver.</summary>
    /// <param name="compilation">TileIR bytecode and exact lowered entry points to load together.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html" />
    public TileCppModule(TileCppCompilationBatch compilation)
    {
        ArgumentNullException.ThrowIfNull(compilation);
        ArgumentNullException.ThrowIfNull(compilation.TileIr);
        ArgumentNullException.ThrowIfNull(compilation.EntryPoints);
        if (compilation.TileIr.Length == 0 || compilation.EntryPoints.Count == 0)
        {
            throw new ArgumentException("A compiled batch must contain TileIR and entry points.", nameof(compilation));
        }
        _tileIr = (byte[])compilation.TileIr.Clone();
        var entryPoints = new Dictionary<string, string>(compilation.EntryPoints.Count, StringComparer.Ordinal);
        foreach (var entry in compilation.EntryPoints)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(entry.Key, nameof(compilation));
            ArgumentException.ThrowIfNullOrWhiteSpace(entry.Value, nameof(compilation));
            if (entry.Key.Contains('\0') || entry.Value.Contains('\0'))
            {
                throw new ArgumentException("Entry points cannot contain a null character.", nameof(compilation));
            }
            entryPoints.Add(entry.Key, entry.Value);
        }
        _entryPoints = new ReadOnlyDictionary<string, string>(entryPoints);
    }

    /// <summary>Loads all exact lowered entry points in the current CUDA context, caching the whole result.</summary>
    /// <returns>An immutable expression-to-function map; repeated calls in the same context return the same map.</returns>
    /// <exception cref="InvalidOperationException">No CUDA context is current.</exception>
    /// <exception cref="ObjectDisposedException">This module owner has been disposed.</exception>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html" />
    public IReadOnlyDictionary<string, CUfunction> LoadFunctions()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        cuCtxGetCurrent(out var context).Ok();
        if (context.Value == IntPtr.Zero)
        {
            throw new InvalidOperationException("A CUDA context must be current before loading a CUDA Tile C++ module.");
        }
        if (_loadedModules.TryGetValue(context, out var loaded))
        {
            return loaded.Functions;
        }

        cuModuleLoadData(out var module, _tileIr).Ok();
        try
        {
            var functions = new Dictionary<string, CUfunction>(_entryPoints.Count, StringComparer.Ordinal);
            foreach (var entry in _entryPoints)
            {
                cuModuleGetFunction(out var function, module, entry.Value).Ok();
                functions.Add(entry.Key, function);
            }
            var readOnlyFunctions = new ReadOnlyDictionary<string, CUfunction>(functions);
            _loadedModules.Add(context, new LoadedModule(module, readOnlyFunctions));
            return readOnlyFunctions;
        }
        catch
        {
            cuModuleUnload(module).Ok();
            throw;
        }
    }

    /// <summary>Unloads every owned module in its CUDA context, restoring the caller's current context.</summary>
    /// <remarks>Repeated calls are safe; if unloading fails, a subsequent call retries only remaining modules.</remarks>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html" />
    public void Dispose()
    {
        _disposed = true;
        CUcontext[] contexts = [.. _loadedModules.Keys];
        foreach (var context in contexts)
        {
            cuCtxGetCurrent(out var previousContext).Ok();
            if (previousContext != context)
            {
                cuCtxSetCurrent(context).Ok();
            }
            try
            {
                var loaded = _loadedModules[context];
                cuModuleUnload(loaded.Module).Ok();
                _loadedModules.Remove(context);
            }
            finally
            {
                if (previousContext != context)
                {
                    cuCtxSetCurrent(previousContext).Ok();
                }
            }
        }
    }

    readonly record struct LoadedModule(CUmodule Module, IReadOnlyDictionary<string, CUfunction> Functions)
    {
    }
}
