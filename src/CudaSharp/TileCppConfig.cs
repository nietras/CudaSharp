using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;

namespace CudaSharp.Tile;

/// <summary>Describes one CUDA Tile C++ kernel compilation configuration.</summary>
/// <remarks>
/// Compiler hints follow the constraints documented for
/// <see href="https://docs.nvidia.com/cuda/cutile-python/execution.html#cuda.tile.kernel">CUDA Tile kernels</see>.
/// </remarks>
public sealed class TileCppConfig
{
    readonly IReadOnlyDictionary<string, string> _parameters;

    /// <summary>Creates a CUDA Tile C++ kernel configuration.</summary>
    /// <param name="parameters">Compile-time parameter names and source-level values.</param>
    /// <param name="numCtas">Number of CTAs in a cluster, or <see langword="null" /> for automatic selection.</param>
    /// <param name="occupancy">Expected active CTAs per SM, or <see langword="null" /> for automatic selection.</param>
    /// <param name="optimizationLevel">Tile compiler optimization level.</param>
    /// <param name="numWorkerWarps">CUDA core worker warps, or <see langword="null" /> for automatic selection.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/cutile-python/execution.html#cuda.tile.kernel" />
    public TileCppConfig(IEnumerable<KeyValuePair<string, string>> parameters,
        int? numCtas = null, int? occupancy = null, int optimizationLevel = 3, int? numWorkerWarps = null)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        if (numCtas is not null && (numCtas < 1 || numCtas > 16 || !int.IsPow2(numCtas.Value)))
            throw new ArgumentOutOfRangeException(nameof(numCtas), "The number of CTAs must be a power of two from 1 through 16.");
        if (occupancy is < 1 or > 32)
            throw new ArgumentOutOfRangeException(nameof(occupancy), "Occupancy must be from 1 through 32.");
        if (optimizationLevel is < 0 or > 3)
            throw new ArgumentOutOfRangeException(nameof(optimizationLevel), "The optimization level must be from 0 through 3.");
        if (numWorkerWarps is not null and not 4 and not 8)
            throw new ArgumentOutOfRangeException(nameof(numWorkerWarps), "Worker warps must be either 4 or 8.");

        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var parameter in parameters)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(parameter.Key);
            ArgumentNullException.ThrowIfNull(parameter.Value);
            if (!values.TryAdd(parameter.Key, parameter.Value))
                throw new ArgumentException($"The parameter '{parameter.Key}' occurs more than once.", nameof(parameters));
        }

        _parameters = new ReadOnlyDictionary<string, string>(values);
        NumCtas = numCtas;
        Occupancy = occupancy;
        OptimizationLevel = optimizationLevel;
        NumWorkerWarps = numWorkerWarps;
    }

    /// <summary>Gets the compile-time parameters for the kernel variant.</summary>
    public IReadOnlyDictionary<string, string> Parameters => _parameters;

    /// <summary>Gets the number of CTAs in a cluster, or <see langword="null" /> for automatic selection.</summary>
    public int? NumCtas { get; }

    /// <summary>Gets the expected active CTAs per SM, or <see langword="null" /> for automatic selection.</summary>
    public int? Occupancy { get; }

    /// <summary>Gets the Tile compiler optimization level.</summary>
    public int OptimizationLevel { get; }

    /// <summary>Gets the CUDA core worker warp count, or <see langword="null" /> for automatic selection.</summary>
    public int? NumWorkerWarps { get; }

    /// <summary>Gets a compile-time parameter value by name.</summary>
    /// <param name="name">Parameter name.</param>
    /// <returns>The source-level parameter value.</returns>
    /// <seealso href="https://docs.nvidia.com/cuda/cutile-python/execution.html#cuda.tile.kernel" />
    public string this[string name] => _parameters[name];

    /// <inheritdoc />
    public override string ToString()
    {
        var parameters = string.Join(", ", _parameters.Select(static pair => $"{pair.Key}={pair.Value}"));
        return $"TileCppConfig({parameters}, NumCtas={NumCtas}, Occupancy={Occupancy}, " +
            $"OptimizationLevel={OptimizationLevel}, NumWorkerWarps={NumWorkerWarps})";
    }
}

/// <summary>Contains CUDA Tile C++ kernel configurations and an optional filtering predicate.</summary>
/// <seealso href="https://docs.nvidia.com/cuda/cutile-python/performance.html" />
public sealed class TileCppSearchSpace : IReadOnlyList<TileCppConfig>
{
    readonly TileCppConfig[] _configs;
    readonly Func<IReadOnlyDictionary<string, object?>, TileCppConfig, bool>? _predicate;

    /// <summary>Creates a search space whose configurations have an identical parameter-name set.</summary>
    /// <param name="configs">Kernel configurations to search.</param>
    /// <param name="predicate">Optional problem-dependent configuration filter.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/cutile-python/performance.html" />
    public TileCppSearchSpace(IEnumerable<TileCppConfig> configs,
        Func<IReadOnlyDictionary<string, object?>, TileCppConfig, bool>? predicate = null)
    {
        ArgumentNullException.ThrowIfNull(configs);
        _configs = [.. configs];
        if (_configs.Length == 0)
            throw new ArgumentException("At least one configuration is required.", nameof(configs));
        if (_configs.Any(static config => config is null))
            throw new ArgumentException("Configurations cannot contain null.", nameof(configs));

        var expectedKeys = _configs[0].Parameters.Keys.ToHashSet(StringComparer.Ordinal);
        for (var i = 1; i < _configs.Length; i++)
        {
            if (!expectedKeys.SetEquals(_configs[i].Parameters.Keys))
                throw new ArgumentException("All configurations must have the same parameter names.", nameof(configs));
        }

        _predicate = predicate;
    }

    /// <inheritdoc />
    public int Count => _configs.Length;

    /// <inheritdoc />
    public TileCppConfig this[int index] => _configs[index];

    /// <summary>Determines whether a configuration is valid for the supplied problem arguments.</summary>
    /// <param name="namedArguments">Problem arguments available to the predicate.</param>
    /// <param name="config">Configuration to test.</param>
    /// <returns><see langword="true" /> when the configuration can be tuned.</returns>
    /// <seealso href="https://docs.nvidia.com/cuda/cutile-python/performance.html" />
    public bool IsMatch(IReadOnlyDictionary<string, object?> namedArguments, TileCppConfig config)
    {
        ArgumentNullException.ThrowIfNull(namedArguments);
        ArgumentNullException.ThrowIfNull(config);
        return _predicate?.Invoke(namedArguments, config) ?? true;
    }

    /// <inheritdoc />
    public IEnumerator<TileCppConfig> GetEnumerator() => ((IEnumerable<TileCppConfig>)_configs).GetEnumerator();

    /// <inheritdoc />
    IEnumerator IEnumerable.GetEnumerator() => _configs.GetEnumerator();
}

/// <summary>Specifies a one-, two-, or three-dimensional CUDA Tile kernel grid.</summary>
/// <seealso href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#launching-kernels" />
public readonly record struct TileCppGrid
{
    /// <summary>Creates a CUDA Tile kernel grid.</summary>
    /// <param name="x">Grid size along the x-axis.</param>
    /// <param name="y">Grid size along the y-axis.</param>
    /// <param name="z">Grid size along the z-axis.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#launching-kernels" />
    public TileCppGrid(uint x, uint y = 1, uint z = 1)
    {
        ArgumentOutOfRangeException.ThrowIfZero(x);
        ArgumentOutOfRangeException.ThrowIfZero(y);
        ArgumentOutOfRangeException.ThrowIfZero(z);
        X = x;
        Y = y;
        Z = z;
    }

    /// <summary>Gets the grid size along the x-axis.</summary>
    public uint X { get; }

    /// <summary>Gets the grid size along the y-axis.</summary>
    public uint Y { get; }

    /// <summary>Gets the grid size along the z-axis.</summary>
    public uint Z { get; }
}

/// <summary>Describes the configuration selected by CUDA Tile C++ autotuning.</summary>
/// <seealso href="https://docs.nvidia.com/cuda/cutile-python/performance.html" />
public sealed record TileCppTunedResult
{
    /// <summary>Creates an autotuning result.</summary>
    /// <param name="config">Selected configuration.</param>
    /// <param name="grid">Grid associated with the selected configuration.</param>
    /// <param name="milliseconds">Median trimmed execution time, or <see cref="float.NaN" /> on a cache hit.</param>
    /// <seealso href="https://docs.nvidia.com/cuda/cutile-python/performance.html" />
    public TileCppTunedResult(TileCppConfig config, TileCppGrid grid, float milliseconds)
    {
        ArgumentNullException.ThrowIfNull(config);
        Config = config;
        Grid = grid;
        Milliseconds = milliseconds;
    }

    /// <summary>Gets the selected configuration.</summary>
    public TileCppConfig Config { get; }

    /// <summary>Gets the selected launch grid.</summary>
    public TileCppGrid Grid { get; }

    /// <summary>Gets the measured execution time in milliseconds.</summary>
    public float Milliseconds { get; }
}
