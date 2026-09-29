using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using CudaSharp.TileGym;
using static CudaSharp.nvcuda;

namespace CudaSharp.Tester;

/// <summary>Result of compiling (NVRTC) and loading (driver JIT) one kernel specialization.</summary>
sealed record TileGymCompiledKernel(TileGymKernelSpec Spec, CUfunction Function,
    double CompileMilliseconds, double LoadMilliseconds, Exception? Error = null);

readonly record struct TileGymPrecompileSummary(int Kernels, int Failed, double WallMilliseconds,
    double CompileMilliseconds, double LoadMilliseconds)
{
    public override string ToString() =>
        $"Precompiled {Kernels} kernels ({Failed} failed) in {WallMilliseconds / 1000:F1} s " +
        $"(sum compile {CompileMilliseconds / 1000:F1} s, sum load {LoadMilliseconds / 1000:F1} s)";
}

/// <summary>
/// Compiles and loads kernel specializations once per runtime. <see cref="Prepare" /> builds every requested
/// specialization in parallel, so later launches and tuning only measure execution.
/// </summary>
sealed class TileGymKernelCache : IDisposable
{
    static readonly TileCppConfig Config = new([]);
    static readonly string SourceRoot = Path.Combine(AppContext.BaseDirectory, "src-tilecpp", "tilegym");
    static readonly ConcurrentDictionary<string, string> Sources = new(StringComparer.Ordinal);
    static readonly TileCppHeader TypeTraits = new("type_traits",
        "namespace std { template<bool B, class T, class F> struct conditional { using type = T; }; " +
        "template<class T, class F> struct conditional<false, T, F> { using type = F; }; " +
        "template<bool B, class T, class F> using conditional_t = typename conditional<B, T, F>::type; " +
        "template<class A, class B> struct is_same { static constexpr bool value = false; }; " +
        "template<class A> struct is_same<A, A> { static constexpr bool value = true; }; " +
        "template<class A, class B> inline constexpr bool is_same_v = is_same<A, B>::value; }");
    static readonly TileCppHeader Cmath = new("cmath",
        "#ifndef INFINITY\n#define INFINITY __builtin_bit_cast(float, 0x7f800000u)\n#endif\n");

    readonly ConcurrentDictionary<TileGymKernelSpec, Lazy<TileGymCompiledKernel>> _kernels = new();
    readonly ConcurrentBag<TileCppKernel> _loaded = [];
    readonly TileCppCompiler _compiler;
    readonly CUcontext _context;

    public TileGymKernelCache(TileCppCompiler compiler, CUcontext context)
    {
        ArgumentNullException.ThrowIfNull(compiler);
        _compiler = compiler;
        _context = context;
    }

    /// <summary>Compiles and loads all distinct specializations in parallel; failures are recorded, not thrown.</summary>
    public TileGymPrecompileSummary Prepare(IEnumerable<TileGymKernelSpec> specs)
    {
        ArgumentNullException.ThrowIfNull(specs);
        var distinct = specs.Distinct().ToArray();
        _compiler.PrepareBundledHeaders();
        var watch = Stopwatch.StartNew();
        var compiled = new TileGymCompiledKernel[distinct.Length];
        Parallel.For(0, distinct.Length, i => compiled[i] = Get(distinct[i]));
        watch.Stop();
        return new(compiled.Length, compiled.Count(static c => c.Error is not null), watch.Elapsed.TotalMilliseconds,
            compiled.Sum(static c => c.CompileMilliseconds), compiled.Sum(static c => c.LoadMilliseconds));
    }

    /// <summary>Gets a compiled specialization, building it on demand if it was not prepared.</summary>
    public TileGymCompiledKernel Get(TileGymKernelSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);
        var lazy = _kernels.GetOrAdd(spec, static (s, cache) => new Lazy<TileGymCompiledKernel>(() => cache.Build(s)), this);
        return lazy.Value;
    }

    TileGymCompiledKernel Build(TileGymKernelSpec spec)
    {
        // Module loading is per context and the context is current per thread, so set it on worker threads.
        cuCtxSetCurrent(_context).Ok();
        var compile = 0d;
        var watch = Stopwatch.StartNew();
        try
        {
            var kernel = Create(spec);
            _loaded.Add(kernel);
            kernel.Compile(Config);
            compile = watch.Elapsed.TotalMilliseconds;
            watch.Restart();
            var function = kernel.LoadFunction(Config);
            return new(spec, function, compile, watch.Elapsed.TotalMilliseconds);
        }
        catch (Exception ex)
        {
            return new(spec, default, compile, 0, ex);
        }
    }

    TileCppKernel Create(TileGymKernelSpec spec)
    {
        var headerName = Path.GetFileName(spec.Header);
        var source = $$"""
            using int32_t = int;
            using uint32_t = unsigned int;
            using int64_t = long long;
            using uint64_t = unsigned long long;
            #include <cmath>
            #include "{{headerName}}"
            template __tile_global__ void {{spec.Name}}<{{spec.TemplateArguments}}>({{spec.Signature}});
            """;
        var headers = new List<TileCppHeader>(4) { new(headerName, ReadSource(spec.Header)), TypeTraits, Cmath };
        if (spec.Header.StartsWith("conv", StringComparison.Ordinal))
        {
            headers.Add(new TileCppHeader("convolution_common.cuh", ReadSource("convolution_common.cuh")));
        }
        return new TileCppKernel(_compiler, source, $"{spec.Name}.cu", spec.Name, headers,
            nameExpression: $"&{spec.Name}<{spec.TemplateArguments}>");
    }

    static string ReadSource(string relativePath) =>
        Sources.GetOrAdd(relativePath, static path => File.ReadAllText(Path.Combine(SourceRoot, path)));

    public void Dispose()
    {
        while (_loaded.TryTake(out var kernel))
        {
            kernel.Dispose();
        }
        _kernels.Clear();
    }
}
