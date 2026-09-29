using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CudaSharp.TileGym;
using static CudaSharp.nvcuda;

namespace CudaSharp.Tester;

/// <summary>Result of compiling (NVRTC) and loading (driver JIT) one kernel specialization.</summary>
sealed record TileGymCompiledKernel(TileGymKernelSpec Spec, CUfunction Function,
    double CompileMilliseconds, double LoadMilliseconds, Exception? Error = null,
    long CompileStartTimestamp = 0, long LoadStartTimestamp = 0, int TileIrBytes = 0);

/// <summary>Per-kernel compile and load timings for one <see cref="TileGymKernelCache.Prepare" /> call.</summary>
sealed record TileGymPrecompileSummary(IReadOnlyList<TileGymCompiledKernel> Results, long StartTimestamp,
    double CompileWallMilliseconds, double LoadWallMilliseconds, int CompileParallelism, int LoadParallelism,
    double CpuMilliseconds, long PeakWorkingSetBytes)
{
    public int Failed => Results.Count(static c => c.Error is not null);
    public double CompileMilliseconds => Results.Sum(static c => c.CompileMilliseconds);
    public double LoadMilliseconds => Results.Sum(static c => c.LoadMilliseconds);
    public double WallMilliseconds => CompileWallMilliseconds + LoadWallMilliseconds;

    public override string ToString() =>
        $"Precompiled {Results.Count} kernels ({Failed} failed) in {WallMilliseconds / 1000:F1} s: " +
        $"compile {CompileWallMilliseconds / 1000:F1} s wall ({CompileMilliseconds / 1000:F1} s sum, " +
        $"parallelism {CompileParallelism}), load {LoadWallMilliseconds / 1000:F1} s wall " +
        $"({LoadMilliseconds / 1000:F1} s sum, parallelism {LoadParallelism}), " +
        $"CPU {CpuMilliseconds / 1000:F1} s, peak working set {PeakWorkingSetBytes / (1024 * 1024)} MB";

    public void Write(string directory)
    {
        Directory.CreateDirectory(directory);
        var csv = new StringBuilder("Kernel,Header,TemplateArguments,TileIrBytes,CompileStartMilliseconds," +
            "CompileMilliseconds,LoadStartMilliseconds,LoadMilliseconds,TotalMilliseconds,Error\n");
        var markdown = new StringBuilder();
        markdown.AppendLine("# CudaSharp TileGym precompile").AppendLine().AppendLine(ToString()).AppendLine();
        markdown.AppendLine("| Kernel | Template arguments | TileIR bytes | Compile start ms | Compile ms | " +
            "Load start ms | Load ms | Total ms | Error |");
        markdown.AppendLine("|---|---|---:|---:|---:|---:|---:|---:|---|");
        foreach (var row in Rows())
        {
            var error = row.Result.Error?.Message.ReplaceLineEndings(" ") ?? string.Empty;
            string[] fields =
            [
                row.Result.Spec.Name, row.Result.Spec.Header, row.Result.Spec.TemplateArguments,
                row.Result.TileIrBytes.ToString(CultureInfo.InvariantCulture),
                Format(row.CompileStart), Format(row.Result.CompileMilliseconds), Format(row.LoadStart),
                Format(row.Result.LoadMilliseconds), Format(row.Total), error
            ];
            csv.AppendJoin(',', fields.Select(static f => $"\"{f.Replace("\"", "\"\"")}\"")).Append('\n');
            markdown.Append("| ").AppendJoin(" | ", fields.Select(static f => f.Replace("|", "\\|"))).AppendLine(" |");
        }
        File.WriteAllText(Path.Combine(directory, "tilegym-precompile.csv"), csv.ToString());
        File.WriteAllText(Path.Combine(directory, "tilegym-precompile.md"), markdown.ToString());
    }

    /// <summary>Writes per-kernel compile and load times, slowest first.</summary>
    public void WriteTo(TextWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);
        writer.WriteLine(ToString());
        writer.WriteLine($"{"TileIR B",10} {"Compile ms",10} {"Load ms",10} {"Load start",10}  Kernel");
        foreach (var row in Rows())
        {
            var failed = row.Result.Error is null ? "" : "  FAILED";
            writer.WriteLine($"{row.Result.TileIrBytes,10} {row.Result.CompileMilliseconds,10:F0} " +
                $"{row.Result.LoadMilliseconds,10:F0} {row.LoadStart,10:F0}  {row.Result.Spec}{failed}");
        }
    }

    Row[] Rows() => Results
        .Select(r => new Row(r, Elapsed(r.CompileStartTimestamp), Elapsed(r.LoadStartTimestamp)))
        .OrderByDescending(static r => r.Total)
        .ToArray();

    double Elapsed(long timestamp) =>
        timestamp == 0 ? 0 : Stopwatch.GetElapsedTime(StartTimestamp, timestamp).TotalMilliseconds;

    static string Format(double value) => value.ToString("0.###", CultureInfo.InvariantCulture);

    readonly record struct Row(TileGymCompiledKernel Result, double CompileStart, double LoadStart)
    {
        public double Total => Result.CompileMilliseconds + Result.LoadMilliseconds;
    }
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

    /// <summary>
    /// Compiles all distinct specializations with NVRTC in parallel, then loads them in the driver with at most
    /// <paramref name="loadParallelism" /> concurrent loads. Failures are recorded, not thrown.
    /// </summary>
    public TileGymPrecompileSummary Prepare(IEnumerable<TileGymKernelSpec> specs, int loadParallelism)
    {
        ArgumentNullException.ThrowIfNull(specs);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(loadParallelism);
        var distinct = specs.Distinct().ToArray();
        _compiler.PrepareBundledHeaders();
        using var process = Process.GetCurrentProcess();
        var cpuStart = process.TotalProcessorTime;
        var start = Stopwatch.GetTimestamp();
        var compileParallelism = Environment.ProcessorCount;
        var pending = new Pending[distinct.Length];
        var compileOptions = new ParallelOptions { MaxDegreeOfParallelism = compileParallelism };
        Parallel.For(0, distinct.Length, compileOptions, i => pending[i] = Compile(distinct[i]));
        var compileWall = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        var loadStart = Stopwatch.GetTimestamp();
        var compiled = new TileGymCompiledKernel[distinct.Length];
        var loadOptions = new ParallelOptions { MaxDegreeOfParallelism = loadParallelism };
        Parallel.For(0, distinct.Length, loadOptions, i =>
        {
            var result = Load(pending[i]);
            compiled[i] = _kernels.GetOrAdd(result.Spec, new Lazy<TileGymCompiledKernel>(result)).Value;
        });
        var loadWall = Stopwatch.GetElapsedTime(loadStart).TotalMilliseconds;
        process.Refresh();
        var cpu = (process.TotalProcessorTime - cpuStart).TotalMilliseconds;
        return new(compiled, start, compileWall, loadWall, compileParallelism, loadParallelism, cpu,
            process.PeakWorkingSet64);
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
        var pending = Compile(spec);
        return Load(pending);
    }

    Pending Compile(TileGymKernelSpec spec)
    {
        var start = Stopwatch.GetTimestamp();
        try
        {
            var kernel = Create(spec);
            _loaded.Add(kernel);
            var bytes = kernel.Compile(Config);
            return new(spec, kernel, Stopwatch.GetElapsedTime(start).TotalMilliseconds, start, null, bytes);
        }
        catch (Exception ex)
        {
            return new(spec, null, Stopwatch.GetElapsedTime(start).TotalMilliseconds, start, ex);
        }
    }

    TileGymCompiledKernel Load(Pending pending)
    {
        if (pending.Kernel is null)
        {
            return new(pending.Spec, default, pending.CompileMilliseconds, 0, pending.Error, pending.Start);
        }
        var start = Stopwatch.GetTimestamp();
        try
        {
            // Module loading is per context and the context is current per thread, so set it on worker threads.
            cuCtxSetCurrent(_context).Ok();
            var function = pending.Kernel.LoadFunction(Config);
            var load = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            return new(pending.Spec, function, pending.CompileMilliseconds, load, null, pending.Start, start,
                pending.TileIrBytes);
        }
        catch (Exception ex)
        {
            var load = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            return new(pending.Spec, default, pending.CompileMilliseconds, load, ex, pending.Start, start,
                pending.TileIrBytes);
        }
    }

    readonly record struct Pending(TileGymKernelSpec Spec, TileCppKernel? Kernel, double CompileMilliseconds,
        long Start, Exception? Error, int TileIrBytes = 0);

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
