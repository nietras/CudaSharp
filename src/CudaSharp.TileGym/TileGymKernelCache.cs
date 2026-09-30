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
    long CompileStartTimestamp = 0, long LoadStartTimestamp = 0, int TileIrBytes = 0)
{
    /// <summary>The shared batch; compile/load milliseconds above are amortized shares when present.</summary>
    public TileGymBatchTiming? Batch { get; init; }
}

/// <summary>
/// Per-specialization and per-batch timings for one
/// <see cref="TileGymKernelCache.Prepare(TileGymCompilationPlan, int)" /> call.
/// </summary>
sealed record TileGymPrecompileSummary(IReadOnlyList<TileGymCompiledKernel> Results, long StartTimestamp,
    double CompileWallMilliseconds, double LoadWallMilliseconds, int CompileParallelism, int LoadParallelism,
    double CpuMilliseconds, long PeakWorkingSetBytes)
{
    public IReadOnlyList<TileGymBatchTiming>? Batches { get; init; }
    public int Failed => Results.Count(static c => c.Error is not null);
    public double CompileMilliseconds => Batches is { } batches
        ? batches.Sum(static batch => batch.CompileMilliseconds) : Results.Sum(static result => result.CompileMilliseconds);
    public double LoadMilliseconds => Batches is { } batches
        ? batches.Sum(static batch => batch.LoadMilliseconds) : Results.Sum(static result => result.LoadMilliseconds);
    public double WallMilliseconds => CompileWallMilliseconds + LoadWallMilliseconds;

    public override string ToString() =>
        $"Prepared {Results.Count} specializations ({Failed} failed) using {Batches?.Count ?? Results.Count} " +
        $"compilation attempts in {WallMilliseconds / 1000:F1} s: " +
        $"compile {CompileWallMilliseconds / 1000:F1} s wall ({CompileMilliseconds / 1000:F1} s sum, " +
        $"parallelism {CompileParallelism}), load {LoadWallMilliseconds / 1000:F1} s wall " +
        $"({LoadMilliseconds / 1000:F1} s sum, parallelism {LoadParallelism}), " +
        $"CPU {CpuMilliseconds / 1000:F1} s, peak working set {PeakWorkingSetBytes / (1024 * 1024)} MB";

    public void Write(string directory)
    {
        Directory.CreateDirectory(directory);
        var csv = new StringBuilder("Kernel,Header,TemplateArguments,TileIrBytes,CompileStartMilliseconds," +
            "CompileMilliseconds,LoadStartMilliseconds,LoadMilliseconds,TotalMilliseconds,Error," +
            "CompilationBatchId,BatchSpecializations,Cached\n");
        var markdown = new StringBuilder();
        markdown.AppendLine("# CudaSharp TileGym precompile").AppendLine();
        var summary = ToString();
        markdown.AppendLine(summary).AppendLine();
        markdown.AppendLine("Compile/load times below are amortized per-specialization shares. " +
            "TileIR bytes describe the shared batch. Cached entries incur zero work in this preparation.").AppendLine();
        markdown.AppendLine("| Kernel | Header | Template arguments | Batch TileIR bytes | Compile start ms | " +
            "Compile share ms | Load start ms | Load share ms | Total share ms | Error | Batch | " +
            "Batch specializations | Cached |");
        markdown.AppendLine("|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|");
        foreach (var row in Rows())
        {
            var error = row.Result.Error?.Message.ReplaceLineEndings(" ") ?? string.Empty;
            string[] fields =
            [
                row.Result.Spec.Name, row.Result.Spec.Header, row.Result.Spec.TemplateArguments,
                row.Result.TileIrBytes.ToString(CultureInfo.InvariantCulture),
                Format(row.CompileStart), Format(row.CompileMilliseconds), Format(row.LoadStart),
                Format(row.LoadMilliseconds), Format(row.Total), error,
                row.Result.Batch?.Id.ToString(CultureInfo.InvariantCulture) ?? string.Empty,
                (row.Result.Batch?.Specializations.Count ?? 1).ToString(CultureInfo.InvariantCulture),
                row.Cached.ToString()
            ];
            var csvFields = fields.Select(static field => $"\"{field.Replace("\"", "\"\"")}\"");
            csv.AppendJoin(',', csvFields).Append('\n');
            var markdownFields = fields.Select(static field => field.Replace("|", "\\|"));
            markdown.Append("| ").AppendJoin(" | ", markdownFields).AppendLine(" |");
        }
        var batchCsv = new StringBuilder("CompilationBatchId,Header,Kernels,Specializations,TileIrBytes," +
            "CompileStartMilliseconds,CompileMilliseconds,LoadStartMilliseconds,LoadMilliseconds," +
            "TotalMilliseconds,Error,Cached\n");
        markdown.AppendLine().AppendLine("## Compilation batches").AppendLine();
        markdown.AppendLine("Includes failed batch attempts before individual retries; each shared cost appears once.")
            .AppendLine();
        markdown.AppendLine("| Batch | Header | Kernels | Specializations | TileIR bytes | Compile start ms | " +
            "Compile ms | Load start ms | Load ms | Total ms | Error | Cached |");
        markdown.AppendLine("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|");
        foreach (var (batch, cached) in BatchRows())
        {
            var error = batch.Error?.Message.ReplaceLineEndings(" ") ?? string.Empty;
            var kernels = batch.Specializations.Select(static spec => (spec.Name, spec.Signature)).Distinct().Count();
            var compile = cached ? 0 : batch.CompileMilliseconds;
            var load = cached ? 0 : batch.LoadMilliseconds;
            string[] fields =
            [
                batch.Id.ToString(CultureInfo.InvariantCulture), batch.Header,
                kernels.ToString(CultureInfo.InvariantCulture),
                batch.Specializations.Count.ToString(CultureInfo.InvariantCulture),
                batch.TileIrBytes.ToString(CultureInfo.InvariantCulture),
                Format(cached ? 0 : Elapsed(batch.CompileStartTimestamp)), Format(compile),
                Format(cached ? 0 : Elapsed(batch.LoadStartTimestamp)), Format(load), Format(compile + load),
                error, cached.ToString()
            ];
            var csvFields = fields.Select(static field => $"\"{field.Replace("\"", "\"\"")}\"");
            batchCsv.AppendJoin(',', csvFields).Append('\n');
            var markdownFields = fields.Select(static field => field.Replace("|", "\\|"));
            markdown.Append("| ").AppendJoin(" | ", markdownFields).AppendLine(" |");
        }
        var csvPath = Path.Combine(directory, "tilegym-precompile.csv");
        var csvContents = csv.ToString();
        File.WriteAllText(csvPath, csvContents);
        var batchPath = Path.Combine(directory, "tilegym-precompile-batches.csv");
        var batchContents = batchCsv.ToString();
        File.WriteAllText(batchPath, batchContents);
        var markdownPath = Path.Combine(directory, "tilegym-precompile.md");
        var markdownContents = markdown.ToString();
        File.WriteAllText(markdownPath, markdownContents);
    }

    /// <summary>Writes amortized specialization shares and full compilation-batch costs, slowest first.</summary>
    public void WriteTo(TextWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);
        var summary = ToString();
        writer.WriteLine(summary);
        writer.WriteLine($"{"TileIR B",10} {"Compile/share",13} {"Load/share",10} {"Batch",6}  Specialization");
        foreach (var row in Rows())
        {
            var status = (row.Cached ? "  CACHED" : "") + (row.Result.Error is null ? "" : "  FAILED");
            writer.WriteLine($"{row.Result.TileIrBytes,10} {row.CompileMilliseconds,13:F1} " +
                $"{row.LoadMilliseconds,10:F1} {row.Result.Batch?.Id,6}  {row.Result.Spec}{status}");
        }
        writer.WriteLine("Full batch timings (including failed attempts before retries):");
        writer.WriteLine($"{"Batch",6} {"Variants",8} {"Compile ms",10} {"Load ms",10}  Header");
        foreach (var (batch, cached) in BatchRows())
        {
            var compile = cached ? 0 : batch.CompileMilliseconds;
            var load = cached ? 0 : batch.LoadMilliseconds;
            var status = (cached ? "  CACHED" : "") + (batch.Error is null ? "" : "  FAILED");
            writer.WriteLine($"{batch.Id,6} {batch.Specializations.Count,8} {compile,10:F1} " +
                $"{load,10:F1}  {batch.Header}{status}");
        }
    }

    Row[] Rows()
    {
        var active = Batches?.Select(static batch => batch.Id).ToHashSet() ?? [];
        return Results.Select(result =>
        {
            var cached = Batches is not null && result.Batch is { } batch && !active.Contains(batch.Id);
            var compileStart = cached ? 0 : Elapsed(result.CompileStartTimestamp);
            var loadStart = cached ? 0 : Elapsed(result.LoadStartTimestamp);
            return new Row(result, compileStart, loadStart) { Cached = cached };
        }).OrderByDescending(static row => row.Total).ToArray();
    }

    (TileGymBatchTiming Timing, bool Cached)[] BatchRows()
    {
        var active = Batches?.Select(static batch => batch.Id).ToHashSet() ?? [];
        var batches = (Batches ?? []).Concat(Results.Select(static result => result.Batch).OfType<TileGymBatchTiming>());
        return batches.DistinctBy(static batch => batch.Id)
            .OrderByDescending(static batch => batch.CompileMilliseconds + batch.LoadMilliseconds)
            .Select(batch => (batch, Batches is not null && !active.Contains(batch.Id))).ToArray();
    }

    double Elapsed(long timestamp) =>
        timestamp == 0 ? 0 : Stopwatch.GetElapsedTime(StartTimestamp, timestamp).TotalMilliseconds;

    static string Format(double value) => value.ToString("0.###", CultureInfo.InvariantCulture);

    readonly record struct Row(TileGymCompiledKernel Result, double CompileStart, double LoadStart)
    {
        public bool Cached { get; init; }
        public double CompileMilliseconds => Cached ? 0 : Result.CompileMilliseconds;
        public double LoadMilliseconds => Cached ? 0 : Result.LoadMilliseconds;
        public double Total => CompileMilliseconds + LoadMilliseconds;
    }
}

/// <summary>
/// Compiles and loads required specializations once per runtime.
/// <see cref="Prepare(TileGymCompilationPlan, int)" /> batches kernels by header and loads a shared module
/// for each batch, so later launches and tuning only measure execution.
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
    readonly List<IDisposable> _loaded = [];
    readonly System.Threading.Lock _lock = new();
    readonly TileCppCompiler _compiler;
    readonly CUcontext _context;
    int _nextBatchId;
    bool _disposed;

    public TileGymKernelCache(TileCppCompiler compiler, CUcontext context)
    {
        ArgumentNullException.ThrowIfNull(compiler);
        _compiler = compiler;
        _context = context;
    }

    /// <summary>Adapts flat specialization requests to the explicit compilation hierarchy.</summary>
    public TileGymPrecompileSummary Prepare(IEnumerable<TileGymKernelSpec> specs, int loadParallelism)
    {
        ArgumentNullException.ThrowIfNull(specs);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(loadParallelism);
        var distinct = specs.Distinct().ToArray();
        var plan = TileGymCompilationPlan.FromSpecs(distinct);
        var summary = Prepare(plan, loadParallelism);
        var results = summary.Results.ToDictionary(static result => result.Spec);
        var ordered = distinct.Select(spec => results[spec]).ToArray();
        return summary with { Results = ordered };
    }

    /// <summary>
    /// Compiles each header once for its required kernels and specializations, then loads one module per batch.
    /// NVRTC compilation failures retry individually to retain per-specialization failure isolation.
    /// </summary>
    public TileGymPrecompileSummary Prepare(TileGymCompilationPlan plan, int loadParallelism)
    {
        ArgumentNullException.ThrowIfNull(plan);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(loadParallelism);
        lock (_lock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            var missing = plan.Where(spec => !_kernels.ContainsKey(spec));
            if (missing.Units.Count != 0)
            {
                _compiler.PrepareBundledHeaders();
            }
            using var process = Process.GetCurrentProcess();
            var cpuStart = process.TotalProcessorTime;
            var start = Stopwatch.GetTimestamp();
            var pending = new List<PendingBatch>();
            var attempts = new List<TileGymBatchTiming>();
            // Compile batches sequentially: parallel requests contend inside NVRTC's frontend.
            foreach (var unit in missing.Units)
            {
                var batch = CompileBatch(unit);
                if (batch.Timing.Error is CudaException<nvrtc.nvrtcResult> && unit.Specializations.Count > 1)
                {
                    attempts.Add(batch.Timing);
                    Trace.WriteLine($"Compilation batch {batch.Timing.Id} ({unit.Header}) failed; " +
                        $"retrying {unit.Specializations.Count} specializations individually: {batch.Timing.Error.Message}");
                    foreach (var spec in unit.Specializations)
                    {
                        var single = Compile(spec);
                        var singletonPlan = TileGymCompilationPlan.FromSpecs([spec]);
                        pending.Add(new(singletonPlan.Units[0], null, single.Batch, single));
                    }
                }
                else
                {
                    pending.Add(batch);
                }
            }
            var compileWall = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            var loadStart = Stopwatch.GetTimestamp();
            var loadedTimings = new TileGymBatchTiming[pending.Count];
            var options = new ParallelOptions { MaxDegreeOfParallelism = loadParallelism };
            Parallel.For(0, pending.Count, options, i =>
            {
                var (batchTiming, results) = LoadBatch(pending[i]);
                loadedTimings[i] = batchTiming;
                foreach (var result in results)
                {
                    _kernels.TryAdd(result.Spec, new Lazy<TileGymCompiledKernel>(result));
                }
            });
            var loadWall = Stopwatch.GetElapsedTime(loadStart).TotalMilliseconds;
            attempts.AddRange(loadedTimings);
            var compiled = plan.Units.SelectMany(static unit => unit.Specializations)
                .Select(spec => _kernels[spec].Value).ToArray();
            process.Refresh();
            var cpu = (process.TotalProcessorTime - cpuStart).TotalMilliseconds;
            return new(compiled, start, compileWall, loadWall, 1, loadParallelism, cpu, process.PeakWorkingSet64)
            {
                Batches = attempts.AsReadOnly(),
            };
        }
    }

    /// <summary>Gets a prepared function, compiling an unrequested specialization individually on demand.</summary>
    public TileGymCompiledKernel Get(TileGymKernelSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);
        lock (_lock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            var lazy = _kernels.GetOrAdd(spec,
                static (s, cache) => new Lazy<TileGymCompiledKernel>(() => cache.Build(s)), this);
            return lazy.Value;
        }
    }

    TileGymCompiledKernel Build(TileGymKernelSpec spec)
    {
        var pending = Compile(spec);
        return Load(pending);
    }

    PendingBatch CompileBatch(TileGymCompilationUnit unit)
    {
        var id = ++_nextBatchId;
        using var timing = new TileCppCompilationTiming($"TileGymKernelCache.Batch[{id}, {unit.Header}]");
        var start = Stopwatch.GetTimestamp();
        try
        {
            timing.Step(nameof(CreateSource));
            var (source, headers) = CreateSource(unit);
            var expressions = unit.Specializations.Select(static spec => spec.NameExpression).ToArray();
            var sourceName = Path.GetFileNameWithoutExtension(unit.Header) + ".cu";
            timing.Step(nameof(TileCppCompiler.CompileKernels));
            var compilation = _compiler.CompileKernels(source, sourceName, expressions, Config, headers);
            timing.Step("Create shared module owner");
            var module = new TileCppModule(compilation);
            _loaded.Add(module);
            var elapsed = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            var batchTiming = new TileGymBatchTiming(id, unit.Header, unit.Specializations, elapsed,
                CompileStartTimestamp: start, TileIrBytes: compilation.TileIr.Length);
            return new(unit, module, batchTiming);
        }
        catch (Exception ex) when (ex is CudaException<nvrtc.nvrtcResult> or IOException or
            UnauthorizedAccessException or ArgumentException or InvalidOperationException)
        {
            var elapsed = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            var batchTiming = new TileGymBatchTiming(id, unit.Header, unit.Specializations, elapsed,
                Error: ex, CompileStartTimestamp: start);
            return new(unit, null, batchTiming);
        }
    }

    (TileGymBatchTiming Timing, TileGymCompiledKernel[] Results) LoadBatch(PendingBatch pending)
    {
        if (pending.Single is { } single)
        {
            var result = Load(single);
            return (result.Batch!, [result]);
        }
        if (pending.Module is null)
        {
            var failed = BatchResults(pending, pending.Timing, null);
            return (pending.Timing, failed);
        }
        var start = Stopwatch.GetTimestamp();
        TileGymBatchTiming batchTiming;
        IReadOnlyDictionary<string, CUfunction>? functions = null;
        try
        {
            cuCtxGetCurrent(out var previousContext).Ok();
            try
            {
                cuCtxSetCurrent(_context).Ok();
                functions = pending.Module.LoadFunctions();
            }
            finally
            {
                cuCtxSetCurrent(previousContext).Ok();
            }
            var elapsed = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            batchTiming = pending.Timing with { LoadMilliseconds = elapsed, LoadStartTimestamp = start };
        }
        catch (Exception ex) when (ex is CudaException<CUresult> or InvalidOperationException)
        {
            var elapsed = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            batchTiming = pending.Timing with { LoadMilliseconds = elapsed, LoadStartTimestamp = start, Error = ex };
            functions = null;
        }
        var results = BatchResults(pending, batchTiming, functions);
        return (batchTiming, results);
    }

    static TileGymCompiledKernel[] BatchResults(PendingBatch pending, TileGymBatchTiming timing,
        IReadOnlyDictionary<string, CUfunction>? functions)
    {
        var count = pending.Unit.Specializations.Count;
        var results = new TileGymCompiledKernel[count];
        for (var i = 0; i < count; i++)
        {
            var spec = pending.Unit.Specializations[i];
            var function = functions is null ? default : functions[spec.NameExpression];
            results[i] = new(spec, function, timing.CompileMilliseconds / count, timing.LoadMilliseconds / count,
                timing.Error, timing.CompileStartTimestamp, timing.LoadStartTimestamp, timing.TileIrBytes)
            {
                Batch = timing,
            };
        }
        return results;
    }

    Pending Compile(TileGymKernelSpec spec)
    {
        var id = ++_nextBatchId;
        using var timing = new TileCppCompilationTiming($"TileGymKernelCache.Compile[{spec}]");
        var start = Stopwatch.GetTimestamp();
        try
        {
            timing.Step(nameof(Create));
            var kernel = Create(spec);
            timing.Step("_loaded.Add");
            _loaded.Add(kernel);
            timing.Step("TileCppKernel.Compile (inclusive of nested stages)");
            var bytes = kernel.Compile(Config);
            var elapsed = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            var batch = new TileGymBatchTiming(id, spec.Header, [spec], elapsed,
                CompileStartTimestamp: start, TileIrBytes: bytes);
            return new(spec, kernel, elapsed, start, null, bytes) { Batch = batch };
        }
        catch (Exception ex)
        {
            var elapsed = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            var batch = new TileGymBatchTiming(id, spec.Header, [spec], elapsed, Error: ex, CompileStartTimestamp: start);
            return new(spec, null, elapsed, start, ex) { Batch = batch };
        }
    }

    TileGymCompiledKernel Load(Pending pending)
    {
        if (pending.Kernel is null)
        {
            return new(pending.Spec, default, pending.CompileMilliseconds, 0, pending.Error, pending.Start)
            {
                Batch = pending.Batch,
            };
        }
        var start = Stopwatch.GetTimestamp();
        try
        {
            cuCtxGetCurrent(out var previousContext).Ok();
            CUfunction function;
            try
            {
                cuCtxSetCurrent(_context).Ok();
                function = pending.Kernel.LoadFunction(Config);
            }
            finally
            {
                cuCtxSetCurrent(previousContext).Ok();
            }
            var load = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            return new(pending.Spec, function, pending.CompileMilliseconds, load, null, pending.Start, start,
                pending.TileIrBytes)
            {
                Batch = pending.Batch with { LoadMilliseconds = load, LoadStartTimestamp = start },
            };
        }
        catch (Exception ex)
        {
            var load = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            return new(pending.Spec, default, pending.CompileMilliseconds, load, ex, pending.Start, start,
                pending.TileIrBytes)
            {
                Batch = pending.Batch with { LoadMilliseconds = load, LoadStartTimestamp = start, Error = ex },
            };
        }
    }

    readonly record struct Pending(TileGymKernelSpec Spec, TileCppKernel? Kernel, double CompileMilliseconds,
        long Start, Exception? Error, int TileIrBytes = 0)
    {
        public required TileGymBatchTiming Batch { get; init; }
    }

    sealed record PendingBatch(TileGymCompilationUnit Unit, TileCppModule? Module, TileGymBatchTiming Timing,
        Pending? Single = null);

    TileCppKernel Create(TileGymKernelSpec spec)
    {
        var plan = TileGymCompilationPlan.FromSpecs([spec]);
        var (source, headers) = CreateSource(plan.Units[0]);
        return new TileCppKernel(_compiler, source, $"{spec.Name}.cu", spec.Name, headers,
            nameExpression: spec.NameExpression);
    }

    static (string Source, IReadOnlyList<TileCppHeader> Headers) CreateSource(TileGymCompilationUnit unit)
    {
        var headerName = Path.GetFileName(unit.Header);
        var prefix = $$"""
            using int32_t = int;
            using uint32_t = unsigned int;
            using int64_t = long long;
            using uint64_t = unsigned long long;
            #include <cmath>
            #include "{{headerName}}"
            """;
        var source = new StringBuilder(prefix);
        source.AppendLine();
        foreach (var kernel in unit.Kernels)
        {
            foreach (var spec in kernel.Specializations)
            {
                source.Append("template __tile_global__ void ").Append(spec.Name).Append('<')
                    .Append(spec.TemplateArguments).Append(">(").Append(spec.Signature).AppendLine(");");
            }
        }
        var headerSource = ReadSource(unit.Header);
        var mainHeader = new TileCppHeader(headerName, headerSource);
        var headers = new List<TileCppHeader>(4) { mainHeader, TypeTraits, Cmath };
        if (unit.Header.StartsWith("conv", StringComparison.Ordinal))
        {
            var commonHeaderSource = ReadSource("convolution_common.cuh");
            var commonHeader = new TileCppHeader("convolution_common.cuh", commonHeaderSource);
            headers.Add(commonHeader);
        }
        var sourceText = source.ToString();
        return (sourceText, headers);
    }

    static string ReadSource(string relativePath) =>
        Sources.GetOrAdd(relativePath, static path =>
        {
            var sourcePath = Path.Combine(SourceRoot, path);
            return File.ReadAllText(sourcePath);
        });

    public void Dispose()
    {
        lock (_lock)
        {
            _disposed = true;
            while (_loaded.Count != 0)
            {
                var index = _loaded.Count - 1;
                _loaded[index].Dispose();
                _loaded.RemoveAt(index);
            }
            _kernels.Clear();
        }
    }
}
