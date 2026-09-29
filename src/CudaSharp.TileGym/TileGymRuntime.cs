using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using CudaSharp.TileGym;
using static CudaSharp.nvcuda;

namespace CudaSharp.Tester;

/// <summary>Selects how the chosen variant is measured after tuning.</summary>
/// <param name="Warmup">Fixed warmup launch count; requires <paramref name="Iterations" />.</param>
/// <param name="Iterations">Fixed measured launch count; requires <paramref name="Warmup" />.</param>
/// <param name="SingleLaunch">Time the first and only launch, for kernels that consume their input.</param>
readonly record struct TileGymMeasureOptions(int? Warmup = null, int? Iterations = null, bool SingleLaunch = false);

/// <summary>Outcome of running one kernel: the selected variant and its measured execution time.</summary>
sealed record TileGymRun(TileGymKernel Kernel, TileGymSelection Selection, double KernelMilliseconds,
    double? HostMilliseconds);

sealed class TileGymRuntime : IDisposable
{
    readonly TileGymTuner _tuner;
    readonly CudaEventTileCppTimer _timer = new();

    public TileGymRuntime(int deviceOrdinal)
    {
        CuInit.EnsureInit();
        cuDeviceGet(out var device, deviceOrdinal).Ok();
        Device = device;
        Architecture = device.GetArchitecture();
        cuDeviceGetAttribute(out var smCount, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device).Ok();
        SmCount = smCount;
        Compiler = new TileCppCompiler(Architecture);
        cuDevicePrimaryCtxRetain(out var context, device).Ok();
        Context = context;
        try
        {
            cuCtxSetCurrent(context).Ok();
            cuStreamCreate(out var stream, 0).Ok();
            Stream = stream;
        }
        catch
        {
            cuDevicePrimaryCtxRelease(device).Ok();
            throw;
        }
        Kernels = new TileGymKernelCache(Compiler, context);
        _tuner = new TileGymTuner(_timer, Stream, Kernels.Get);
    }

    public CUdevice Device { get; }
    public CUcontext Context { get; }
    public CUstream Stream { get; }
    public int Architecture { get; }
    public int SmCount { get; }
    public TileCppCompiler Compiler { get; }
    public TileGymKernelCache Kernels { get; }
    public bool EnableAutotuning { get; init; } = true;

    public CudaBuffer<T> Allocate<T>(int length) where T : unmanaged => new(length);

    /// <summary>Gets the variants that will be compiled and considered for a kernel.</summary>
    public IReadOnlyList<TileGymVariant> GetVariants(TileGymKernel kernel)
    {
        ArgumentNullException.ThrowIfNull(kernel);
        return EnableAutotuning ? kernel.Variants : [kernel.Variants[0]];
    }

    /// <summary>Compiles in parallel, then loads, every variant used by the benchmarks before any launch.</summary>
    public TileGymPrecompileSummary Precompile(IEnumerable<TileGymBenchmark> benchmarks, int loadParallelism)
    {
        ArgumentNullException.ThrowIfNull(benchmarks);
        var specs = benchmarks.SelectMany(static b => b.Kernels).SelectMany(GetVariants).Select(static v => v.Spec);
        return Kernels.Prepare(specs, loadParallelism);
    }

    /// <summary>Launches a fixed (single variant) kernel, for example to produce inputs for another kernel.</summary>
    public void Launch(TileGymKernel kernel, TileGymLaunch launch)
    {
        ArgumentNullException.ThrowIfNull(kernel);
        ArgumentNullException.ThrowIfNull(launch);
        if (kernel.Variants.Count != 1)
        {
            throw new ArgumentException($"{kernel.Name} has multiple variants; use Run to select one.", nameof(kernel));
        }
        var variant = kernel.Variants[0];
        var compiled = Kernels.Get(variant.Spec);
        if (compiled.Error is { } error)
        {
            throw new InvalidOperationException($"Compilation or loading failed for {variant.Spec}.", error);
        }
        launch(compiled.Function, variant.Grid);
    }

    public void Synchronize() => cuStreamSynchronize(Stream).Ok();

    /// <summary>
    /// Selects the fastest valid variant, measures it, and validates its output. <paramref name="reset" /> restores
    /// inputs or outputs before every validated launch.
    /// </summary>
    public TileGymRun Run(TileGymKernel kernel, TileGymLaunch launch, Action? validate = null, Action? reset = null,
        TileGymMeasureOptions options = default)
    {
        ArgumentNullException.ThrowIfNull(kernel);
        ArgumentNullException.ThrowIfNull(launch);
        var variants = GetVariants(kernel);
        if (options.SingleLaunch)
        {
            if (variants.Count != 1)
            {
                throw new ArgumentException("Single-launch measurement requires a fixed kernel.", nameof(options));
            }
            return RunOnce(kernel, variants[0], launch, validate, reset);
        }

        var selection = _tuner.Select(kernel.Name, variants, launch, validate, reset);
        var function = selection.Compiled.Function;
        var grid = selection.Variant.Grid;
        void LaunchSelected() => launch(function, grid);
        var (kernelMilliseconds, hostMilliseconds) = options is { Warmup: { } warmup, Iterations: { } iterations }
            ? MeasureFixed(LaunchSelected, warmup, iterations)
            : Measure(LaunchSelected);
        if (validate is not null)
        {
            reset?.Invoke();
            LaunchSelected();
            Synchronize();
            validate();
        }
        return new(kernel, selection, kernelMilliseconds, hostMilliseconds);
    }

    TileGymRun RunOnce(TileGymKernel kernel, TileGymVariant variant, TileGymLaunch launch, Action? validate, Action? reset)
    {
        var compiled = Kernels.Get(variant.Spec);
        if (compiled.Error is { } error)
        {
            throw new InvalidOperationException($"Compilation or loading failed for {variant.Spec}.", error);
        }
        reset?.Invoke();
        cuEventCreate(out var start, 0).Ok();
        cuEventCreate(out var end, 0).Ok();
        try
        {
            var watch = Stopwatch.StartNew();
            cuEventRecord(start, Stream).Ok();
            launch(compiled.Function, variant.Grid);
            cuEventRecord(end, Stream).Ok();
            cuEventSynchronize(end).Ok();
            watch.Stop();
            cuEventElapsedTime(out var milliseconds, start, end).Ok();
            validate?.Invoke();
            var selection = new TileGymSelection(variant, compiled, watch.Elapsed.TotalMilliseconds, 0, 1, []);
            return new(kernel, selection, milliseconds, null);
        }
        finally
        {
            cuEventDestroy(start).Ok();
            cuEventDestroy(end).Ok();
        }
    }

    (double Kernel, double Host) Measure(Action launch)
    {
        var measurement = _timer.MeasureWithHost(launch, Stream, new TileCppTimingOptions());
        return (measurement.KernelMilliseconds, measurement.HostMilliseconds);
    }

    (double Kernel, double Host) MeasureFixed(Action launch, int warmupCount, int iterationCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(warmupCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(iterationCount);
        for (var i = 0; i < warmupCount; i++)
        {
            launch();
        }
        Synchronize();
        cuEventCreate(out var start, 0).Ok();
        cuEventCreate(out var end, 0).Ok();
        try
        {
            var host = Stopwatch.StartNew();
            cuEventRecord(start, Stream).Ok();
            for (var i = 0; i < iterationCount; i++)
            {
                launch();
            }
            cuEventRecord(end, Stream).Ok();
            cuEventSynchronize(end).Ok();
            host.Stop();
            cuEventElapsedTime(out var milliseconds, start, end).Ok();
            return (milliseconds / iterationCount, host.Elapsed.TotalMilliseconds / iterationCount);
        }
        finally
        {
            cuEventDestroy(start).Ok();
            cuEventDestroy(end).Ok();
        }
    }

    public void Dispose()
    {
        cuStreamSynchronize(Stream).Ok();
        Kernels.Dispose();
        cuStreamDestroy(Stream).Ok();
        cuCtxSetCurrent(default).Ok();
        cuDevicePrimaryCtxRelease(Device).Ok();
    }
}

sealed class CudaBuffer<T> : IDisposable where T : unmanaged
{
    public CudaBuffer(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        Length = length;
        ByteLength = checked((nuint)length * (nuint)Unsafe.SizeOf<T>());
        cuMemAlloc_v2(out var pointer, ByteLength).Ok();
        Pointer = pointer;
    }

    public int Length { get; }
    public nuint ByteLength { get; }
    public CUdeviceptr Pointer { get; }

    public unsafe void CopyFrom(ReadOnlySpan<T> source)
    {
        if (source.Length != Length)
        {
            throw new ArgumentException("Source length must match the device buffer.", nameof(source));
        }
        fixed (T* pointer = source)
        {
            cuMemcpyHtoD_v2(Pointer, (IntPtr)pointer, ByteLength).Ok();
        }
    }

    public unsafe T[] CopyToHost()
    {
        var destination = GC.AllocateUninitializedArray<T>(Length);
        fixed (T* pointer = destination)
        {
            cuMemcpyDtoH_v2((IntPtr)pointer, Pointer, ByteLength).Ok();
        }
        return destination;
    }

    public void Clear() => cuMemsetD8_v2(Pointer, 0, ByteLength).Ok();

    public void Dispose() => cuMemFree_v2(Pointer).Ok();
}
