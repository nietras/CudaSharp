using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using static CudaSharp.nvcuda;
using static CudaSharp.nvrtc;

namespace CudaSharp.ComparisonBenchmarks;

unsafe sealed class StraightLineGraphPerf : IDisposable
{
    const string KernelName = "straight_line_noop";
    const string KernelSource =
        """
        extern "C" __global__ void straight_line_noop()
        {
        }
        """;

    readonly int _graphLength;

    CUcontext _context;
    CUstream _stream;
    CUmodule _module;
    CUfunction _function;
    CUgraph _graph;
    CUgraphExec _graphExec;
    CUevent _startEvent;
    CUevent _endEvent;

    public StraightLineGraphPerf(int graphLength)
    {
        if (graphLength < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(graphLength), graphLength, "Graph length must be at least 1.");
        }

        _graphLength = graphLength;

        CuInit.EnsureInit();
        Setup();
    }

    public StraightLineGraphMetrics Measure(int warmupCount, int repetitionCount)
    {
        if (warmupCount < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(warmupCount), warmupCount, "Warmup count cannot be negative.");
        }

        if (repetitionCount < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(repetitionCount), repetitionCount, "Repetition count must be at least 1.");
        }

        DestroyGraphExec();
        var instantiationMilliseconds = InstantiateGraphExec();

        var firstLaunch = MeasureLaunch(syncAfterLaunch: true);

        for (var i = 0; i < warmupCount; i++)
        {
            MeasureLaunch(syncAfterLaunch: true);
        }

        var repeatLaunchApiMicroseconds = 0.0;
        var repeatLaunchTotalMicroseconds = 0.0;
        var repeatLaunchDeviceMicroseconds = 0.0;

        for (var i = 0; i < repetitionCount; i++)
        {
            var launch = MeasureLaunch(syncAfterLaunch: true);
            repeatLaunchApiMicroseconds += launch.ApiMicroseconds;
            repeatLaunchTotalMicroseconds += launch.TotalMicroseconds;
            repeatLaunchDeviceMicroseconds += launch.DeviceMicroseconds;
        }

        DestroyGraphExec();
        InstantiateGraphExec();
        var upload = MeasureUpload();

        return new StraightLineGraphMetrics(
            _graphLength,
            instantiationMilliseconds,
            upload.ApiMicroseconds,
            upload.TotalMicroseconds,
            firstLaunch.ApiMicroseconds,
            firstLaunch.TotalMicroseconds,
            repeatLaunchApiMicroseconds / repetitionCount,
            repeatLaunchTotalMicroseconds / repetitionCount,
            repeatLaunchDeviceMicroseconds / repetitionCount);
    }

    public void Dispose()
    {
        DestroyGraphExec();

        if (_graph.Value != IntPtr.Zero)
        {
            cuGraphDestroy(_graph).Ok();
            _graph = default;
        }

        if (_endEvent.Value != IntPtr.Zero)
        {
            cuEventDestroy(_endEvent).Ok();
            _endEvent = default;
        }

        if (_startEvent.Value != IntPtr.Zero)
        {
            cuEventDestroy(_startEvent).Ok();
            _startEvent = default;
        }

        if (_module.Value != IntPtr.Zero)
        {
            cuModuleUnload(_module).Ok();
            _module = default;
        }

        if (_stream.Value != IntPtr.Zero)
        {
            cuStreamDestroy(_stream).Ok();
            _stream = default;
        }

        if (_context.Value != IntPtr.Zero)
        {
            cuCtxDestroy(_context).Ok();
            _context = default;
        }
    }

    void Setup()
    {
        cuDeviceGet(out var device, 0).Ok();
        cuCtxCreate(out _context, CUctx_flags.CU_CTX_SCHED_SPIN, device).Ok();
        cuCtxSetCurrent(_context).Ok();
        cuStreamCreate(out _stream, 0).Ok();

        var image = CompileKernel(device, KernelSource, KernelName);
        cuModuleLoadData(out _module, image).Ok();
        cuModuleGetFunction(out _function, _module, KernelName).Ok();
        cuEventCreate(out _startEvent, (uint)CUevent_flags.CU_EVENT_DEFAULT).Ok();
        cuEventCreate(out _endEvent, (uint)CUevent_flags.CU_EVENT_DEFAULT).Ok();

        BuildGraph();
    }

    void BuildGraph()
    {
        cuGraphCreate(out _graph, 0).Ok();

        void** kernelParams = null;
        var nodeParams = new CUDA_KERNEL_NODE_PARAMS
        {
            func = _function,
            gridDimX = 1,
            gridDimY = 1,
            gridDimZ = 1,
            blockDimX = 1,
            blockDimY = 1,
            blockDimZ = 1,
            sharedMemBytes = 0,
            kernelParams = (IntPtr)kernelParams,
            extra = IntPtr.Zero,
        };

        cuGraphAddKernelNode(out var previousNode,
            _graph,
            [],
            0,
            nodeParams).Ok();

        if (_graphLength == 1)
        {
            return;
        }

        var dependencyStorage = stackalloc CUgraphNode[1];
        dependencyStorage[0] = previousNode;

        for (var i = 1; i < _graphLength; i++)
        {
            cuGraphAddKernelNode(out previousNode,
                _graph,
                new ReadOnlySpan<CUgraphNode>(dependencyStorage, 1),
                1,
                nodeParams).Ok();
            dependencyStorage[0] = previousNode;
        }
    }

    double InstantiateGraphExec()
    {
        Span<byte> logBuffer = stackalloc byte[2048];
        var startTimestamp = Stopwatch.GetTimestamp();
        var result = cuGraphInstantiate(out _graphExec,
            _graph,
            out var errorNode,
            logBuffer,
            (nuint)logBuffer.Length);
        var endTimestamp = Stopwatch.GetTimestamp();

        if (result.IsError())
        {
            var log = Encoding.UTF8.GetString(logBuffer).TrimEnd('\0');
            throw new InvalidOperationException(
                $"Straight-line graph instantiation failed with {result.ToStringFast()} at node {errorNode.Value}:\n{log}");
        }

        return Stopwatch.GetElapsedTime(startTimestamp, endTimestamp).TotalMilliseconds;
    }

    LaunchMeasurement MeasureLaunch(bool syncAfterLaunch)
    {
        cuEventRecord(_startEvent, _stream).Ok();

        var launchStart = Stopwatch.GetTimestamp();
        cuGraphLaunch(_graphExec, _stream).Ok();
        var launchEnd = Stopwatch.GetTimestamp();

        cuEventRecord(_endEvent, _stream).Ok();

        if (syncAfterLaunch)
        {
            cuStreamSynchronize(_stream).Ok();
        }

        var totalEnd = Stopwatch.GetTimestamp();
        cuEventElapsedTime(out var deviceMilliseconds, _startEvent, _endEvent).Ok();

        return new LaunchMeasurement(
            Stopwatch.GetElapsedTime(launchStart, launchEnd).TotalMicroseconds,
            Stopwatch.GetElapsedTime(launchStart, totalEnd).TotalMicroseconds,
            deviceMilliseconds * 1000.0);
    }

    UploadMeasurement MeasureUpload()
    {
        var uploadStart = Stopwatch.GetTimestamp();
        cuGraphUpload(_graphExec, _stream).Ok();
        var uploadEnd = Stopwatch.GetTimestamp();
        cuStreamSynchronize(_stream).Ok();
        var totalEnd = Stopwatch.GetTimestamp();

        return new UploadMeasurement(
            Stopwatch.GetElapsedTime(uploadStart, uploadEnd).TotalMicroseconds,
            Stopwatch.GetElapsedTime(uploadStart, totalEnd).TotalMicroseconds);
    }

    void DestroyGraphExec()
    {
        if (_graphExec.Value != IntPtr.Zero)
        {
            cuGraphExecDestroy(_graphExec).Ok();
            _graphExec = default;
        }
    }

    static byte[] CompileKernel(CUdevice device, string source, string kernelName)
    {
        cuDeviceComputeCapability(out var major, out var minor, device).Ok();
        var targetArchitecture = $"sm_{major}{minor}";

        nvrtcCreateProgram(out var program, source, kernelName, 0, [], []).Ok();
        try
        {
            var optionBytes = Encoding.UTF8.GetBytes($"--gpu-architecture={targetArchitecture}\0");
            nvrtcResult result;
            fixed (byte* optionPtr = optionBytes)
            {
                var optionPointers = stackalloc byte*[1];
                optionPointers[0] = optionPtr;
                result = nvrtcCompileProgram(program, 1, optionPointers);
            }

            if (result.IsError())
            {
                var log = GetCompileLog(program);
                if (!IsUnsupportedArchitecture(result, log))
                {
                    throw new InvalidOperationException(
                        $"Kernel compilation failed with {result.ToStringFast()}:\n{log}");
                }

                result = nvrtcCompileProgram(program, 0, []);
                if (result.IsError())
                {
                    throw new InvalidOperationException(
                        $"Kernel compilation fallback failed with {result.ToStringFast()}:\n{GetCompileLog(program)}");
                }

                nvrtcGetPTXSize(program, out var ptxSize).Ok();
                var ptx = new byte[ptxSize];
                nvrtcGetPTX(program, ptx).Ok();
                return ptx;
            }

            nvrtcGetCUBINSize(program, out var cubinSize).Ok();
            var cubin = new byte[cubinSize];
            nvrtcGetCUBIN(program, cubin).Ok();
            return cubin;
        }
        finally
        {
            nvrtcDestroyProgram(ref program).Ok();
        }
    }

    static bool IsUnsupportedArchitecture(nvrtcResult result, string log) =>
        result == nvrtcResult.NVRTC_ERROR_INVALID_OPTION &&
        log.Contains("unsupported gpu architecture", StringComparison.OrdinalIgnoreCase);

    static string GetCompileLog(nvrtcProgram program)
    {
        nvrtcGetProgramLogSize(program, out var logSize).Ok();
        var logBuffer = new byte[logSize];
        nvrtcGetProgramLog(program, logBuffer).Ok();
        return Encoding.UTF8.GetString(logBuffer).TrimEnd('\0');
    }

    readonly record struct LaunchMeasurement(
        double ApiMicroseconds,
        double TotalMicroseconds,
        double DeviceMicroseconds);

    readonly record struct UploadMeasurement(
        double ApiMicroseconds,
        double TotalMicroseconds);
}

readonly record struct StraightLineGraphMetrics(
    int GraphLength,
    double InstantiationMilliseconds,
    double UploadApiMicroseconds,
    double UploadTotalMicroseconds,
    double FirstLaunchApiMicroseconds,
    double FirstLaunchTotalMicroseconds,
    double RepeatLaunchApiMicroseconds,
    double RepeatLaunchTotalMicroseconds,
    double RepeatLaunchDeviceMicroseconds);