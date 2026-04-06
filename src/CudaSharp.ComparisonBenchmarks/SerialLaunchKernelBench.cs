using System;
using System.Runtime.InteropServices;
using System.Text;
using BenchmarkDotNet.Attributes;
using static CudaSharp.nvcuda;
using static CudaSharp.nvrtc;

namespace CudaSharp.ComparisonBenchmarks;

[BenchmarkCategory("LaunchKernel")]
public unsafe class SerialLaunchKernelBench
{
    const string InitKernelName = "serial_init";
    const string KernelName = "serial_accumulate";
    const string KernelSource =
        """
        extern "C" __global__ void serial_init(
            int* output,
            int* accumulator)
        {
            output[0] = 1;
            accumulator[0] = 1;
        }

        extern "C" __global__ void serial_accumulate(
            const int* input,
            int* output,
            int* accumulator,
            int increment)
        {
            const int value = input[0] + increment;
            output[0] = value;
            accumulator[0] += value;
        }
        """;

    CUcontext _context;
    CUmodule _module;
    CUfunction _initFunction;
    CUfunction _function;
    CUstream _stream;
    CUgraph _graph;
    CUgraphExec _graphExec;
    CUgraph _capturedGraph;
    CUgraphExec _capturedGraphExec;
    CUdeviceptr _buffer0;
    CUdeviceptr _buffer1;
    CUdeviceptr _accumulator;

    CUdeviceptr* _graphInputs;
    CUdeviceptr* _graphOutputs;
    CUdeviceptr* _graphAccumulator;
    int* _graphIncrements;
    void** _graphKernelParams;

    public SerialLaunchKernelBench() => CuInit.EnsureInit();

    [Params(256)]
    public int SerialLaunchCount { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        cuDeviceGet(out var device, 0).Ok();
        cuCtxCreate(out _context, CUctx_flags.CU_CTX_SCHED_SPIN, device).Ok();
        cuCtxSetCurrent(_context).Ok();
        cuStreamCreate(out _stream, 0).Ok();

        var ptx = CompileKernel(KernelSource, KernelName);
        cuModuleLoadData(out _module, ptx).Ok();
        cuModuleGetFunction(out _initFunction, _module, InitKernelName).Ok();
        cuModuleGetFunction(out _function, _module, KernelName).Ok();

        const nuint bufferSize = sizeof(int);
        cuMemAlloc(out _buffer0, bufferSize).Ok();
        cuMemAlloc(out _buffer1, bufferSize).Ok();
        cuMemAlloc(out _accumulator, bufferSize).Ok();

        BuildSerialGraph();
        BuildCapturedGraph();
        cuGraphUpload(_graphExec, _stream).Ok();
        cuGraphUpload(_capturedGraphExec, _stream).Ok();
        cuStreamSynchronize(_stream).Ok();
        ValidateImplementations();
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        if (_context.Value == IntPtr.Zero) { return; }

        cuCtxSetCurrent(_context).Ok();

        if (_capturedGraphExec.Value != IntPtr.Zero)
        { cuGraphExecDestroy(_capturedGraphExec).Ok(); }

        if (_capturedGraph.Value != IntPtr.Zero)
        { cuGraphDestroy(_capturedGraph).Ok(); }

        if (_graphExec.Value != IntPtr.Zero)
        { cuGraphExecDestroy(_graphExec).Ok(); }

        if (_graph.Value != IntPtr.Zero)
        { cuGraphDestroy(_graph).Ok(); }

        if (_stream.Value != IntPtr.Zero)
        { cuStreamDestroy(_stream).Ok(); }

        if (_buffer0.Value != IntPtr.Zero)
        { cuMemFree(_buffer0).Ok(); }

        if (_buffer1.Value != IntPtr.Zero)
        { cuMemFree(_buffer1).Ok(); }

        if (_accumulator.Value != IntPtr.Zero)
        { cuMemFree(_accumulator).Ok(); }

        if (_module.Value != IntPtr.Zero)
        { cuModuleUnload(_module).Ok(); }

        FreeGraphStorage();
        cuCtxDestroy(_context).Ok();
    }

    [Benchmark(Baseline = true)]
    public void cuLaunchKernel_Raw_SerialTripleBuffer_StreamSync()
    {
        LaunchSerialRaw();
        cuStreamSynchronize(_stream).Ok();
    }

    [Benchmark]
    public void cuGraphLaunch_SerialTripleBuffer_StreamSync()
    {
        cuGraphLaunch(_graphExec, _stream).Ok();
        cuStreamSynchronize(_stream).Ok();
    }

    [Benchmark]
    public void cuGraphLaunch_CapturedSerialTripleBuffer_StreamSync()
    {
        cuGraphLaunch(_capturedGraphExec, _stream).Ok();
        cuStreamSynchronize(_stream).Ok();
    }

    void LaunchSerialRaw()
    {
        LaunchInitRaw();

        if (SerialLaunchCount == 1)
        {
            return;
        }

        var accumulator = _accumulator;
        var input = _buffer1;
        var output = _buffer0;
        var increment = 0;
        var kernelParams = stackalloc void*[] { &input, &output, &accumulator, &increment };

        for (var i = 1; i < SerialLaunchCount; i++)
        {
            input = (i & 1) == 1 ? _buffer1 : _buffer0;
            output = (i & 1) == 1 ? _buffer0 : _buffer1;
            increment = i + 1;

            cuLaunchKernel(_function,
                1, 1, 1,
                1, 1, 1,
                0, _stream,
                kernelParams, null).Ok();
        }
    }

    void BuildCapturedGraph()
    {
        cuStreamBeginCapture(_stream, CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_GLOBAL).Ok();
        LaunchSerialRaw();
        cuStreamEndCapture(_stream, out _capturedGraph).Ok();

        Span<byte> logBuffer = stackalloc byte[2048];
        var instantiateResult = cuGraphInstantiate(out _capturedGraphExec,
            _capturedGraph,
            out var errorNode,
            logBuffer,
            (nuint)logBuffer.Length);
        if (instantiateResult.IsError())
        {
            var log = Encoding.UTF8.GetString(logBuffer).TrimEnd('\0');
            throw new InvalidOperationException(
                $"Captured graph instantiation failed with {instantiateResult.ToStringFast()} at node {errorNode.Value}:\n{log}");
        }
    }

    void LaunchInitRaw()
    {
        var output = _buffer1;
        var accumulator = _accumulator;
        var kernelParams = stackalloc void*[] { &output, &accumulator };

        cuLaunchKernel(_initFunction,
            1, 1, 1,
            1, 1, 1,
            0, _stream,
            kernelParams, null).Ok();
    }

    void BuildSerialGraph()
    {
        cuGraphCreate(out _graph, 0).Ok();

        _graphInputs = (CUdeviceptr*)NativeMemory.Alloc((nuint)SerialLaunchCount, (nuint)sizeof(CUdeviceptr));
        _graphOutputs = (CUdeviceptr*)NativeMemory.Alloc((nuint)SerialLaunchCount, (nuint)sizeof(CUdeviceptr));
        _graphAccumulator = (CUdeviceptr*)NativeMemory.Alloc(1, (nuint)sizeof(CUdeviceptr));
        _graphIncrements = (int*)NativeMemory.Alloc((nuint)SerialLaunchCount, (nuint)sizeof(int));
        _graphKernelParams = (void**)NativeMemory.Alloc((nuint)(2 + ((SerialLaunchCount - 1) * 4)), (nuint)sizeof(void*));

        *_graphAccumulator = _accumulator;

        var initOutput = _buffer1;
        var initKernelParams = _graphKernelParams;
        initKernelParams[0] = &initOutput;
        initKernelParams[1] = _graphAccumulator;

        var initNodeParams = new CUDA_KERNEL_NODE_PARAMS
        {
            func = _initFunction,
            gridDimX = 1,
            gridDimY = 1,
            gridDimZ = 1,
            blockDimX = 1,
            blockDimY = 1,
            blockDimZ = 1,
            sharedMemBytes = 0,
            kernelParams = (IntPtr)initKernelParams,
            extra = IntPtr.Zero,
        };

        cuGraphAddKernelNode(out var previousNode,
            _graph,
            [],
            0,
            initNodeParams).Ok();

        if (SerialLaunchCount == 1)
        {
            InstantiateGraph();
            return;
        }

        var dependencyStorage = stackalloc CUgraphNode[1];

        for (var i = 1; i < SerialLaunchCount; i++)
        {
            _graphInputs[i] = (i & 1) == 1 ? _buffer1 : _buffer0;
            _graphOutputs[i] = (i & 1) == 1 ? _buffer0 : _buffer1;
            _graphIncrements[i] = i + 1;

            var kernelParams = _graphKernelParams + 2 + ((i - 1) * 4);
            kernelParams[0] = &_graphInputs[i];
            kernelParams[1] = &_graphOutputs[i];
            kernelParams[2] = _graphAccumulator;
            kernelParams[3] = &_graphIncrements[i];

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

            dependencyStorage[0] = previousNode;
            cuGraphAddKernelNode(out previousNode,
                _graph,
                new ReadOnlySpan<CUgraphNode>(dependencyStorage, 1),
                1,
                nodeParams).Ok();
        }

        InstantiateGraph();
    }

    void InstantiateGraph()
    {
        Span<byte> logBuffer = stackalloc byte[2048];
        var instantiateResult = cuGraphInstantiate(out _graphExec,
            _graph,
            out var errorNode,
            logBuffer,
            (nuint)logBuffer.Length);
        if (instantiateResult.IsError())
        {
            var log = Encoding.UTF8.GetString(logBuffer).TrimEnd('\0');
            throw new InvalidOperationException(
                $"Graph instantiation failed with {instantiateResult.ToStringFast()} at node {errorNode.Value}:\n{log}");
        }
    }

    void ValidateImplementations()
    {
        LaunchSerialRaw();
        cuStreamSynchronize(_stream).Ok();
        ValidateResults(nameof(cuLaunchKernel_Raw_SerialTripleBuffer_StreamSync));

        cuGraphLaunch(_graphExec, _stream).Ok();
        cuStreamSynchronize(_stream).Ok();
        ValidateResults(nameof(cuGraphLaunch_SerialTripleBuffer_StreamSync));

        cuGraphLaunch(_capturedGraphExec, _stream).Ok();
        cuStreamSynchronize(_stream).Ok();
        ValidateResults(nameof(cuGraphLaunch_CapturedSerialTripleBuffer_StreamSync));
    }

    void ValidateResults(string benchmarkName)
    {
        var expectedState = CheckedExpectedState(SerialLaunchCount);
        var expectedAccumulator = CheckedExpectedAccumulator(SerialLaunchCount);

        var actualState = 0;
        var actualAccumulator = 0;

        cuMemcpyDtoH((IntPtr)(&actualState), GetFinalStateBuffer(), sizeof(int)).Ok();
        cuMemcpyDtoH((IntPtr)(&actualAccumulator), _accumulator, sizeof(int)).Ok();

        if (actualState != expectedState || actualAccumulator != expectedAccumulator)
        {
            throw new InvalidOperationException(
                $"{benchmarkName} produced state={actualState}, accumulator={actualAccumulator}, " +
                $"expected state={expectedState}, accumulator={expectedAccumulator}.");
        }
    }

    CUdeviceptr GetFinalStateBuffer() =>
        (SerialLaunchCount & 1) == 0 ? _buffer0 : _buffer1;

    void FreeGraphStorage()
    {
        if (_graphInputs != null)
        {
            NativeMemory.Free(_graphInputs);
            _graphInputs = null;
        }

        if (_graphOutputs != null)
        {
            NativeMemory.Free(_graphOutputs);
            _graphOutputs = null;
        }

        if (_graphAccumulator != null)
        {
            NativeMemory.Free(_graphAccumulator);
            _graphAccumulator = null;
        }

        if (_graphIncrements != null)
        {
            NativeMemory.Free(_graphIncrements);
            _graphIncrements = null;
        }

        if (_graphKernelParams != null)
        {
            NativeMemory.Free(_graphKernelParams);
            _graphKernelParams = null;
        }
    }

    static int CheckedExpectedState(int count)
    {
        checked
        {
            return count * (count + 1) / 2;
        }
    }

    static int CheckedExpectedAccumulator(int count)
    {
        checked
        {
            return count * (count + 1) * (count + 2) / 6;
        }
    }

    static byte[] CompileKernel(string source, string kernelName)
    {
        nvrtcCreateProgram(out var program, source, kernelName, 0, [], []).Ok();
        try
        {
            var result = nvrtcCompileProgram(program, 0, []);
            if (result.IsError())
            {
                throw new InvalidOperationException(
                    $"Kernel compilation failed with {result.ToStringFast()}:\n{GetCompileLog(program)}");
            }

            nvrtcGetPTXSize(program, out var ptxSize).Ok();
            var ptx = new byte[ptxSize];
            nvrtcGetPTX(program, ptx).Ok();
            return ptx;
        }
        finally
        {
            nvrtcDestroyProgram(ref program).Ok();
        }
    }

    static string GetCompileLog(nvrtcProgram program)
    {
        nvrtcGetProgramLogSize(program, out var logSize).Ok();
        var logBuffer = new byte[logSize];
        nvrtcGetProgramLog(program, logBuffer).Ok();
        return Encoding.UTF8.GetString(logBuffer).TrimEnd('\0');
    }
}
