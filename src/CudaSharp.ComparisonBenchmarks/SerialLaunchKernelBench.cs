using System;
using System.IO;
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
    const string DeviceLaunchSchedulerKernelName = "serial_device_graph_tail_launch";
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
    const string DeviceLaunchSchedulerSource =
        """
        extern "C" __global__ void serial_device_graph_tail_launch(
            cudaGraphExec_t graphExec,
            int* launchStatus)
        {
            if ((blockIdx.x | blockIdx.y | blockIdx.z | threadIdx.x | threadIdx.y | threadIdx.z) != 0)
            {
                return;
            }

            launchStatus[0] = (int)cudaGraphLaunch(graphExec, cudaStreamGraphTailLaunch);
        }
        """;

    CUcontext _context;
    CUmodule _module;
    CUfunction _initFunction;
    CUfunction _function;
    CUmodule _deviceLaunchSchedulerModule;
    CUfunction _deviceLaunchSchedulerFunction;
    CUstream _stream;
    CUgraph _graph;
    CUgraphExec _graphExec;
    CUgraphExec _deviceLaunchGraphExec;
    CUgraph _trueDeviceLaunchGraph;
    CUgraphExec _trueDeviceLaunchGraphExec;
    CUgraph _capturedGraph;
    CUgraphExec _capturedGraphExec;
    CUdeviceptr _buffer0;
    CUdeviceptr _buffer1;
    CUdeviceptr _accumulator;
    CUdeviceptr _deviceLaunchStatus;

    CUdeviceptr* _graphInputs;
    CUdeviceptr* _graphOutputs;
    CUdeviceptr* _graphAccumulator;
    int* _graphIncrements;
    void** _graphKernelParams;
    CUgraphExec* _deviceLaunchSchedulerGraphExecArgument;
    CUdeviceptr* _deviceLaunchSchedulerStatusArgument;
    void** _deviceLaunchSchedulerKernelParams;

    bool _deviceGraphLaunchSupported;

    public SerialLaunchKernelBench() => CuInit.EnsureInit();

    [Params(256)]
    public int SerialLaunchCount { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        cuDeviceGet(out var device, 0).Ok();
        cuDeviceComputeCapability(out var major, out var minor, device).Ok();
        Console.WriteLine($"DEVICE ARCH: sm_{major}{minor}");
        _deviceGraphLaunchSupported = major >= 9;

        cuCtxCreate(out _context, CUctx_flags.CU_CTX_SCHED_SPIN, device).Ok();
        cuCtxSetCurrent(_context).Ok();
        cuStreamCreate(out _stream, 0).Ok();

        var image = CompileKernel(device, KernelSource, KernelName);
        LoadModule(out _module, image, nameof(SerialLaunchKernelBench));
        cuModuleGetFunction(out _initFunction, _module, InitKernelName).Ok();
        cuModuleGetFunction(out _function, _module, KernelName).Ok();

        const nuint bufferSize = sizeof(int);
        cuMemAlloc(out _buffer0, bufferSize).Ok();
        cuMemAlloc(out _buffer1, bufferSize).Ok();
        cuMemAlloc(out _accumulator, bufferSize).Ok();
        cuMemAlloc(out _deviceLaunchStatus, bufferSize).Ok();

        BuildSerialGraph();
        if (_deviceGraphLaunchSupported)
        {
            BuildTrueDeviceLaunchSchedulerGraph(device);
        }
        BuildCapturedGraph();
        cuGraphUpload(_graphExec, _stream).Ok();
        cuGraphUpload(_deviceLaunchGraphExec, _stream).Ok();
        if (_deviceGraphLaunchSupported)
        {
            cuGraphUpload(_trueDeviceLaunchGraphExec, _stream).Ok();
        }
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

        if (_trueDeviceLaunchGraphExec.Value != IntPtr.Zero)
        { cuGraphExecDestroy(_trueDeviceLaunchGraphExec).Ok(); }

        if (_trueDeviceLaunchGraph.Value != IntPtr.Zero)
        { cuGraphDestroy(_trueDeviceLaunchGraph).Ok(); }

        if (_graphExec.Value != IntPtr.Zero)
        { cuGraphExecDestroy(_graphExec).Ok(); }

        if (_deviceLaunchGraphExec.Value != IntPtr.Zero)
        { cuGraphExecDestroy(_deviceLaunchGraphExec).Ok(); }

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

        if (_deviceLaunchStatus.Value != IntPtr.Zero)
        { cuMemFree(_deviceLaunchStatus).Ok(); }

        if (_deviceLaunchSchedulerModule.Value != IntPtr.Zero)
        { cuModuleUnload(_deviceLaunchSchedulerModule).Ok(); }

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
    public void cuGraphLaunch_DeviceLaunchCapableSerialTripleBuffer_StreamSync()
    {
        cuGraphLaunch(_deviceLaunchGraphExec, _stream).Ok();
        cuStreamSynchronize(_stream).Ok();
    }

    [Benchmark]
    public void cuGraphLaunch_TrueDeviceTailLaunchSerialTripleBuffer_StreamSync()
    {
        if (!_deviceGraphLaunchSupported) { return; }
        cuGraphLaunch(_trueDeviceLaunchGraphExec, _stream).Ok();
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

    void BuildTrueDeviceLaunchSchedulerGraph(CUdevice device)
    {
        var image = CompileLinkedKernel(
            device,
            DeviceLaunchSchedulerSource,
            DeviceLaunchSchedulerKernelName,
            GetCudaDeviceRuntimeLibraryPath());

        var result = cuModuleLoadDataEx(out _deviceLaunchSchedulerModule, image, 0, null, null);
        if (result.IsError())
        {
            if (result == CUresult.CUDA_ERROR_INVALID_IMAGE)
            {
                Console.WriteLine($"Module load failed for {DeviceLaunchSchedulerKernelName} with {result.ToStringFast()}. Skipping true device launch.");
                _deviceGraphLaunchSupported = false;
                return;
            }
            throw new InvalidOperationException($"Module load failed for {DeviceLaunchSchedulerKernelName} with {result.ToStringFast()}");
        }
        cuModuleGetFunction(out _deviceLaunchSchedulerFunction, _deviceLaunchSchedulerModule, DeviceLaunchSchedulerKernelName).Ok();

        _deviceLaunchSchedulerGraphExecArgument =
            (CUgraphExec*)NativeMemory.Alloc(1, (nuint)sizeof(CUgraphExec));
        _deviceLaunchSchedulerStatusArgument =
            (CUdeviceptr*)NativeMemory.Alloc(1, (nuint)sizeof(CUdeviceptr));
        _deviceLaunchSchedulerKernelParams =
            (void**)NativeMemory.Alloc(2, (nuint)sizeof(void*));

        *_deviceLaunchSchedulerGraphExecArgument = _deviceLaunchGraphExec;
        *_deviceLaunchSchedulerStatusArgument = _deviceLaunchStatus;
        _deviceLaunchSchedulerKernelParams[0] = _deviceLaunchSchedulerGraphExecArgument;
        _deviceLaunchSchedulerKernelParams[1] = _deviceLaunchSchedulerStatusArgument;

        cuGraphCreate(out _trueDeviceLaunchGraph, 0).Ok();

        var nodeParams = new CUDA_KERNEL_NODE_PARAMS
        {
            func = _deviceLaunchSchedulerFunction,
            gridDimX = 1,
            gridDimY = 1,
            gridDimZ = 1,
            blockDimX = 1,
            blockDimY = 1,
            blockDimZ = 1,
            sharedMemBytes = 0,
            kernelParams = (IntPtr)_deviceLaunchSchedulerKernelParams,
            extra = IntPtr.Zero,
        };

        cuGraphAddKernelNode(out _,
            _trueDeviceLaunchGraph,
            [],
            0,
            nodeParams).Ok();

        Span<byte> logBuffer = stackalloc byte[2048];
        var instantiateResult = cuGraphInstantiate(out _trueDeviceLaunchGraphExec,
            _trueDeviceLaunchGraph,
            out var errorNode,
            logBuffer,
            (nuint)logBuffer.Length);
        if (instantiateResult.IsError())
        {
            var log = Encoding.UTF8.GetString(logBuffer).TrimEnd('\0');
            throw new InvalidOperationException(
                $"True device-launch scheduler graph instantiation failed with {instantiateResult.ToStringFast()} at node {errorNode.Value}:\n{log}");
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

        var instantiateParams = new CUDA_GRAPH_INSTANTIATE_PARAMS
        {
            flags = (ulong)CUgraphInstantiate_flags.CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH,
            hUploadStream = default,
            hErrNode_out = default,
            result_out = default,
        };
        instantiateResult = cuGraphInstantiateWithParams(
            out _deviceLaunchGraphExec,
            _graph,
            ref instantiateParams);
        if (instantiateResult.IsError())
        {
            throw new InvalidOperationException(
                $"Device-launch-capable graph instantiation failed with {instantiateResult.ToStringFast()} and {instantiateParams.result_out} at node {instantiateParams.hErrNode_out.Value}.");
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

        cuGraphLaunch(_deviceLaunchGraphExec, _stream).Ok();
        cuStreamSynchronize(_stream).Ok();
        ValidateResults(nameof(cuGraphLaunch_DeviceLaunchCapableSerialTripleBuffer_StreamSync));

        ResetDeviceLaunchStatus();
        if (_deviceGraphLaunchSupported)
        {
            cuGraphLaunch(_trueDeviceLaunchGraphExec, _stream).Ok();
            cuStreamSynchronize(_stream).Ok();
            ValidateDeviceLaunchStatus(nameof(cuGraphLaunch_TrueDeviceTailLaunchSerialTripleBuffer_StreamSync));
            ValidateResults(nameof(cuGraphLaunch_TrueDeviceTailLaunchSerialTripleBuffer_StreamSync));
        }

        cuGraphLaunch(_capturedGraphExec, _stream).Ok();
        cuStreamSynchronize(_stream).Ok();
        ValidateResults(nameof(cuGraphLaunch_CapturedSerialTripleBuffer_StreamSync));
    }

    void ResetDeviceLaunchStatus() =>
        cuMemsetD32(_deviceLaunchStatus, unchecked((uint)-1), 1).Ok();

    void ValidateDeviceLaunchStatus(string benchmarkName)
    {
        var launchStatus = -1;
        cuMemcpyDtoH((IntPtr)(&launchStatus), _deviceLaunchStatus, sizeof(int)).Ok();

        if (launchStatus != 0)
        {
            throw new InvalidOperationException(
                $"{benchmarkName} produced device launch status={launchStatus}, expected 0.");
        }
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

        if (_deviceLaunchSchedulerGraphExecArgument != null)
        {
            NativeMemory.Free(_deviceLaunchSchedulerGraphExecArgument);
            _deviceLaunchSchedulerGraphExecArgument = null;
        }

        if (_deviceLaunchSchedulerStatusArgument != null)
        {
            NativeMemory.Free(_deviceLaunchSchedulerStatusArgument);
            _deviceLaunchSchedulerStatusArgument = null;
        }

        if (_deviceLaunchSchedulerKernelParams != null)
        {
            NativeMemory.Free(_deviceLaunchSchedulerKernelParams);
            _deviceLaunchSchedulerKernelParams = null;
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

    static void GetComputeCapability(CUdevice device, out int major, out int minor)
    {
        cuDeviceGetAttribute(out major, (CUdevice_attribute)75, device).Ok();
        cuDeviceGetAttribute(out minor, (CUdevice_attribute)76, device).Ok();
    }

    static byte[] CompileKernel(CUdevice device, string source, string kernelName)
    {
        GetComputeCapability(device, out var major, out var minor);
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

    static unsafe byte[] CompileLinkedKernel(
        CUdevice device,
        string source,
        string kernelName,
        string deviceRuntimeLibraryPath)
    {
        GetComputeCapability(device, out var major, out var minor);

        var compileMode = GetLinkedKernelCompileMode();
        var compileArchitecture = $"compute_{major}{minor}";
        var linkArchitecture = $"sm_{major}{minor}";
        var compileOptions = GetLinkedKernelCompileOptions(compileArchitecture, compileMode);

        nvrtcCreateProgram(out var program, source, kernelName, 0, [], []).Ok();
        try
        {
            var compileResult = CompileProgram(program, compileOptions);
            if (compileResult.IsError())
            {
                throw new InvalidOperationException(
                    $"Kernel compilation failed with {compileResult.ToStringFast()}:\n{GetCompileLog(program)}");
            }

            nvrtcGetPTXSize(program, out var ptxSize).Ok();
            var ptx = new byte[ptxSize];
            nvrtcGetPTX(program, ptx).Ok();
            return LinkPtx(ptx, kernelName, compileMode, linkArchitecture, deviceRuntimeLibraryPath);
        }
        finally
        {
            nvrtcDestroyProgram(ref program).Ok();
        }
    }

    static unsafe nvrtcResult CompileProgram(nvrtcProgram program, string[] options)
    {
        var optionPointers = stackalloc byte*[options.Length];
        var allocatedOptions = new IntPtr[options.Length];

        try
        {
            for (var i = 0; i < options.Length; i++)
            {
                var optionBytes = Encoding.UTF8.GetBytes($"{options[i]}\0");
                allocatedOptions[i] = Marshal.AllocHGlobal(optionBytes.Length);
                Marshal.Copy(optionBytes, 0, allocatedOptions[i], optionBytes.Length);
                optionPointers[i] = (byte*)allocatedOptions[i];
            }

            return nvrtcCompileProgram(program, options.Length, optionPointers);
        }
        finally
        {
            for (var i = 0; i < allocatedOptions.Length; i++)
            {
                if (allocatedOptions[i] != IntPtr.Zero)
                {
                    Marshal.FreeHGlobal(allocatedOptions[i]);
                }
            }
        }
    }

    static unsafe byte[] LinkPtx(
        byte[] ptx,
        string kernelName,
        string compileMode,
        string targetArchitecture,
        string deviceRuntimeLibraryPath)
    {
        var linkOptions = new[]
        {
            $"-arch={targetArchitecture}",
            "-verbose",
        };

        nvJitLink.nvJitLinkCreate(out var linkState, linkOptions).Ok();
        try
        {
            fixed (byte* ptxPtr = ptx)
            {
                nvJitLink.nvJitLinkAddData(
                    linkState,
                    nvJitLink.nvJitLinkInputType.NVJITLINK_INPUT_PTX,
                    ptxPtr,
                    (nuint)ptx.Length,
                    kernelName).Ok();
            }

            nvJitLink.nvJitLinkAddFile(
                linkState,
                nvJitLink.nvJitLinkInputType.NVJITLINK_INPUT_LIBRARY,
                deviceRuntimeLibraryPath).Ok();

            nvJitLink.nvJitLinkComplete(linkState).Ok();

            nvJitLink.nvJitLinkGetLinkedCubinSize(linkState, out var cubinSize).Ok();
            var cubin = new byte[cubinSize];
            nvJitLink.nvJitLinkGetLinkedCubin(linkState, cubin).Ok();
            DumpLinkedKernelArtifacts(kernelName, compileMode, ptx, cubin, GetLinkLog(linkState));
            return cubin;
        }
        catch (CudaException<nvJitLink.nvJitLinkResult> exception)
        {
            throw new InvalidOperationException(
                $"nvJitLink failed with {exception.Result.ToStringFast()}:\n{GetLinkLog(linkState)}",
                exception);
        }
        finally
        {
            nvJitLink.nvJitLinkDestroy(ref linkState).Ok();
        }
    }

    static string GetLinkLog(nvJitLink.nvJitLinkHandle linkState)
    {
        var errorLog = nvJitLink.nvJitLinkGetErrorLogString(linkState);
        var infoLog = nvJitLink.nvJitLinkGetInfoLogString(linkState);

        if (string.IsNullOrWhiteSpace(errorLog))
        {
            return string.IsNullOrWhiteSpace(infoLog) ? "No linker log output." : infoLog;
        }

        if (string.IsNullOrWhiteSpace(infoLog))
        {
            return errorLog;
        }

        return $"{errorLog}\n{infoLog}";
    }

    static unsafe void LoadLibraryKernelFunction(
        ReadOnlySpan<byte> image,
        string kernelName,
        out CUlibrary library,
        out CUfunction function)
    {
        Span<byte> infoLogBuffer = stackalloc byte[8192];
        Span<byte> errorLogBuffer = stackalloc byte[8192];

        fixed (byte* infoLogPtr = infoLogBuffer)
        fixed (byte* errorLogPtr = errorLogBuffer)
        {
            var options = stackalloc CUjit_option[4];
            var optionValues = stackalloc void*[4];

            options[0] = CUjit_option.CU_JIT_INFO_LOG_BUFFER;
            optionValues[0] = infoLogPtr;
            options[1] = CUjit_option.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
            optionValues[1] = (void*)(nuint)infoLogBuffer.Length;
            options[2] = CUjit_option.CU_JIT_ERROR_LOG_BUFFER;
            optionValues[2] = errorLogPtr;
            options[3] = CUjit_option.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
            optionValues[3] = (void*)(nuint)errorLogBuffer.Length;

            var result = cuLibraryLoadData(
                out library,
                image,
                options,
                optionValues,
                4,
                null,
                null,
                0);
            if (result.IsError())
            {
                var infoLog = Encoding.UTF8.GetString(infoLogBuffer).TrimEnd('\0');
                var errorLog = Encoding.UTF8.GetString(errorLogBuffer).TrimEnd('\0');
                throw new InvalidOperationException(
                    $"Library load failed for '{kernelName}' with {result.ToStringFast()}:\n{FormatModuleLoadLog(infoLog, errorLog)}");
            }
        }

        cuLibraryGetModule(out var module, library).Ok();
        cuModuleGetFunction(out function, module, kernelName).Ok();
    }

    static unsafe void LoadModule(out CUmodule module, ReadOnlySpan<byte> image, string moduleName)
    {
        Span<byte> infoLogBuffer = stackalloc byte[8192];
        Span<byte> errorLogBuffer = stackalloc byte[8192];

        fixed (byte* infoLogPtr = infoLogBuffer)
        fixed (byte* errorLogPtr = errorLogBuffer)
        {
            var options = stackalloc CUjit_option[4];
            var optionValues = stackalloc void*[4];

            options[0] = CUjit_option.CU_JIT_INFO_LOG_BUFFER;
            optionValues[0] = infoLogPtr;
            options[1] = CUjit_option.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
            optionValues[1] = (void*)(nuint)infoLogBuffer.Length;
            options[2] = CUjit_option.CU_JIT_ERROR_LOG_BUFFER;
            optionValues[2] = errorLogPtr;
            options[3] = CUjit_option.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
            optionValues[3] = (void*)(nuint)errorLogBuffer.Length;

            var result = cuModuleLoadDataEx(out module, image, 4, options, optionValues);
            if (result.IsError())
            {
                var infoLog = Encoding.UTF8.GetString(infoLogBuffer).TrimEnd('\0');
                var errorLog = Encoding.UTF8.GetString(errorLogBuffer).TrimEnd('\0');
                throw new InvalidOperationException(
                    $"Module load failed for '{moduleName}' with {result.ToStringFast()}:\n{FormatModuleLoadLog(infoLog, errorLog)}");
            }
        }
    }

    static string FormatModuleLoadLog(string infoLog, string errorLog)
    {
        if (string.IsNullOrWhiteSpace(infoLog) && string.IsNullOrWhiteSpace(errorLog))
        {
            return "No driver log output.";
        }

        if (string.IsNullOrWhiteSpace(infoLog))
        {
            return errorLog;
        }

        if (string.IsNullOrWhiteSpace(errorLog))
        {
            return infoLog;
        }

        return $"Error log:\n{errorLog}\nInfo log:\n{infoLog}";
    }

    static string GetLinkedKernelCompileMode()
    {
        var mode = Environment.GetEnvironmentVariable("CUDASHARP_LINKED_KERNEL_MODE");
        return string.Equals(mode, "rdc", StringComparison.OrdinalIgnoreCase) ? "rdc" : "ewp";
    }

    static string[] GetLinkedKernelCompileOptions(string compileArchitecture, string compileMode)
    {
        return compileMode switch
        {
            "rdc" =>
            [
                $"--gpu-architecture={compileArchitecture}",
                "--relocatable-device-code=true",
                "--std=c++17",
            ],
            _ =>
            [
                $"--gpu-architecture={compileArchitecture}",
                "--extensible-whole-program",
                "--std=c++17",
            ],
        };
    }

    static void DumpLinkedKernelArtifacts(
        string kernelName,
        string compileMode,
        byte[] ptx,
        byte[] cubin,
        string linkLog)
    {
        var dumpRoot = Environment.GetEnvironmentVariable("CUDASHARP_DUMP_LINKED_KERNELS");
        if (string.IsNullOrWhiteSpace(dumpRoot))
        {
            return;
        }

        Directory.CreateDirectory(dumpRoot);

        var safeKernelNameChars = kernelName.ToCharArray();
        var invalidFileNameChars = Path.GetInvalidFileNameChars();
        for (var i = 0; i < safeKernelNameChars.Length; i++)
        {
            if (Array.IndexOf(invalidFileNameChars, safeKernelNameChars[i]) >= 0)
            {
                safeKernelNameChars[i] = '_';
            }
        }

        var safeKernelName = new string(safeKernelNameChars);
        var artifactPrefix = Path.Combine(dumpRoot, $"{safeKernelName}.{compileMode}");

        File.WriteAllBytes($"{artifactPrefix}.ptx", ptx);
        File.WriteAllBytes($"{artifactPrefix}.cubin", cubin);
        File.WriteAllText($"{artifactPrefix}.link.log", linkLog);
    }

    static string GetCudaDeviceRuntimeLibraryPath()
    {
        var cudaPath = Environment.GetEnvironmentVariable("CUDA_PATH");
        if (string.IsNullOrWhiteSpace(cudaPath))
        {
            throw new InvalidOperationException("CUDA_PATH is not set.");
        }

        var deviceRuntimeLibraryPath = Path.Combine(cudaPath, "lib", "x64", "cudadevrt.lib");
        if (!File.Exists(deviceRuntimeLibraryPath))
        {
            throw new InvalidOperationException(
                $"The CUDA device runtime library was not found at '{deviceRuntimeLibraryPath}'.");
        }

        return deviceRuntimeLibraryPath;
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
}
