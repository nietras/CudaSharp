using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using BenchmarkDotNet.Attributes;
using static CudaSharp.cudart;
using static CudaSharp.nvcuda;
using static CudaSharp.nvrtc;

namespace CudaSharp.ComparisonBenchmarks;

[BenchmarkCategory("LaunchKernel")]
public unsafe class SerialLaunchKernelBench
{

    const string dumpRoot = "dumps";

    const string InitKernelName = "serial_init";
    const string KernelName = "serial_accumulate";
    const string DeviceLaunchSchedulerKernelName = "serial_device_graph_tail_launch";
    const string DeviceLaunchSchedulerFireAndForgetKernelName = "serial_device_graph_fire_and_forget_launch";
    const string DeviceLaunchSchedulerNoOpKernelName = "serial_device_graph_tail_launch_noop";
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
        #include <cuda_device_runtime_api.h>
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
    const string DeviceLaunchSchedulerFireAndForgetSource =
        """
        #include <cuda_device_runtime_api.h>
        extern "C" __global__ void serial_device_graph_fire_and_forget_launch(
            cudaGraphExec_t graphExec,
            int* launchStatus)
        {
            if ((blockIdx.x | blockIdx.y | blockIdx.z | threadIdx.x | threadIdx.y | threadIdx.z) != 0)
            {
                return;
            }

            launchStatus[0] = (int)cudaGraphLaunch(graphExec, cudaStreamGraphFireAndForget);
        }
        """;
    const string DeviceLaunchSchedulerNoOpSource =
        """
        #include <cuda_device_runtime_api.h>
        extern "C" __global__ void serial_device_graph_tail_launch_noop(
            cudaGraphExec_t graphExec,
            int* launchStatus)
        {
            if ((blockIdx.x | blockIdx.y | blockIdx.z | threadIdx.x | threadIdx.y | threadIdx.z) != 0)
            {
                return;
            }

            if (graphExec == nullptr)
            {
                launchStatus[0] = -1;
                return;
            }

            launchStatus[0] = 0;
        }
        """;

    CUdevice _device;
    CUcontext _context;
    CUmodule _module;
    CUfunction _initFunction;
    CUfunction _function;
    CUmodule _deviceLaunchSchedulerModule;
    CUlibrary _deviceLaunchSchedulerLibrary;
    CUfunction _deviceLaunchSchedulerFunction;
    CUmodule _deviceFireAndForgetSchedulerModule;
    CUlibrary _deviceFireAndForgetSchedulerLibrary;
    CUfunction _deviceFireAndForgetSchedulerFunction;
    CUstream _stream;
    CUgraph _graph;
    CUgraphExec _graphExec;
    CUgraphExec _deviceLaunchGraphExec;
    CUgraph _trueDeviceLaunchGraph;
    CUgraphExec _trueDeviceLaunchGraphExec;
    CUgraph _trueDeviceFireAndForgetLaunchGraph;
    CUgraphExec _trueDeviceFireAndForgetLaunchGraphExec;
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
    void** _deviceFireAndForgetSchedulerKernelParams;

    bool _deviceGraphLaunchSupported;

    public SerialLaunchKernelBench() => CuInit.EnsureInit();

    [Params(256)]
    public int SerialLaunchCount { get; set; } = 256;

    [GlobalSetup]
    public void Setup()
    {
        try
        {
            cudaFree(IntPtr.Zero);
            Console.WriteLine("[DEBUG] CUDA Runtime initialized successfully.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[WARNING] Failed to load CUDA Runtime DLL: {ex.Message}");
        }

        cuDeviceGet(out var device, 0).Ok();
        _device = device;
        cuDeviceComputeCapability(out var major, out var minor, device).Ok();
        Console.WriteLine($"DEVICE ARCH: sm_{major}{minor}");
        _deviceGraphLaunchSupported = major >= 9;

        cuCtxCreate_v2(out _context, CUctx_flags.CU_CTX_SCHED_SPIN, device).Ok();
        cuCtxSetCurrent(_context).Ok();
        cuStreamCreate(out _stream, 0).Ok();

        var image = CompileKernel(device, KernelSource, KernelName);
        Console.WriteLine($"[DEBUG] Kernel compiled.");
        LoadModule(out _module, image, nameof(SerialLaunchKernelBench));
        Console.WriteLine($"[DEBUG] Module loaded: {_module.Value}");
        cuModuleGetFunction(out _initFunction, _module, InitKernelName).Ok();
        cuModuleGetFunction(out _function, _module, KernelName).Ok();
        Console.WriteLine($"[DEBUG] Functions obtained.");

        const nuint bufferSize = sizeof(int);
        cuCtxGetCurrent(out var activeCtx).Ok();
        Console.WriteLine($"[DEBUG] Active context before alloc: {activeCtx.Value}");
        Console.WriteLine($"[DEBUG] Allocating buffer0...");
        cuMemAlloc_v2(out _buffer0, bufferSize).Ok();
        cuMemAlloc_v2(out _buffer1, bufferSize).Ok();
        cuMemAlloc_v2(out _accumulator, bufferSize).Ok();
        cuMemAlloc_v2(out _deviceLaunchStatus, bufferSize).Ok();

        BuildSerialGraph();
        if (_deviceGraphLaunchSupported)
        {
            BuildTrueDeviceLaunchSchedulerGraph(device);
            BuildTrueDeviceFireAndForgetSchedulerGraph(device);
        }
        BuildCapturedGraph();
        cuGraphUpload(_graphExec, _stream).Ok();
        cuGraphUpload(_deviceLaunchGraphExec, _stream).Ok();
        if (_deviceGraphLaunchSupported)
        {
            cuGraphUpload(_trueDeviceLaunchGraphExec, _stream).Ok();
            cuGraphUpload(_trueDeviceFireAndForgetLaunchGraphExec, _stream).Ok();
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

        if (_trueDeviceFireAndForgetLaunchGraphExec.Value != IntPtr.Zero)
        { cuGraphExecDestroy(_trueDeviceFireAndForgetLaunchGraphExec).Ok(); }

        if (_trueDeviceFireAndForgetLaunchGraph.Value != IntPtr.Zero)
        { cuGraphDestroy(_trueDeviceFireAndForgetLaunchGraph).Ok(); }

        if (_graphExec.Value != IntPtr.Zero)
        { cuGraphExecDestroy(_graphExec).Ok(); }

        if (_deviceLaunchGraphExec.Value != IntPtr.Zero)
        { cuGraphExecDestroy(_deviceLaunchGraphExec).Ok(); }

        if (_graph.Value != IntPtr.Zero)
        { cuGraphDestroy(_graph).Ok(); }

        if (_stream.Value != IntPtr.Zero)
        { cuStreamDestroy(_stream).Ok(); }

        if (_buffer0.Value != IntPtr.Zero)
        { cuMemFree_v2(_buffer0).Ok(); }

        if (_buffer1.Value != IntPtr.Zero)
        { cuMemFree_v2(_buffer1).Ok(); }

        if (_accumulator.Value != IntPtr.Zero)
        { cuMemFree_v2(_accumulator).Ok(); }

        if (_deviceLaunchStatus.Value != IntPtr.Zero)
        { cuMemFree_v2(_deviceLaunchStatus).Ok(); }

        if (_deviceLaunchSchedulerLibrary.Value != IntPtr.Zero)
        { cuLibraryUnload(_deviceLaunchSchedulerLibrary).Ok(); }
        else if (_deviceLaunchSchedulerModule.Value != IntPtr.Zero)
        { cuModuleUnload(_deviceLaunchSchedulerModule).Ok(); }

        if (_deviceFireAndForgetSchedulerLibrary.Value != IntPtr.Zero)
        { cuLibraryUnload(_deviceFireAndForgetSchedulerLibrary).Ok(); }
        else if (_deviceFireAndForgetSchedulerModule.Value != IntPtr.Zero)
        { cuModuleUnload(_deviceFireAndForgetSchedulerModule).Ok(); }

        if (_module.Value != IntPtr.Zero)
        { cuModuleUnload(_module).Ok(); }

        FreeGraphStorage();
        if (_context.Value != IntPtr.Zero)
        {
            cuCtxSetCurrent(default).Ok();
            cuCtxDestroy(_context).Ok();
        }
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
        if (!_deviceGraphLaunchSupported)
        {
            throw new PlatformNotSupportedException("Device-side graph launch is not supported on this platform/configuration (e.g., Windows WDDM driver mode).");
        }
        cuGraphLaunch(_trueDeviceLaunchGraphExec, _stream).Ok();
        cuStreamSynchronize(_stream).Ok();
    }

    [Benchmark]
    public void cuGraphLaunch_TrueDeviceFireAndForgetSerialTripleBuffer_StreamSync()
    {
        if (!_deviceGraphLaunchSupported)
        {
            throw new PlatformNotSupportedException("Device-side graph launch is not supported on this platform/configuration (e.g., Windows WDDM driver mode).");
        }
        cuGraphLaunch(_trueDeviceFireAndForgetLaunchGraphExec, _stream).Ok();
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
        var kernelParams = stackalloc void*[4];

        for (var i = 1; i < SerialLaunchCount; i++)
        {
            input = (i & 1) == 1 ? _buffer1 : _buffer0;
            output = (i & 1) == 1 ? _buffer0 : _buffer1;
            increment = i + 1;

            kernelParams[0] = &input;
            kernelParams[1] = &output;
            kernelParams[2] = &accumulator;
            kernelParams[3] = &increment;

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
        var compileMode = GetLinkedKernelCompileMode();
        GetComputeCapability(device, out var major, out var minor);
        var compileArchitecture = $"compute_{major}{minor}";
        var linkArchitecture = $"sm_{major}{minor}";
        var deviceRuntimeLibraryPath = GetCudaDeviceRuntimeLibraryPath();
        var artifactPrefix = GetFullLinkedKernelArtifactPrefix(DeviceLaunchSchedulerKernelName, compileMode);
        var image = CompileLinkedKernel(
            device,
            DeviceLaunchSchedulerSource,
            DeviceLaunchSchedulerKernelName,
            deviceRuntimeLibraryPath,
            compileMode);

        try
        {
            LoadLibraryModule(out _deviceLaunchSchedulerModule, out _deviceLaunchSchedulerLibrary, image, nameof(BuildTrueDeviceLaunchSchedulerGraph));
        }
        catch (InvalidOperationException exception) when (exception.Message.Contains(CUresult.CUDA_ERROR_INVALID_IMAGE.ToStringFast(), StringComparison.Ordinal))
        {
            cuDriverGetVersion(out var driverVersion).Ok();
            var alternateCompileMode = GetAlternateLinkedKernelCompileMode(compileMode);
            var alternateProblematicProbeDiagnostic = ProbeLinkedKernelLoad(
                device,
                DeviceLaunchSchedulerSource,
                DeviceLaunchSchedulerKernelName,
                deviceRuntimeLibraryPath,
                alternateCompileMode);
            var noOpSameModeProbeDiagnostic = ProbeLinkedKernelLoad(
                device,
                DeviceLaunchSchedulerNoOpSource,
                DeviceLaunchSchedulerNoOpKernelName,
                deviceRuntimeLibraryPath,
                compileMode);
            var noOpAlternateModeProbeDiagnostic = ProbeLinkedKernelLoad(
                device,
                DeviceLaunchSchedulerNoOpSource,
                DeviceLaunchSchedulerNoOpKernelName,
                deviceRuntimeLibraryPath,
                alternateCompileMode);
            var fireAndForgetSameModeProbeDiagnostic = ProbeLinkedKernelLoad(
                device,
                DeviceLaunchSchedulerFireAndForgetSource,
                DeviceLaunchSchedulerFireAndForgetKernelName,
                deviceRuntimeLibraryPath,
                compileMode);
            var fireAndForgetAlternateModeProbeDiagnostic = ProbeLinkedKernelLoad(
                device,
                DeviceLaunchSchedulerFireAndForgetSource,
                DeviceLaunchSchedulerFireAndForgetKernelName,
                deviceRuntimeLibraryPath,
                alternateCompileMode);

            Console.WriteLine(
                $"Module load failed for {DeviceLaunchSchedulerKernelName} with {CUresult.CUDA_ERROR_INVALID_IMAGE.ToStringFast()}. " +
                $"driverVersion={driverVersion}, CUDA_PATH='{Environment.GetEnvironmentVariable("CUDA_PATH")}', " +
                $"compileMode={compileMode}, compileArchitecture={compileArchitecture}, linkArchitecture={linkArchitecture}, " +
                $"deviceRuntimeLibraryPath='{deviceRuntimeLibraryPath}', artifactDumpRoot='{dumpRoot}'.");
            Console.WriteLine($"Module load exception: {exception.Message}");
            Console.WriteLine($"Primary linked artifact prefix: '{artifactPrefix}'");
            Console.WriteLine(
                $"Inspect linked artifacts with: cuobjdump --dump-elf '{artifactPrefix}.cubin', " +
                $"cuobjdump --dump-sass '{artifactPrefix}.cubin', nvdisasm '{artifactPrefix}.cubin'.");
            Console.WriteLine(
                $"Linked artifact files: '{artifactPrefix}.ptx', '{artifactPrefix}.cubin', '{artifactPrefix}.link.log'.");
            Console.WriteLine(
                "Probe interpretation: no-op success means linked cudadevrt images load; fire-and-forget success isolates the failure to tail launch; fire-and-forget failure means any device-side cudaGraphLaunch is being rejected.");
            Console.WriteLine(alternateProblematicProbeDiagnostic);
            Console.WriteLine(noOpSameModeProbeDiagnostic);
            Console.WriteLine(noOpAlternateModeProbeDiagnostic);
            Console.WriteLine(fireAndForgetSameModeProbeDiagnostic);
            Console.WriteLine(fireAndForgetAlternateModeProbeDiagnostic);

            if (exception.Message.Contains("No driver log output.", StringComparison.Ordinal))
            {
                Console.WriteLine(
                    "The CUDA driver returned no module-load diagnostics, which usually points to a malformed or incompatible cubin, or a toolkit/driver mismatch.");
            }

            if (exception.Message.Contains("No driver log output.", StringComparison.Ordinal))
            {
                _deviceGraphLaunchSupported = false;
                return;
            }

            throw;
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
        var kernelParamsCount = (nuint)(2 + ((SerialLaunchCount - 1) * 4));
        _graphKernelParams = (void**)NativeMemory.Alloc(kernelParamsCount, (nuint)sizeof(void*));

        *_graphAccumulator = _accumulator;

        _graphOutputs[0] = _buffer1;
        var initKernelParams = _graphKernelParams;
        initKernelParams[0] = &_graphOutputs[0];
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

        ResetDeviceLaunchStatus();
        if (_deviceGraphLaunchSupported)
        {
            cuGraphLaunch(_trueDeviceFireAndForgetLaunchGraphExec, _stream).Ok();
            cuStreamSynchronize(_stream).Ok();
            ValidateDeviceLaunchStatus(nameof(cuGraphLaunch_TrueDeviceFireAndForgetSerialTripleBuffer_StreamSync));
            ValidateResults(nameof(cuGraphLaunch_TrueDeviceFireAndForgetSerialTripleBuffer_StreamSync));
        }

        cuGraphLaunch(_capturedGraphExec, _stream).Ok();
        cuStreamSynchronize(_stream).Ok();
        ValidateResults(nameof(cuGraphLaunch_CapturedSerialTripleBuffer_StreamSync));
    }

    void ResetDeviceLaunchStatus() =>
        cuMemsetD32_v2(_deviceLaunchStatus, unchecked((uint)-1), 1).Ok();

    void ValidateDeviceLaunchStatus(string benchmarkName)
    {
        var launchStatus = -1;
        cuMemcpyDtoH_v2((IntPtr)(&launchStatus), _deviceLaunchStatus, sizeof(int)).Ok();

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

        cuMemcpyDtoH_v2((IntPtr)(&actualState), GetFinalStateBuffer(), sizeof(int)).Ok();
        cuMemcpyDtoH_v2((IntPtr)(&actualAccumulator), _accumulator, sizeof(int)).Ok();

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

        if (_deviceFireAndForgetSchedulerKernelParams != null)
        {
            NativeMemory.Free(_deviceFireAndForgetSchedulerKernelParams);
            _deviceFireAndForgetSchedulerKernelParams = null;
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
        return CompileLinkedKernel(device, source, kernelName, deviceRuntimeLibraryPath, GetLinkedKernelCompileMode());
    }

    static unsafe byte[] CompileLinkedKernel(
        CUdevice device,
        string source,
        string kernelName,
        string deviceRuntimeLibraryPath,
        string compileMode)
    {
        GetComputeCapability(device, out var major, out var minor);

        var compileArchitecture = $"compute_{major}{minor}";
        var linkArchitecture = $"sm_{major}{minor}";
        var compileOptions = GetLinkedKernelCompileOptions(compileArchitecture, compileMode);

        Console.WriteLine($"[DEBUG] NVRTC Options: {string.Join(" ", compileOptions)}");

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

    static unsafe void LoadLibraryModule(out CUmodule module, out CUlibrary library, ReadOnlySpan<byte> image, string moduleName)
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

            var result = cuLibraryLoadData(out library, image, options, optionValues, 4, null, null, 0);
            if (result.IsError())
            {
                var infoLog = Encoding.UTF8.GetString(infoLogBuffer).TrimEnd('\0');
                var errorLog = Encoding.UTF8.GetString(errorLogBuffer).TrimEnd('\0');
                throw new InvalidOperationException(
                    $"Library load failed for '{moduleName}' with {result.ToStringFast()}:\n{FormatModuleLoadLog(infoLog, errorLog)}");
            }

            var moduleResult = cuLibraryGetModule(out module, library);
            if (moduleResult.IsError())
            {
                cuLibraryUnload(library).Ok();
                library = default;
                var infoLog = Encoding.UTF8.GetString(infoLogBuffer).TrimEnd('\0');
                var errorLog = Encoding.UTF8.GetString(errorLogBuffer).TrimEnd('\0');
                throw new InvalidOperationException(
                    $"Library GetModule failed for '{moduleName}' with {moduleResult.ToStringFast()}:\n{FormatModuleLoadLog(infoLog, errorLog)}");
            }
        }
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
                var tempPath = Path.Combine(Path.GetTempPath(), $"{moduleName}_{Guid.NewGuid()}.cubin");
                try
                {
                    File.WriteAllBytes(tempPath, image.ToArray());
                    var fallbackResult = cuModuleLoad(out module, tempPath);
                    if (fallbackResult == CUresult.CUDA_SUCCESS)
                    {
                        Console.WriteLine($"[INFO] Successfully loaded '{moduleName}' via cuModuleLoad file-fallback.");
                        return;
                    }
                }
                catch
                {
                    // Ignore fallback write errors
                }
                finally
                {
                    try { if (File.Exists(tempPath)) { File.Delete(tempPath); } } catch { }
                }

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

    static string GetAlternateLinkedKernelCompileMode(string compileMode) =>
        string.Equals(compileMode, "rdc", StringComparison.OrdinalIgnoreCase) ? "ewp" : "rdc";

    static string[] GetLinkedKernelCompileOptions(string compileArchitecture, string compileMode)
    {
        var cudaPath = Environment.GetEnvironmentVariable("CUDA_PATH");
        var includeOption = string.IsNullOrWhiteSpace(cudaPath) ? "" : $"-I{Path.Combine(cudaPath, "include")}";

        return compileMode switch
        {
            "rdc" =>
            [
                $"--gpu-architecture={compileArchitecture}",
                "--relocatable-device-code=true",
                "--std=c++17",
                includeOption,
            ],
            _ =>
            [
                $"--gpu-architecture={compileArchitecture}",
                "--extensible-whole-program",
                "--std=c++17",
                includeOption,
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
        Directory.CreateDirectory(dumpRoot);

        var artifactPrefix = GetLinkedKernelArtifactPrefix(dumpRoot, kernelName, compileMode);

        File.WriteAllBytes($"{artifactPrefix}.ptx", ptx);
        File.WriteAllBytes($"{artifactPrefix}.cubin", cubin);
        File.WriteAllText($"{artifactPrefix}.link.log", linkLog);
    }

    static string GetLinkedKernelArtifactPrefix(string dumpRoot, string kernelName, string compileMode)
    {
        var safeKernelName = GetSafeArtifactKernelName(kernelName);
        return Path.Combine(dumpRoot, $"{safeKernelName}.{compileMode}");
    }

    static string GetFullLinkedKernelArtifactPrefix(string kernelName, string compileMode) =>
        Path.GetFullPath(GetLinkedKernelArtifactPrefix(dumpRoot, kernelName, compileMode));

    static string GetSafeArtifactKernelName(string kernelName)
    {
        var safeKernelNameChars = kernelName.ToCharArray();
        var invalidFileNameChars = Path.GetInvalidFileNameChars();
        for (var i = 0; i < safeKernelNameChars.Length; i++)
        {
            if (Array.IndexOf(invalidFileNameChars, safeKernelNameChars[i]) >= 0)
            {
                safeKernelNameChars[i] = '_';
            }
        }

        return new string(safeKernelNameChars);
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

    static string ProbeLinkedKernelLoad(
        CUdevice device,
        string source,
        string kernelName,
        string deviceRuntimeLibraryPath,
        string compileMode)
    {
        var artifactPrefix = GetFullLinkedKernelArtifactPrefix(kernelName, compileMode);

        try
        {
            var image = CompileLinkedKernel(device, source, kernelName, deviceRuntimeLibraryPath, compileMode);
            LoadLibraryModule(out var module, out var library, image, $"{kernelName}.{compileMode}.probe");
            try
            {
                return $"Alternate linked-kernel load probe succeeded for compileMode={compileMode}. Artifact prefix: '{artifactPrefix}'.";
            }
            finally
            {
                if (library.Value != IntPtr.Zero)
                {
                    cuLibraryUnload(library).Ok();
                }
            }
        }
        catch (Exception probeException)
        {
            return $"Alternate linked-kernel load probe failed for compileMode={compileMode}. Artifact prefix: '{artifactPrefix}'. {probeException.Message}";
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

    void BuildTrueDeviceFireAndForgetSchedulerGraph(CUdevice device)
    {
        var compileMode = GetLinkedKernelCompileMode();
        var deviceRuntimeLibraryPath = GetCudaDeviceRuntimeLibraryPath();
        var image = CompileLinkedKernel(
            device,
            DeviceLaunchSchedulerFireAndForgetSource,
            DeviceLaunchSchedulerFireAndForgetKernelName,
            deviceRuntimeLibraryPath,
            compileMode);

        LoadLibraryModule(out _deviceFireAndForgetSchedulerModule, out _deviceFireAndForgetSchedulerLibrary, image, nameof(BuildTrueDeviceFireAndForgetSchedulerGraph));
        cuModuleGetFunction(out _deviceFireAndForgetSchedulerFunction, _deviceFireAndForgetSchedulerModule, DeviceLaunchSchedulerFireAndForgetKernelName).Ok();

        _deviceFireAndForgetSchedulerKernelParams =
            (void**)NativeMemory.Alloc(2, (nuint)sizeof(void*));

        _deviceFireAndForgetSchedulerKernelParams[0] = _deviceLaunchSchedulerGraphExecArgument;
        _deviceFireAndForgetSchedulerKernelParams[1] = _deviceLaunchSchedulerStatusArgument;

        cuGraphCreate(out _trueDeviceFireAndForgetLaunchGraph, 0).Ok();

        var nodeParams = new CUDA_KERNEL_NODE_PARAMS
        {
            func = _deviceFireAndForgetSchedulerFunction,
            gridDimX = 1,
            gridDimY = 1,
            gridDimZ = 1,
            blockDimX = 1,
            blockDimY = 1,
            blockDimZ = 1,
            sharedMemBytes = 0,
            kernelParams = (IntPtr)_deviceFireAndForgetSchedulerKernelParams,
            extra = IntPtr.Zero,
        };

        cuGraphAddKernelNode(out _,
            _trueDeviceFireAndForgetLaunchGraph,
            [],
            0,
            nodeParams).Ok();

        Span<byte> logBuffer = stackalloc byte[2048];
        var instantiateResult = cuGraphInstantiate(out _trueDeviceFireAndForgetLaunchGraphExec,
            _trueDeviceFireAndForgetLaunchGraph,
            out var errorNode,
            logBuffer,
            (nuint)logBuffer.Length);
        if (instantiateResult.IsError())
        {
            var log = Encoding.UTF8.GetString(logBuffer).TrimEnd('\0');
            throw new InvalidOperationException(
                $"True device fire-and-forget scheduler graph instantiation failed with {instantiateResult.ToStringFast()} at node {errorNode.Value}:\n{log}");
        }
    }
}
