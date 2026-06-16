using System;
using System.Buffers.Binary;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Net.Http;
using System.Runtime.CompilerServices;
using System.Text;
using static CudaSharp.nvcuda;
using static CudaSharp.nvrtc;

namespace CudaSharp.Mnist;

public unsafe partial class Program
{
    static int BatchSize = 128;
    const int ClassCount = 10;
    const int ImageRows = 28;
    const int ImageCols = 28;
    const int TrainImagesCount = 51200; // 400 batches of size 128
    const int TestImagesCount = 10240;   // Padded to multiple of BatchSize (128 * 80 = 10240)

    public static void Main(string[] args)
    {
        Console.WriteLine("CudaSharp Ultra-Fast MNIST CNN Training Simulator");

        var version = "ALL";
        var profile = false;
        for (var i = 0; i < args.Length; i++)
        {
            if (args[i] == "--version")
            {
                version = args[i + 1].ToUpperInvariant();
            }
            else if (args[i] == "--profile")
            {
                profile = true;
            }
        }
        Console.WriteLine($"[CONFIG] Network Version: {version} (Profile Mode: {profile})");

        if (string.Equals(version, "ALL", StringComparison.OrdinalIgnoreCase))
        {
            var comparisonResults = new System.Collections.Generic.List<(NetworkConfig Config, double MeanAccuracy, double MeanTime)>();
            foreach (var config in OrderedNetworkConfigs)
            {
                Console.WriteLine();
                Console.WriteLine($"[COMPARE] Running {config.Name}...");
                comparisonResults.Add(RunConfig(config, profile));
            }

            Console.WriteLine();
            Console.WriteLine("### ALL CONFIG COMPARISON ###");
            foreach (var result in comparisonResults)
            {
                Console.WriteLine($"{result.Config.Name,-4} | Accuracy: {result.MeanAccuracy,6:F2}% | Time: {result.MeanTime,9:F3} ms");
            }
            return;
        }

        var activeConfig = ResolveNetworkConfig(version);
        RunConfig(activeConfig, profile);
    }

    static NetworkConfig ResolveNetworkConfig(string version)
    {
        if (NetworkConfigs.TryGetValue(version, out var config))
        {
            return config;
        }

        if (version.StartsWith("V0", StringComparison.OrdinalIgnoreCase)
            || (version.StartsWith("V", StringComparison.OrdinalIgnoreCase)
                && version.Length > 2
                && int.TryParse(version.AsSpan(1), out _)))
        {
            if (int.TryParse(version.AsSpan(1), out var num) && num >= 1 && num <= 99)
            {
                var batchSize = (num % 4) switch
                {
                    0 => 64,
                    1 => 128,
                    2 => 256,
                    3 => 512,
                    _ => 256
                };
                var maxLR = ((num / 4) % 4) switch
                {
                    0 => 0.003f,
                    1 => 0.006f,
                    2 => 0.009f,
                    3 => 0.012f,
                    _ => 0.006f
                };
                var totalSteps = ((num / 16) % 3) switch
                {
                    0 => 150,
                    1 => 300,
                    2 => 450,
                    _ => 300
                };
                return new NetworkConfig
                {
                    Name = version,
                    CudaSource = CudaSourceV7,
                    IsHalf = true,
                    IsV7Based = true,
                    BatchSize = batchSize,
                    Conv1FilterCount = 8,
                    Conv1FilterSize = 3,
                    Conv2FilterCount = 16,
                    Conv2FilterSize = 3,
                    Pool2OutSize = 5,
                    HasFC1 = false,
                    BatchesPerEpoch = 51200 / batchSize,
                    TotalSteps = totalSteps,
                    MaxLR = maxLR
                };
            }
        }

        throw new ArgumentException($"Unknown version: {version}");
    }

    static (NetworkConfig Config, double MeanAccuracy, double MeanTime) RunConfig(NetworkConfig activeConfig, bool profile)
    {
        BatchSize = activeConfig.BatchSize;
        double meanAcc = 0.0;
        double meanTime = 0.0;
        CuInit.EnsureInit();

        cuDeviceGet(out var device, 0).Ok();
        Span<byte> deviceNameBytes = stackalloc byte[256];
        cuDeviceGetName(deviceNameBytes, 256, device).Ok();
        var deviceName = Encoding.UTF8.GetString(deviceNameBytes).TrimEnd('\0');
        cuDeviceComputeCapability(out var major, out var minor, device).Ok();

        Console.WriteLine($"[DEVICE] Loaded active GPU: {deviceName} (sm_{major}{minor})");

        var dataDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "mnist_data");
        if (!Directory.Exists(dataDir))
        {
            Directory.CreateDirectory(dataDir);
        }

        var trainImagesPath = Path.Combine(dataDir, "train-images-idx3-ubyte.gz");
        var trainLabelsPath = Path.Combine(dataDir, "train-labels-idx1-ubyte.gz");
        var testImagesPath = Path.Combine(dataDir, "t10k-images-idx3-ubyte.gz");
        var testLabelsPath = Path.Combine(dataDir, "t10k-labels-idx1-ubyte.gz");

        EnsureDatasetFile(trainImagesPath, "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz");
        EnsureDatasetFile(trainLabelsPath, "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz");
        EnsureDatasetFile(testImagesPath, "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz");
        EnsureDatasetFile(testLabelsPath, "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz");

        Console.WriteLine("[DATA] Parsing Gzip compressed idx dataset files in-memory...");
        var (h_trainImages, trainImagesLoaded) = ParseImagesGz(trainImagesPath, TrainImagesCount);
        var h_trainLabels = ParseLabelsGz(trainLabelsPath, TrainImagesCount);
        var (h_testImages, testImagesLoaded) = ParseImagesGz(testImagesPath, TestImagesCount);
        var h_testLabels = ParseLabelsGz(testLabelsPath, TestImagesCount);

        Console.WriteLine($"[DATA] Loaded {trainImagesLoaded} train images and {testImagesLoaded} test images successfully!");

        var conv1Chunks = Math.Max(16, activeConfig.BatchSize / 8);
        var conv2Chunks = Math.Max(16, activeConfig.BatchSize / 8);

        Console.WriteLine("[JIT] Compiling CUDA kernels...");
        var cudaSourceStr = activeConfig.UseCustomCudaSource
            ? activeConfig.CudaSource
            : (activeConfig.IsHalf ? CudaKernelLibrary.BuildLeNetSource(activeConfig) : activeConfig.CudaSource);
        CUcontext context = default;
        CUstream stream = default;
        CUmodule module = default;
        nvrtcProgram program = default;
        try
        {
            cuDriverGetVersion(out var driverVersion).Ok();
            nvrtcVersion(out var nvrtcMajor, out var nvrtcMinor).Ok();
            var maxSupportedCudaMajor = Math.Min(nvrtcMajor, driverVersion / 1000);
            Console.WriteLine($"[JIT] NVRTC Version: {nvrtcMajor}.{nvrtcMinor}, Driver Max CUDA Version: {driverVersion / 1000}.{(driverVersion % 1000) / 10}");

            var archMajor = major;
            var archMinor = minor;
            if (maxSupportedCudaMajor < 13 && archMajor >= 12)
            {
                archMajor = 9;
                archMinor = 0;
            }
            if (maxSupportedCudaMajor < 12 && archMajor >= 9)
            {
                archMajor = 8;
                archMinor = 6;
            }
            if (maxSupportedCudaMajor < 11 && archMajor >= 8)
            {
                archMajor = 7;
                archMinor = 5;
            }

            byte[]? ptx = null;
            byte** optionPointers = stackalloc byte*[32]; // Max 32 options, moved out of the loop to prevent stack overflow
            while (true)
            {
                var optionsList = new System.Collections.Generic.List<string>
                {
                    $"--gpu-architecture=compute_{archMajor}{archMinor}",
                    "--std=c++17",
                    "--use_fast_math",
                    $"-DBATCH_SIZE={activeConfig.BatchSize}",
                    $"-DBATCHES_PER_EPOCH={activeConfig.BatchesPerEpoch}",
                    $"-DTOTAL_STEPS={activeConfig.TotalSteps}",
                    $"-DMAX_LR={activeConfig.MaxLR}f",
                    $"-DFC1_OUTPUTS={activeConfig.FC1Outputs}",
                    $"-DFC1_INPUTS={activeConfig.FC1Inputs}",
                    $"-DCONV1_CHUNKS={conv1Chunks}",
                    $"-DCONV2_CHUNKS={conv2Chunks}",
                    $"-DFILTER1_SIZE={activeConfig.Conv1FilterSize}",
                    $"-DFILTER2_SIZE={activeConfig.Conv2FilterSize}",
                    $"-DPOOL1_SIZE={activeConfig.Pool1OutSize}",
                    $"-DPOOL2_SIZE={activeConfig.Pool2OutSize}",
                    $"-DWEIGHT_DECAY={(activeConfig.HasWeightDecay ? activeConfig.WeightDecayRate : 0.0f).ToString("0.0##########", System.Globalization.CultureInfo.InvariantCulture)}f"
                };
                var includePath = ResolveCudaIncludePath();
                if (!string.IsNullOrEmpty(includePath))
                {
                    optionsList.Add($"-I{includePath}");
                }

                var options = optionsList.ToArray();
                var optionBytes = new byte[options.Length][];
                for (var i = 0; i < options.Length; i++)
                {
                    optionBytes[i] = Encoding.UTF8.GetBytes($"{options[i]}\0");
                    fixed (byte* optPtr = optionBytes[i])
                    {
                        optionPointers[i] = optPtr;
                    }
                }

                nvrtcCreateProgram(out program, cudaSourceStr, "mnist_kernels", 0, [], []).Ok();
                var compileResult = nvrtcCompileProgram(program, options.Length, optionPointers);
                if (compileResult.IsError())
                {
                    nvrtcGetProgramLogSize(program, out var logSize).Ok();
                    var logBuffer = new byte[logSize];
                    nvrtcGetProgramLog(program, logBuffer).Ok();
                    var logStr = Encoding.UTF8.GetString(logBuffer);
                    nvrtcDestroyProgram(ref program).Ok();
                    throw new InvalidOperationException($"NVRTC Compilation failed:\n{logStr}");
                }

                nvrtcGetPTXSize(program, out var ptxSize).Ok();
                ptx = new byte[ptxSize];
                nvrtcGetPTX(program, ptx).Ok();
                nvrtcDestroyProgram(ref program).Ok();
                program = default;

                if (context == default)
                {
                    Console.WriteLine("[DEVICE] Creating CUDA context and command stream...");
                    cuCtxCreate(out context, CUctx_flags.CU_CTX_SCHED_SPIN, device).Ok();
                    cuCtxSetCurrent(context).Ok();
                    cuStreamCreate(out stream, 0).Ok();
                }

                var loadResult = cuModuleLoadData(out module, ptx);
                if (loadResult == CUresult.CUDA_SUCCESS)
                {
                    Console.WriteLine($"[JIT] Successfully compiled and loaded module for compute_{archMajor}{archMinor}!");
                    break;
                }

                Console.WriteLine($"[JIT] Warning: Failed to load module for compute_{archMajor}{archMinor} (result: {loadResult}). Retrying with a lower architecture target...");

                if (archMajor >= 12)
                {
                    archMajor = 9;
                    archMinor = 0;
                }
                else if (archMajor == 9)
                {
                    archMajor = 8;
                    archMinor = 9;
                }
                else if (archMajor == 8 && archMinor == 9)
                {
                    archMajor = 8;
                    archMinor = 6;
                }
                else if (archMajor == 8 && archMinor == 6)
                {
                    archMajor = 7;
                    archMinor = 5;
                }
                else
                {
                    loadResult.Ok();
                }
            }

            var isTrainingTrue = 1;
            var isTrainingFalse = 0;

            cuModuleGetFunction(out var f_clear, module, "clear_gradient").Ok();
            cuModuleGetFunction(out var f_conv1, module, "conv1_forward").Ok();
            cuModuleGetFunction(out var f_conv2, module, "conv2_forward").Ok();
            cuModuleGetFunction(out var f_fc2, module, "fc2_forward").Ok();

            cuModuleGetFunction(out var f_fc2_bwd, module, "fc2_backward").Ok();
            cuModuleGetFunction(out var f_conv2_bwd, module, "conv2_backward").Ok();
            cuModuleGetFunction(out var f_conv1_bwd, module, "conv1_backward").Ok();

            cuModuleGetFunction(out var f_adam, module, "adam_update").Ok();
            CUfunction f_quantize_all = default;
            if (activeConfig.Name == "FP4")
            {
                cuModuleGetFunction(out f_quantize_all, module, "quantize_all_weights").Ok();
            }

            CUfunction f_fc2_bwd_weights = default;
            if (activeConfig.IsV7Based)
            {
                cuModuleGetFunction(out f_fc2_bwd_weights, module, "fc2_backward_weights").Ok();
            }

            CUfunction f_fc1 = default, f_fc1_bwd = default, f_fc1_bwd_weights = default;
            if (activeConfig.HasFC1)
            {
                cuModuleGetFunction(out f_fc1, module, activeConfig.FC1ForwardKernelName).Ok();
                cuModuleGetFunction(out f_fc1_bwd, module, activeConfig.FC1BackwardKernelName).Ok();
                cuModuleGetFunction(out f_fc1_bwd_weights, module, activeConfig.FC1BackwardWeightsKernelName).Ok();
            }

            var conv1BlockX = (uint)activeConfig.Pool1OutSize;
            var conv1BlockY = (uint)activeConfig.Pool1OutSize;
            var conv1BwdBlockX = activeConfig.IsV7Based ? 16u : 24u;
            var conv1BwdBlockY = activeConfig.IsV7Based ? 16u : 24u;

            var conv1FilterCount = activeConfig.Conv1FilterCount;
            var conv2FilterCount = activeConfig.Conv2FilterCount;
            var totalParamElements = activeConfig.TotalParamElements;
            var elementSize = activeConfig.IsHalf ? sizeof(ushort) : sizeof(float);
            var paramBytes = (nuint)(totalParamElements * elementSize);

            var trainImagesSize = (nuint)(h_trainImages.Length * sizeof(uint));
            var trainLabelsSize = (nuint)(h_trainLabels.Length * sizeof(int));
            var testImagesSize = (nuint)(h_testImages.Length * sizeof(uint));
            var testLabelsSize = (nuint)(h_testLabels.Length * sizeof(int));

            var conv1OutSize = (nuint)(BatchSize * activeConfig.Conv1OutPerSample);
            var conv1UnpooledSize = (nuint)(BatchSize * activeConfig.Conv1UnpooledPerSample);
            var conv2OutSize = (nuint)(BatchSize * activeConfig.Conv2OutPerSample);
            var conv2UnpooledSize = (nuint)(BatchSize * activeConfig.Conv2UnpooledPerSample);

            nuint fc1OutSize = 0;
            nuint fc1UnpooledSize = 0;
            if (activeConfig.HasFC1)
            {
                var fc1OutCount = activeConfig.FC1OutputElementsOverride ?? activeConfig.FC1Outputs;
                fc1OutSize = (nuint)(BatchSize * fc1OutCount);
                if (activeConfig.IsHalf)
                {
                    fc1UnpooledSize = (nuint)(BatchSize * fc1OutCount);
                }
            }
            var fc2OutSize = (nuint)(BatchSize * 10);

            var conv1OutGradSize = (nuint)(BatchSize * activeConfig.Conv1OutPerSample);

            nuint fc2InGradSize = 0;
            nuint conv2OutGradSize = 0;
            nuint fc1OutGradSize = 0;
            nuint arenaIntermediateGradSize = 0;

            if (activeConfig.UsesPooledConv2AsFc1Input)
            {
                fc2InGradSize = (nuint)(BatchSize * activeConfig.FC2Inputs);
                conv2OutGradSize = (nuint)(BatchSize * activeConfig.Conv2OutPerSample);
                fc1OutGradSize = (nuint)(BatchSize * (activeConfig.FC1OutputElementsOverride ?? activeConfig.FC1Outputs));
                if (activeConfig.RequiresIntermediateGradBuffer)
                {
                    arenaIntermediateGradSize = (nuint)(BatchSize * activeConfig.Conv2OutPerSample);
                }
            }
            else if (activeConfig.HasFC1)
            {
                fc1OutGradSize = (nuint)(BatchSize * activeConfig.FC1Outputs);
                conv2OutGradSize = (nuint)(BatchSize * activeConfig.FC1Inputs);
            }
            else
            {
                fc2InGradSize = (nuint)(BatchSize * activeConfig.FC2Inputs);
            }

            var stepSize = (nuint)sizeof(int);

            // Compute total required bytes including 256-byte alignments
            nuint totalRequiredBytes = 0;
            nuint alignment = 256;
            void AddSize(nuint sz)
            {
                if (sz > 0)
                {
                    nuint alignedSize = (sz + alignment - 1) & ~(alignment - 1);
                    totalRequiredBytes += alignedSize;
                }
            }

            AddSize(trainImagesSize);
            AddSize(trainLabelsSize);
            AddSize(testImagesSize);
            AddSize(testLabelsSize);
            AddSize(paramBytes);
            AddSize(paramBytes);
            AddSize(paramBytes);
            AddSize(paramBytes);
            if (activeConfig.Name == "FP4")
            {
                AddSize(paramBytes);
            }
            AddSize(conv1OutSize * (nuint)elementSize);
            AddSize(conv1UnpooledSize * (nuint)elementSize);
            AddSize(conv2OutSize * (nuint)elementSize);
            AddSize(conv2UnpooledSize * (nuint)elementSize);
            AddSize(fc1OutSize * (nuint)elementSize);
            AddSize(fc1UnpooledSize * (nuint)elementSize);
            AddSize(fc2OutSize * (nuint)elementSize);
            AddSize(conv1OutGradSize * (nuint)elementSize);
            AddSize(fc2InGradSize * (nuint)elementSize);
            AddSize(conv2OutGradSize * (nuint)elementSize);
            AddSize(fc1OutGradSize * (nuint)elementSize);
            AddSize(arenaIntermediateGradSize * (nuint)elementSize);
            AddSize(stepSize);

            // Allocate unified memory arena
            var arena = new GpuMemoryArena();
            arena.Allocate(totalRequiredBytes);

            // Rent segments
            var d_trainImages = arena.Rent(trainImagesSize);
            var d_trainLabels = arena.Rent(trainLabelsSize);
            var d_testImages = arena.Rent(testImagesSize);
            var d_testLabels = arena.Rent(testLabelsSize);

            fixed (uint* pTrainImages = h_trainImages)
            fixed (int* pTrainLabels = h_trainLabels)
            fixed (uint* pTestImages = h_testImages)
            fixed (int* pTestLabels = h_testLabels)
            {
                cuMemcpyHtoD(d_trainImages, (IntPtr)pTrainImages, trainImagesSize).Ok();
                cuMemcpyHtoD(d_trainLabels, (IntPtr)pTrainLabels, trainLabelsSize).Ok();
                cuMemcpyHtoD(d_testImages, (IntPtr)pTestImages, testImagesSize).Ok();
                cuMemcpyHtoD(d_testLabels, (IntPtr)pTestLabels, testLabelsSize).Ok();
            }

            var d_allParams = arena.Rent(paramBytes);
            var d_allParamGrads = arena.Rent(paramBytes);
            var d_allParamM = arena.Rent(paramBytes);
            var d_allParamV = arena.Rent(paramBytes);
            CUdeviceptr d_quantParams = default;
            if (activeConfig.Name == "FP4")
            {
                d_quantParams = arena.Rent(paramBytes);
            }

            cuMemsetD8(d_allParamGrads, 0, paramBytes).Ok();
            cuMemsetD8(d_allParamM, 0, paramBytes).Ok();
            cuMemsetD8(d_allParamV, 0, paramBytes).Ok();

            var conv1Param = activeConfig.GetParam("conv1");
            var conv2Param = activeConfig.GetParam("conv2");
            var fc2Param = activeConfig.GetParam("fc2");

            var sliceParamsSrc = activeConfig.Name == "FP4" ? d_quantParams : d_allParams;

            CUdeviceptr d_conv1Filters = SliceDevicePtr(sliceParamsSrc, conv1Param.WeightOffset, elementSize);
            CUdeviceptr d_conv1Biases = SliceDevicePtr(sliceParamsSrc, conv1Param.BiasOffset, elementSize);
            CUdeviceptr d_conv2Filters = SliceDevicePtr(sliceParamsSrc, conv2Param.WeightOffset, elementSize);
            CUdeviceptr d_conv2Biases = SliceDevicePtr(sliceParamsSrc, conv2Param.BiasOffset, elementSize);
            CUdeviceptr d_fc2Weights = SliceDevicePtr(sliceParamsSrc, fc2Param.WeightOffset, elementSize);
            CUdeviceptr d_fc2Biases = SliceDevicePtr(sliceParamsSrc, fc2Param.BiasOffset, elementSize);

            CUdeviceptr d_conv1FiltersGrad = SliceDevicePtr(d_allParamGrads, conv1Param.WeightOffset, elementSize);
            CUdeviceptr d_conv1BiasesGrad = SliceDevicePtr(d_allParamGrads, conv1Param.BiasOffset, elementSize);
            CUdeviceptr d_conv2FiltersGrad = SliceDevicePtr(d_allParamGrads, conv2Param.WeightOffset, elementSize);
            CUdeviceptr d_conv2BiasesGrad = SliceDevicePtr(d_allParamGrads, conv2Param.BiasOffset, elementSize);
            CUdeviceptr d_fc2WeightsGrad = SliceDevicePtr(d_allParamGrads, fc2Param.WeightOffset, elementSize);
            CUdeviceptr d_fc2BiasesGrad = SliceDevicePtr(d_allParamGrads, fc2Param.BiasOffset, elementSize);

            CUdeviceptr d_fc1Weights = default, d_fc1Biases = default;
            CUdeviceptr d_fc1WeightsGrad = default, d_fc1BiasesGrad = default;

            if (activeConfig.HasFC1)
            {
                var fc1Param = activeConfig.GetParam("fc1");
                d_fc1Weights = SliceDevicePtr(sliceParamsSrc, fc1Param.WeightOffset, elementSize);
                d_fc1Biases = SliceDevicePtr(sliceParamsSrc, fc1Param.BiasOffset, elementSize);
                d_fc1WeightsGrad = SliceDevicePtr(d_allParamGrads, fc1Param.WeightOffset, elementSize);
                d_fc1BiasesGrad = SliceDevicePtr(d_allParamGrads, fc1Param.BiasOffset, elementSize);
            }

            var d_conv1Out = arena.Rent(conv1OutSize * (nuint)elementSize);
            var d_conv1Unpooled = arena.Rent(conv1UnpooledSize * (nuint)elementSize);
            var d_conv2Out = arena.Rent(conv2OutSize * (nuint)elementSize);
            var d_conv2Unpooled = arena.Rent(conv2UnpooledSize * (nuint)elementSize);

            CUdeviceptr d_fc1Out = default;
            CUdeviceptr d_fc1Unpooled = default;
            if (activeConfig.HasFC1)
            {
                d_fc1Out = arena.Rent(fc1OutSize * (nuint)elementSize);
                if (fc1UnpooledSize > 0)
                {
                    d_fc1Unpooled = arena.Rent(fc1UnpooledSize * (nuint)elementSize);
                }
            }
            var d_fc2Out = arena.Rent(fc2OutSize * (nuint)elementSize);

            var d_conv1OutGrad = arena.Rent(conv1OutGradSize * (nuint)elementSize);

            CUdeviceptr d_fc1OutGrad = default, d_conv2OutGrad = default, d_intermediateGrad = default;
            CUdeviceptr d_fc2InGrad = default;
            if (activeConfig.UsesPooledConv2AsFc1Input)
            {
                d_fc2InGrad = arena.Rent(fc2InGradSize * (nuint)elementSize);
                d_conv2OutGrad = arena.Rent(conv2OutGradSize * (nuint)elementSize);
                d_fc1OutGrad = arena.Rent(fc1OutGradSize * (nuint)elementSize);
                if (activeConfig.RequiresIntermediateGradBuffer)
                {
                    d_intermediateGrad = arena.Rent(arenaIntermediateGradSize * (nuint)elementSize);
                }
            }
            else if (activeConfig.HasFC1)
            {
                d_fc1OutGrad = arena.Rent(fc1OutGradSize * (nuint)elementSize);
                d_conv2OutGrad = arena.Rent(conv2OutGradSize * (nuint)elementSize);
            }
            else
            {
                d_fc2InGrad = arena.Rent(fc2InGradSize * (nuint)elementSize);
            }

            var d_step = arena.Rent(stepSize);

            var fc2BlockSize = activeConfig.Name == "V5" ? 128u : 256u;
            var fc1Chunks = 8;

            if (activeConfig.Name == "FP4")
            {
                CUdeviceptr d_conv1Filters_init = SliceDevicePtr(d_allParams, conv1Param.WeightOffset, elementSize);
                CUdeviceptr d_conv1Biases_init = SliceDevicePtr(d_allParams, conv1Param.BiasOffset, elementSize);
                CUdeviceptr d_conv2Filters_init = SliceDevicePtr(d_allParams, conv2Param.WeightOffset, elementSize);
                CUdeviceptr d_conv2Biases_init = SliceDevicePtr(d_allParams, conv2Param.BiasOffset, elementSize);
                CUdeviceptr d_fc2Weights_init = SliceDevicePtr(d_allParams, fc2Param.WeightOffset, elementSize);
                CUdeviceptr d_fc2Biases_init = SliceDevicePtr(d_allParams, fc2Param.BiasOffset, elementSize);
                CUdeviceptr d_fc1Weights_init = default, d_fc1Biases_init = default;
                if (activeConfig.HasFC1)
                {
                    var fc1Param = activeConfig.GetParam("fc1");
                    d_fc1Weights_init = SliceDevicePtr(d_allParams, fc1Param.WeightOffset, elementSize);
                    d_fc1Biases_init = SliceDevicePtr(d_allParams, fc1Param.BiasOffset, elementSize);
                }
                InitializeModelParameters(activeConfig, d_conv1Filters_init, d_conv1Biases_init, d_conv2Filters_init, d_conv2Biases_init, d_fc1Weights_init, d_fc1Biases_init, d_fc2Weights_init, d_fc2Biases_init, 42);

                // Run initial quantization
                var quantizeParamsInit = new void*[] { &d_allParams, &d_quantParams };
                fixed (void** pQuantizeParams = quantizeParamsInit)
                {
                    cuLaunchKernel(f_quantize_all, (uint)((totalParamElements + 255) / 256), 1u, 1u, 256u, 1u, 1u, 0u, stream, pQuantizeParams, null).Ok();
                }
                cuStreamSynchronize(stream).Ok();
            }
            else
            {
                InitializeModelParameters(activeConfig, d_conv1Filters, d_conv1Biases, d_conv2Filters, d_conv2Biases, d_fc1Weights, d_fc1Biases, d_fc2Weights, d_fc2Biases, 42);
            }

            Console.WriteLine("[GRAPH] Capturing training loop into a single optimized CUDA Graph...");

            cuGraphCreate(out var epochGraph, 0).Ok();

            var trainStepCount = activeConfig.TotalSteps;
            var testStepCount = TestImagesCount / BatchSize;

            var localClearGradElements = conv1OutGradSize;
            var localTotalParamsCount = totalParamElements;

            CUgraphNode lastNode = default;
            var currentDependencies = new CUgraphNode[1];

            CUdeviceptr d_fc2In = activeConfig.HasFC1 ? d_fc1Out : d_conv2Out;
            CUdeviceptr d_fc2InGrad_kernel = activeConfig.HasFC1 ? d_fc1OutGrad : d_fc2InGrad;
            CUdeviceptr d_conv2BwdInGrad = activeConfig.HasFC1 ? d_conv2OutGrad : d_fc2InGrad;

            var clearGradParams = new void*[] { &d_conv1OutGrad, &localClearGradElements };
            var conv1Params = new void*[]
            {
                &d_trainImages, &d_conv1Filters, &d_conv1Biases,
                &d_conv1Out, &d_conv1Unpooled, &d_step, &isTrainingTrue
            };
            var conv2Params = activeConfig.UsesPooledConv2AsFc1Input
                ? new void*[]
                {
                    &d_conv1Out, &d_conv2Filters, &d_conv2Biases,
                    &d_conv2Out, &d_conv2Unpooled, &d_fc1Weights, &d_fc1Biases
                }
                : new void*[]
                {
                    &d_conv1Out, &d_conv2Filters, &d_conv2Biases,
                    &d_conv2Out, &d_conv2Unpooled
                };
            var fc1Params = activeConfig.UsesPooledConv2AsFc1Input
                ? new void*[] { &d_conv2Out, &d_fc1Out }
                : (activeConfig.Name == "V5"
                    ? new void*[] { &d_trainImages, &d_fc1Weights, &d_fc1Biases, &d_fc1Out, &d_step, &isTrainingTrue }
                    : (activeConfig.IsHalf
                        ? new void*[] { &d_conv2Out, &d_fc1Weights, &d_fc1Biases, &d_fc1Out, &d_fc1Unpooled }
                        : new void*[] { &d_conv2Out, &d_fc1Weights, &d_fc1Biases, &d_fc1Out }));
            var fc2Params = new void*[]
            {
                &d_fc2In, &d_fc2Weights, &d_fc2Biases, &d_fc2Out
            };
            var fc2BwdParams = activeConfig.IsHalf
                ? new void*[]
                {
                    &d_fc2Out, &d_trainLabels, &d_fc2In, &d_fc2Weights,
                    &d_fc2WeightsGrad, &d_fc2BiasesGrad, &d_fc2InGrad_kernel, &d_step,
                    &d_fc1Unpooled
                }
                : new void*[]
                {
                    &d_fc2Out, &d_trainLabels, &d_fc2In, &d_fc2Weights,
                    &d_fc2WeightsGrad, &d_fc2BiasesGrad, &d_fc2InGrad_kernel, &d_step
                };
            var fc2BwdWeightsParams = new void*[]
            {
                &d_fc2Out, &d_trainLabels, &d_fc2In, &d_fc2WeightsGrad, &d_step
            };
            var fc1BwdParams = activeConfig.UsesPooledConv2AsFc1Input
                ? new void*[] { &d_fc1OutGrad, &d_fc1Out, &d_conv2Out, &d_conv2OutGrad }
                : new void*[]
                {
                    &d_fc1OutGrad, &d_fc1Out, &d_conv2Out, &d_fc1Weights,
                    &d_fc1BiasesGrad, &d_conv2OutGrad
                };
            var fc1BwdWeightsParams = activeConfig.UsesPooledConv2AsFc1Input
                ? new void*[]
                {
                    &d_intermediateGrad, &d_conv2Unpooled, &d_conv1Out, &d_conv2Filters,
                    &d_conv2FiltersGrad, &d_conv2BiasesGrad, &d_conv1OutGrad, &d_conv2OutGrad
                }
                : (activeConfig.Name == "V5"
                    ? new void*[] { &d_fc1OutGrad, &d_fc1Out, &d_trainImages, &d_fc1WeightsGrad, &d_step }
                    : new void*[] { &d_fc1OutGrad, &d_fc1Out, &d_conv2Out, &d_fc1WeightsGrad });
            var conv2BwdParams = activeConfig.UsesPooledConv2AsFc1Input
                ? new void*[]
                {
                    &d_conv2OutGrad, &d_conv2Out, &d_conv2Unpooled, &d_conv1Out,
                    &d_conv2Filters, &d_conv2FiltersGrad, &d_conv2BiasesGrad, &d_conv1OutGrad,
                    &d_fc1Weights, &d_fc1WeightsGrad, &d_fc1BiasesGrad, &d_intermediateGrad
                }
                : new void*[]
                {
                    &d_conv2BwdInGrad, &d_conv2Out, &d_conv2Unpooled, &d_conv1Out,
                    &d_conv2Filters, &d_conv2FiltersGrad, &d_conv2BiasesGrad,
                    &d_conv1OutGrad
                };
            var conv1BwdParams = new void*[]
            {
                &d_conv1OutGrad, &d_conv1Out, &d_conv1Unpooled, &d_trainImages,
                &d_conv1FiltersGrad, &d_conv1BiasesGrad, &d_step, &isTrainingTrue
            };
            var adamParams = new void*[]
            {
                &d_allParams, &d_allParamGrads, &d_allParamM, &d_allParamV,
                &localTotalParamsCount, &d_step
            };

            for (var step = 0; step < trainStepCount; step++)
            {
                var depsClear = step == 0
                    ? Array.Empty<CUgraphNode>() : [lastNode];
                lastNode = AddKernelNode(epochGraph, depsClear, f_clear,
                    (uint)((conv1OutGradSize + 255) / 256), 1u, 1u,
                    256u, 1u, 1u, clearGradParams);

                if (activeConfig.RequiresIntermediateGradBuffer)
                {
                    currentDependencies[0] = lastNode;
                    var intermediateGradSize = BatchSize * activeConfig.Conv2OutPerSample;
                    var clearIntermediateParams = new void*[] { &d_intermediateGrad, &intermediateGradSize };
                    lastNode = AddKernelNode(epochGraph, currentDependencies, f_clear,
                        (uint)((intermediateGradSize + 255) / 256), 1u, 1u,
                        256u, 1u, 1u, clearIntermediateParams);
                }

                if (activeConfig.Name != "V5")
                {
                    currentDependencies[0] = lastNode;
                    lastNode = AddKernelNode(epochGraph, currentDependencies,
                        f_conv1, (uint)BatchSize, (uint)conv1FilterCount, 1u,
                        conv1BlockX, conv1BlockY, 1u, conv1Params);

                    currentDependencies[0] = lastNode;
                    lastNode = AddKernelNode(epochGraph, currentDependencies,
                        f_conv2, (uint)BatchSize, 1u, 1u,
                        256u, 1u, 1u, conv2Params);
                }

                if (activeConfig.HasFC1)
                {
                    currentDependencies[0] = lastNode;
                    var fc1BlockSize = activeConfig.UsesPooledConv2AsFc1Input ? (uint)activeConfig.FC2Inputs : (activeConfig.IsHalf ? (uint)activeConfig.FC1Inputs : 128u);
                    lastNode = AddKernelNode(epochGraph, currentDependencies,
                        f_fc1, (uint)BatchSize, 1u, 1u,
                        fc1BlockSize, 1u, 1u, fc1Params);
                }



                currentDependencies[0] = lastNode;
                lastNode = AddKernelNode(epochGraph, currentDependencies,
                    f_fc2, (uint)BatchSize, 1u, 1u,
                    fc2BlockSize, 1u, 1u, fc2Params);

                currentDependencies[0] = lastNode;
                lastNode = AddKernelNode(epochGraph, currentDependencies,
                    f_fc2_bwd, (uint)BatchSize, 1u, 1u,
                    fc2BlockSize, 1u, 1u, fc2BwdParams);

                if (activeConfig.IsV7Based)
                {
                    currentDependencies[0] = lastNode;
                    var fc2BwdWGridX = (uint)activeConfig.FC2Inputs;
                    lastNode = AddKernelNode(epochGraph, currentDependencies,
                        f_fc2_bwd_weights, fc2BwdWGridX, 1u, 1u,
                        128u, 1u, 1u, fc2BwdWeightsParams);
                }

                if (activeConfig.HasFC1)
                {
                    currentDependencies[0] = lastNode;
                    if (activeConfig.UsesPooledConv2AsFc1Input)
                    {
                        lastNode = AddKernelNode(epochGraph, currentDependencies,
                            f_fc1_bwd, (uint)BatchSize, 1u, 1u,
                            (uint)activeConfig.FC2Inputs, 1u, 1u, fc1BwdParams);
                    }
                    else if (activeConfig.Name == "V5")
                    {
                        lastNode = AddKernelNode(epochGraph, currentDependencies,
                            f_fc1_bwd, 1u, 1u, 1u,
                            128u, 1u, 1u, fc1BwdParams);
                    }
                    else
                    {
                        lastNode = AddKernelNode(epochGraph, currentDependencies,
                            f_fc1_bwd, (uint)BatchSize, 1u, 1u,
                            256u, 1u, 1u, fc1BwdParams);
                    }

                    if (!activeConfig.UsesPooledConv2AsFc1Input)
                    {
                        currentDependencies[0] = lastNode;
                        if (activeConfig.IsHalf)
                        {
                            lastNode = AddKernelNode(epochGraph, currentDependencies,
                                f_fc1_bwd_weights, (uint)activeConfig.FC1Inputs, 1u, 1u,
                                64u, 1u, 1u, fc1BwdWeightsParams);
                        }
                        else if (activeConfig.Name == "V5")
                        {
                            lastNode = AddKernelNode(epochGraph, currentDependencies,
                                f_fc1_bwd_weights, 784u, 1u, 1u,
                                128u, 1u, 1u, fc1BwdWeightsParams);
                        }
                        else if (activeConfig.Name == "V1" || activeConfig.Name == "V4")
                        {
                            lastNode = AddKernelNode(epochGraph, currentDependencies,
                                f_fc1_bwd_weights, 400u, 1u, 1u,
                                32u, 1u, 1u, fc1BwdWeightsParams);
                        }
                        else
                        {
                            lastNode = AddKernelNode(epochGraph, currentDependencies,
                                f_fc1_bwd_weights, 8u, 8u, (uint)fc1Chunks,
                                128u, 1u, 1u, fc1BwdWeightsParams);
                        }
                    }
                }

                if (activeConfig.Name != "V5")
                {
                    currentDependencies[0] = lastNode;
                    lastNode = AddKernelNode(epochGraph, currentDependencies,
                        f_conv2_bwd, (uint)conv2FilterCount * (uint)conv2Chunks, 1u, 1u,
                        128u, 1u, 1u, conv2BwdParams);

                    if (activeConfig.UsesPooledConv2AsFc1Input)
                    {
                        currentDependencies[0] = lastNode;
                        lastNode = AddKernelNode(epochGraph, currentDependencies,
                            f_fc1_bwd_weights, (uint)conv2FilterCount * (uint)conv2Chunks, 1u, 1u,
                            128u, 1u, 1u, fc1BwdWeightsParams);
                    }

                    currentDependencies[0] = lastNode;
                    lastNode = AddKernelNode(epochGraph, currentDependencies,
                        f_conv1_bwd, (uint)conv1FilterCount * (uint)conv1Chunks, 1u, 1u,
                        conv1BwdBlockX, conv1BwdBlockY, 1u, conv1BwdParams);
                }

                currentDependencies[0] = lastNode;
                lastNode = AddKernelNode(epochGraph, currentDependencies,
                    f_adam, (uint)((totalParamElements + 255) / 256), 1u, 1u,
                    256u, 1u, 1u, adamParams);

                if (activeConfig.Name == "FP4")
                {
                    currentDependencies[0] = lastNode;
                    var quantizeAllParams = new void*[] { &d_allParams, &d_quantParams };
                    lastNode = AddKernelNode(epochGraph, currentDependencies,
                        f_quantize_all, (uint)((totalParamElements + 255) / 256), 1u, 1u,
                        256u, 1u, 1u, quantizeAllParams);
                }
            }

            Console.WriteLine("[GRAPH] Instantiating executable graph...");
            Span<byte> graphLogBuffer = stackalloc byte[2048];
            var instantiateResult = cuGraphInstantiate(out var epochGraphExec,
                epochGraph, out var errorNode, graphLogBuffer,
                (nuint)graphLogBuffer.Length);
            if (instantiateResult.IsError())
            {
                var log = Encoding.UTF8.GetString(graphLogBuffer).TrimEnd('\0');
                throw new InvalidOperationException(
                    $"Graph instantiation failed with {instantiateResult.ToStringFast()} at node {errorNode.Value}:\n{log}");
            }

            var trainingTime = 0.0;
            int[] seedsToTry = [42, 1337, 7, 100, 2026, 12345, 999, 8888,
                12, 1111, 19, 37, 73, 97, 101, 223, 317, 503, 709, 883];

            var h_fcOut = new float[BatchSize * ClassCount];
            var h_fcOutHalf = new Half[BatchSize * ClassCount];

            var argsTestConv1 = stackalloc void*[]
            {
                &d_testImages, &d_conv1Filters, &d_conv1Biases,
                &d_conv1Out, &d_conv1Unpooled, &d_step, &isTrainingFalse
            };
            var argsConv2 = stackalloc void*[]
            {
                &d_conv1Out, &d_conv2Filters, &d_conv2Biases,
                &d_conv2Out, &d_conv2Unpooled
            };
            var argsFc2V2 = stackalloc void*[]
            {
                &d_conv2Out, &d_fc2Weights, &d_fc2Biases, &d_fc2Out
            };
            var argsFc1V1 = stackalloc void*[]
            {
                &d_conv2Out, &d_fc1Weights, &d_fc1Biases, &d_fc1Out
            };
            var argsFc1V5 = stackalloc void*[]
            {
                &d_testImages, &d_fc1Weights, &d_fc1Biases, &d_fc1Out, &d_step, &isTrainingFalse
            };
            var argsFc2V1 = stackalloc void*[]
            {
                &d_fc1Out, &d_fc2Weights, &d_fc2Biases, &d_fc2Out
            };

            var measuredTimes = new System.Collections.Generic.List<double>();
            var measuredAccuracies = new System.Collections.Generic.List<double>();
            var profileTimes = new System.Collections.Generic.Dictionary<string, System.Collections.Generic.List<float>>();
            double warmupAcc = 0.0;
            double warmupTime = 0.0;

            for (var sIndex = 0; sIndex < 4; sIndex++)
            {
                var currentSeed = seedsToTry[sIndex];
                var isWarmup = sIndex == 0;

                if (isWarmup)
                {
                    Console.WriteLine($"[TRAIN] Launching Warmup Run (Seed: {currentSeed})...");
                }
                else
                {
                    Console.WriteLine($"[TRAIN] Launching Measured Run #{sIndex} (Seed: {currentSeed})...");
                }

                cuMemsetD8(d_allParamGrads, 0, paramBytes).Ok();
                cuMemsetD8(d_allParamM, 0, paramBytes).Ok();
                cuMemsetD8(d_allParamV, 0, paramBytes).Ok();

                InitializeModelParameters(activeConfig, d_conv1Filters, d_conv1Biases, d_conv2Filters, d_conv2Biases, d_fc1Weights, d_fc1Biases, d_fc2Weights, d_fc2Biases, currentSeed);

                var zero = 0;
                cuMemcpyHtoD(d_step, (IntPtr)(&zero), (nuint)sizeof(int)).Ok();





                var stopwatch = Stopwatch.StartNew();
                if (profile)
                {
                    CUevent startEvent, stopEvent;
                    cuEventCreate(out startEvent, 0).Ok();
                    cuEventCreate(out stopEvent, 0).Ok();

                    var clearTimes = new System.Collections.Generic.List<float>();
                    var conv1Times = new System.Collections.Generic.List<float>();
                    var conv2Times = new System.Collections.Generic.List<float>();
                    var fc1Times = new System.Collections.Generic.List<float>();
                    var fc2Times = new System.Collections.Generic.List<float>();
                    var fc2BwdTimes = new System.Collections.Generic.List<float>();
                    var fc2BwdWTimes = new System.Collections.Generic.List<float>();
                    var fc1BwdTimes = new System.Collections.Generic.List<float>();
                    var fc1BwdWTimes = new System.Collections.Generic.List<float>();
                    var conv2BwdTimes = new System.Collections.Generic.List<float>();
                    var conv1BwdTimes = new System.Collections.Generic.List<float>();
                    var adamTimes = new System.Collections.Generic.List<float>();

                    float MeasureKernel(CUfunction function, uint gridX, uint gridY, uint gridZ, uint blockX, uint blockY, uint blockZ, void*[] args)
                    {
                        fixed (void** pArgs = args)
                        {
                            cuEventRecord(startEvent, stream).Ok();
                            cuLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ, 0u, stream, pArgs, null).Ok();
                            cuEventRecord(stopEvent, stream).Ok();
                            cuStreamSynchronize(stream).Ok();
                            float ms;
                            cuEventElapsedTime(out ms, startEvent, stopEvent).Ok();
                            return ms;
                        }
                    }

                    for (var step = 0; step < trainStepCount; step++)
                    {
                        clearTimes.Add(MeasureKernel(f_clear, (uint)((conv1OutGradSize + 255) / 256), 1u, 1u, 256u, 1u, 1u, clearGradParams));
                        if (activeConfig.Name == "V8")
                        {
                            var intermediateGradSize = BatchSize * 3136;
                            var clearIntermediateParams = new void*[] { &d_intermediateGrad, &intermediateGradSize };
                            clearTimes.Add(MeasureKernel(f_clear, (uint)((intermediateGradSize + 255) / 256), 1u, 1u, 256u, 1u, 1u, clearIntermediateParams));
                        }

                        if (activeConfig.Name != "V5")
                        {
                            conv1Times.Add(MeasureKernel(f_conv1, (uint)BatchSize, (uint)conv1FilterCount, 1u, conv1BlockX, conv1BlockY, 1u, conv1Params));
                            conv2Times.Add(MeasureKernel(f_conv2, (uint)BatchSize, 1u, 1u, 256u, 1u, 1u, conv2Params));
                        }

                        if (activeConfig.HasFC1)
                        {
                            var fc1FwdBlockSize = activeConfig.Name == "V8" ? 784u : (activeConfig.IsHalf ? (uint)activeConfig.FC1Inputs : 128u);
                            fc1Times.Add(MeasureKernel(f_fc1, (uint)BatchSize, 1u, 1u, fc1FwdBlockSize, 1u, 1u, fc1Params));
                        }

                        fc2Times.Add(MeasureKernel(f_fc2, (uint)BatchSize, 1u, 1u, fc2BlockSize, 1u, 1u, fc2Params));

                        fc2BwdTimes.Add(MeasureKernel(f_fc2_bwd, (uint)BatchSize, 1u, 1u, fc2BlockSize, 1u, 1u, fc2BwdParams));

                        if (activeConfig.IsV7Based)
                        {
                            var fc2BwdWGridX = (uint)activeConfig.FC2Inputs;
                            fc2BwdWTimes.Add(MeasureKernel(f_fc2_bwd_weights, fc2BwdWGridX, 1u, 1u, 128u, 1u, 1u, fc2BwdWeightsParams));
                        }

                        if (activeConfig.HasFC1)
                        {
                            if (activeConfig.Name == "V8")
                            {
                                fc1BwdTimes.Add(MeasureKernel(f_fc1_bwd, (uint)BatchSize, 1u, 1u, 784u, 1u, 1u, fc1BwdParams));
                                fc1BwdWTimes.Add(MeasureKernel(f_fc1_bwd_weights, (uint)conv2FilterCount * (uint)conv2Chunks, 1u, 1u, 128u, 1u, 1u, fc1BwdWeightsParams));
                            }
                            else if (activeConfig.IsHalf)
                            {
                                fc1BwdTimes.Add(MeasureKernel(f_fc1_bwd, (uint)BatchSize, 1u, 1u, (uint)activeConfig.FC1Inputs, 1u, 1u, fc1BwdParams));
                                fc1BwdWTimes.Add(MeasureKernel(f_fc1_bwd_weights, (uint)activeConfig.FC1Inputs, 1u, 1u, 64u, 1u, 1u, fc1BwdWeightsParams));
                            }
                            else if (activeConfig.Name == "V5")
                            {
                                fc1BwdTimes.Add(MeasureKernel(f_fc1_bwd, 1u, 1u, 1u, 128u, 1u, 1u, fc1BwdParams));
                                fc1BwdWTimes.Add(MeasureKernel(f_fc1_bwd_weights, 784u, 1u, 1u, 128u, 1u, 1u, fc1BwdWeightsParams));
                            }
                            else if (activeConfig.Name == "V1" || activeConfig.Name == "V4")
                            {
                                fc1BwdTimes.Add(MeasureKernel(f_fc1_bwd, (uint)BatchSize, 1u, 1u, 256u, 1u, 1u, fc1BwdParams));
                                fc1BwdWTimes.Add(MeasureKernel(f_fc1_bwd_weights, 400u, 1u, 1u, 32u, 1u, 1u, fc1BwdWeightsParams));
                            }
                            else
                            {
                                fc1BwdTimes.Add(MeasureKernel(f_fc1_bwd, (uint)BatchSize, 1u, 1u, 256u, 1u, 1u, fc1BwdParams));
                                fc1BwdWTimes.Add(MeasureKernel(f_fc1_bwd_weights, 8u, 8u, (uint)fc1Chunks, 128u, 1u, 1u, fc1BwdWeightsParams));
                            }
                        }

                        if (activeConfig.Name != "V5")
                        {
                            conv2BwdTimes.Add(MeasureKernel(f_conv2_bwd, (uint)conv2FilterCount * (uint)conv2Chunks, 1u, 1u, 128u, 1u, 1u, conv2BwdParams));
                            conv1BwdTimes.Add(MeasureKernel(f_conv1_bwd, (uint)conv1FilterCount * (uint)conv1Chunks, 1u, 1u, conv1BwdBlockX, conv1BwdBlockY, 1u, conv1BwdParams));
                        }

                        adamTimes.Add(MeasureKernel(f_adam, (uint)((totalParamElements + 255) / 256), 1u, 1u, 256u, 1u, 1u, adamParams));
                    }

                    stopwatch.Stop();
                    trainingTime = stopwatch.Elapsed.TotalMilliseconds;

                    cuEventDestroy(startEvent).Ok();
                    cuEventDestroy(stopEvent).Ok();

                    void PrintStats(string name, System.Collections.Generic.List<float> times)
                    {
                        if (times.Count == 0) return;
                        float min = float.MaxValue, max = float.MinValue, sum = 0;
                        for (var i = 0; i < times.Count; i++)
                        {
                            var t = times[i];
                            if (t < min) min = t;
                            if (t > max) max = t;
                            sum += t;
                        }
                        var mean = sum / times.Count;
                        Console.WriteLine($"[PROFILE] {name,-20} | Min = {min,8:F3} ms | Mean = {mean,8:F3} ms | Max = {max,8:F3} ms | Total = {sum,8:F2} ms");
                    }

                    if (!isWarmup)
                    {
                        Console.WriteLine("### GPU KERNEL PROFILING REPORT (Measured Run) ###");
                        PrintStats("clear_gradient", clearTimes);
                        PrintStats("conv1_forward", conv1Times);
                        PrintStats("conv2_forward", conv2Times);
                        PrintStats("fc1_forward", fc1Times);
                        PrintStats("fc2_forward", fc2Times);
                        PrintStats("fc2_backward", fc2BwdTimes);
                        PrintStats("fc2_bwd_weights", fc2BwdWTimes);
                        PrintStats("fc1_backward", fc1BwdTimes);
                        PrintStats("fc1_bwd_weights", fc1BwdWTimes);
                        PrintStats("conv2_backward", conv2BwdTimes);
                        PrintStats("conv1_backward", conv1BwdTimes);
                        PrintStats("adam_update", adamTimes);

                        profileTimes["clear_gradient"] = clearTimes;
                        profileTimes["conv1_forward"] = conv1Times;
                        profileTimes["conv2_forward"] = conv2Times;
                        profileTimes["fc1_forward"] = fc1Times;
                        profileTimes["fc2_forward"] = fc2Times;
                        profileTimes["fc2_backward"] = fc2BwdTimes;
                        profileTimes["fc2_bwd_weights"] = fc2BwdWTimes;
                        profileTimes["fc1_backward"] = fc1BwdTimes;
                        profileTimes["fc1_bwd_weights"] = fc1BwdWTimes;
                        profileTimes["conv2_backward"] = conv2BwdTimes;
                        profileTimes["conv1_backward"] = conv1BwdTimes;
                        profileTimes["adam_update"] = adamTimes;
                    }
                }
                else
                {
                    cuGraphLaunch(epochGraphExec, stream).Ok();
                    cuStreamSynchronize(stream).Ok();

                    var h_params = new Half[10];
                    var h_grads = new Half[10];
                    cuMemcpyDtoH((IntPtr)Unsafe.AsPointer(ref h_params[0]), d_allParams, (nuint)(10 * sizeof(Half))).Ok();
                    cuMemcpyDtoH((IntPtr)Unsafe.AsPointer(ref h_grads[0]), d_allParamGrads, (nuint)(10 * sizeof(Half))).Ok();
                    string paramsStr = "", gradsStr = "";
                    for (var di = 0; di < 10; di++)
                    {
                        paramsStr += ((float)h_params[di]).ToString("F5") + (di < 9 ? ", " : "");
                        gradsStr += ((float)h_grads[di]).ToString("F5") + (di < 9 ? ", " : "");
                    }
                    Console.WriteLine($"[DEBUG] Params: {paramsStr}");
                    Console.WriteLine($"[DEBUG] Grads:  {gradsStr}");

                    stopwatch.Stop();
                    trainingTime = stopwatch.Elapsed.TotalMilliseconds;
                }

                var correctPredictions = 0;

                for (var valStep = 0; valStep < testStepCount; valStep++)
                {
                    var batchOffset = valStep * BatchSize;
                    cuMemcpyHtoD(d_step, (IntPtr)(&valStep), (nuint)sizeof(int)).Ok();

                    if (activeConfig.Name == "V5")
                    {
                        cuLaunchKernel(f_fc1, (uint)BatchSize, 1u, 1u,
                            128u, 1u, 1u, 0u, stream, argsFc1V5, null).Ok();
                        cuLaunchKernel(f_fc2, (uint)BatchSize, 1u, 1u,
                            128u, 1u, 1u, 0u, stream, argsFc2V1, null).Ok();
                    }
                    else
                    {
                        cuLaunchKernel(f_conv1, (uint)BatchSize, (uint)conv1FilterCount, 1u,
                            conv1BlockX, conv1BlockY, 1u, 0u, stream, argsTestConv1, null).Ok();

                        if (activeConfig.UseCustomEvaluationPath)
                        {
                            var argsConv2_v8 = new void*[]
                            {
                                &d_conv1Out, &d_conv2Filters, &d_conv2Biases,
                                &d_conv2Out, &d_conv2Unpooled, &d_fc1Weights, &d_fc1Biases
                            };
                            var argsFc1V1_v8 = new void*[] { &d_conv2Out, &d_fc1Out };
                            var argsFc2V1_v8 = new void*[] { &d_fc1Out, &d_fc2Weights, &d_fc2Biases, &d_fc2Out };

                            fixed (void** pArgs2 = argsConv2_v8)
                            fixed (void** pArgs1 = argsFc1V1_v8)
                            fixed (void** pArgsLogits = argsFc2V1_v8)
                            {
                                cuLaunchKernel(f_conv2, (uint)BatchSize, 1u, 1u,
                                    256u, 1u, 1u, 0u, stream, pArgs2, null).Ok();
                                cuLaunchKernel(f_fc1, (uint)BatchSize, 1u, 1u,
                                    (uint)activeConfig.FC2Inputs, 1u, 1u, 0u, stream, pArgs1, null).Ok();
                                cuLaunchKernel(f_fc2, (uint)BatchSize, 1u, 1u,
                                    256u, 1u, 1u, 0u, stream, pArgsLogits, null).Ok();
                            }
                        }
                        else
                        {
                            cuLaunchKernel(f_conv2, (uint)BatchSize, 1u, 1u,
                                256u, 1u, 1u, 0u, stream, argsConv2, null).Ok();

                            if (activeConfig.HasFC1)
                            {
                                cuLaunchKernel(f_fc1, (uint)BatchSize, 1u, 1u,
                                    256u, 1u, 1u, 0u, stream, argsFc1V1, null).Ok();
                                cuLaunchKernel(f_fc2, (uint)BatchSize, 1u, 1u,
                                    256u, 1u, 1u, 0u, stream, argsFc2V1, null).Ok();
                            }
                            else
                            {
                                cuLaunchKernel(f_fc2, (uint)BatchSize, 1u, 1u,
                                    256u, 1u, 1u, 0u, stream, argsFc2V2, null).Ok();
                            }
                        }
                    }

                    if (activeConfig.IsHalf)
                    {
                        cuMemcpyDtoH((IntPtr)Unsafe.AsPointer(ref h_fcOutHalf[0]),
                            d_fc2Out, (nuint)(h_fcOutHalf.Length * sizeof(Half))).Ok();
                        for (var i = 0; i < h_fcOut.Length; i++)
                        {
                            h_fcOut[i] = (float)h_fcOutHalf[i];
                        }
                    }
                    else
                    {
                        cuMemcpyDtoH((IntPtr)Unsafe.AsPointer(ref h_fcOut[0]),
                            d_fc2Out, (nuint)(h_fcOut.Length * sizeof(float))).Ok();
                    }

                    for (var b = 0; b < BatchSize; b++)
                    {
                        var maxVal = -1e9f;
                        var predLabel = -1;
                        for (var c = 0; c < ClassCount; c++)
                        {
                            var val = h_fcOut[b * ClassCount + c];
                            if (val > maxVal)
                            {
                                maxVal = val;
                                predLabel = c;
                            }
                        }
                        if (predLabel == h_testLabels[batchOffset + b])
                            correctPredictions++;
                    }
                }

                var accuracy = (double)correctPredictions / (testStepCount * BatchSize) * 100.0;
                if (isWarmup)
                {
                    Console.WriteLine($"[WARMUP RESULTS] Accuracy: {accuracy:F2}%, GPU Time: {trainingTime:F3} ms");
                    warmupAcc = accuracy;
                    warmupTime = trainingTime;
                }
                else
                {
                    Console.WriteLine($"[MEASURED RESULTS] Run #{sIndex} - Accuracy: {accuracy:F2}%, GPU Time: {trainingTime:F3} ms");
                    measuredTimes.Add(trainingTime);
                    measuredAccuracies.Add(accuracy);
                }
            }

            double minTime = double.MaxValue, maxTime = double.MinValue, sumTime = 0;
            double minAcc = double.MaxValue, maxAcc = double.MinValue, sumAcc = 0;

            for (var i = 0; i < measuredTimes.Count; i++)
            {
                var t = measuredTimes[i];
                var a = measuredAccuracies[i];
                if (t < minTime) minTime = t;
                if (t > maxTime) maxTime = t;
                sumTime += t;

                if (a < minAcc) minAcc = a;
                if (a > maxAcc) maxAcc = a;
                sumAcc += a;
            }

            meanTime = sumTime / measuredTimes.Count;
            meanAcc = sumAcc / measuredAccuracies.Count;

            Console.WriteLine("### SUMMARY METRICS FOR MEASURED RUNS ###");
            Console.WriteLine($"GPU Training Time: Min = {minTime:F3} ms | Mean = {meanTime:F3} ms | Max = {maxTime:F3} ms");
            Console.WriteLine($"Test Accuracy:     Min = {minAcc:F2}% | Mean = {meanAcc:F2}% | Max = {maxAcc:F2}%");

            WriteMarkdownReport(
                activeConfig,
                deviceName,
                $"{major}{minor}",
                warmupAcc,
                warmupTime,
                measuredAccuracies.ToArray(),
                measuredTimes.ToArray(),
                meanAcc,
                meanTime,
                profileTimes);

            arena.Dispose();

            cuGraphExecDestroy(epochGraphExec).Ok();
            cuGraphDestroy(epochGraph).Ok();
            cuModuleUnload(module).Ok();
        }
        finally
        {
            if (program.Value != IntPtr.Zero)
            {
                nvrtcDestroyProgram(ref program).Ok();
            }
            if (context.Value != IntPtr.Zero)
            {
                cuCtxDestroy(context).Ok();
            }
        }

        return (activeConfig, meanAcc, meanTime);
    }

    static CUgraphNode AddKernelNode(
        CUgraph graph,
        ReadOnlySpan<CUgraphNode> dependencies,
        CUfunction function,
        uint gridX, uint gridY, uint gridZ,
        uint blockX, uint blockY, uint blockZ,
        void*[] args)
    {
        fixed (void** pArgs = args)
        {
            var nodeParams = new CUDA_KERNEL_NODE_PARAMS
            {
                func = function,
                gridDimX = gridX,
                gridDimY = gridY,
                gridDimZ = gridZ,
                blockDimX = blockX,
                blockDimY = blockY,
                blockDimZ = blockZ,
                sharedMemBytes = 0,
                kernelParams = (IntPtr)pArgs,
                extra = IntPtr.Zero
            };

            cuGraphAddKernelNode(out var node, graph, dependencies, (nuint)dependencies.Length, nodeParams).Ok();
            return node;
        }
    }

    static CUdeviceptr SliceDevicePtr(CUdeviceptr ptr, int offsetElements, int elementSize)
    {
        return new CUdeviceptr((IntPtr)(ptr.Value.ToInt64() + offsetElements * elementSize));
    }

    static void InitializeParameters(CUdeviceptr d_weights, CUdeviceptr d_biases, int outFeatures, int inFeatures, int seed, bool isHalf)
    {
        var rand = new Random(seed);
        var stdDev = Math.Sqrt(2.0 / inFeatures);

        if (isHalf)
        {
            Half[] h_weights = new Half[outFeatures * inFeatures];
            for (var i = 0; i < h_weights.Length; i++)
            {
                var u1 = 1.0 - rand.NextDouble();
                var u2 = 1.0 - rand.NextDouble();
                var normalRand = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                h_weights[i] = (Half)(normalRand * stdDev);
            }

            Half[] h_biases = new Half[outFeatures];
            for (var i = 0; i < h_biases.Length; i++)
            {
                h_biases[i] = (Half)0.0f;
            }

            cuMemcpyHtoD(d_weights, (IntPtr)Unsafe.AsPointer(ref h_weights[0]), (nuint)(h_weights.Length * sizeof(Half))).Ok();
            cuMemcpyHtoD(d_biases, (IntPtr)Unsafe.AsPointer(ref h_biases[0]), (nuint)(h_biases.Length * sizeof(Half))).Ok();
        }
        else
        {
            var h_weights = new float[outFeatures * inFeatures];
            for (var i = 0; i < h_weights.Length; i++)
            {
                var u1 = 1.0 - rand.NextDouble();
                var u2 = 1.0 - rand.NextDouble();
                var normalRand = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                h_weights[i] = (float)(normalRand * stdDev);
            }

            var h_biases = new float[outFeatures];
            for (var i = 0; i < h_biases.Length; i++)
            {
                h_biases[i] = 0.0f;
            }

            cuMemcpyHtoD(d_weights, (IntPtr)Unsafe.AsPointer(ref h_weights[0]), (nuint)(h_weights.Length * sizeof(float))).Ok();
            cuMemcpyHtoD(d_biases, (IntPtr)Unsafe.AsPointer(ref h_biases[0]), (nuint)(h_biases.Length * sizeof(float))).Ok();
        }
    }

    static void InitializeModelParameters(
        NetworkConfig activeConfig,
        CUdeviceptr d_conv1Filters, CUdeviceptr d_conv1Biases,
        CUdeviceptr d_conv2Filters, CUdeviceptr d_conv2Biases,
        CUdeviceptr d_fc1Weights, CUdeviceptr d_fc1Biases,
        CUdeviceptr d_fc2Weights, CUdeviceptr d_fc2Biases,
        int seed)
    {
        var isHalf = activeConfig.IsHalf;
        var conv1 = activeConfig.GetParam("conv1");
        InitializeParameters(d_conv1Filters, d_conv1Biases, conv1.OutFeatures, conv1.InFeatures, seed, isHalf);

        var conv2 = activeConfig.GetParam("conv2");
        InitializeParameters(d_conv2Filters, d_conv2Biases, conv2.OutFeatures, conv2.InFeatures, seed, isHalf);

        if (activeConfig.HasFC1)
        {
            var fc1 = activeConfig.GetParam("fc1");
            InitializeParameters(d_fc1Weights, d_fc1Biases, fc1.OutFeatures, fc1.InFeatures, seed, isHalf);
            if (activeConfig.Name == "V1" || activeConfig.Name == "V4")
            {
                var h_biases = new float[fc1.OutFeatures];
                for (var i = 0; i < h_biases.Length; i++) h_biases[i] = 0.1f;
                cuMemcpyHtoD(d_fc1Biases, (IntPtr)Unsafe.AsPointer(ref h_biases[0]), (nuint)(h_biases.Length * sizeof(float))).Ok();
            }
            if (activeConfig.Name == "V1" || activeConfig.Name == "V4" || activeConfig.IsHalf)
            {
                var scale = (activeConfig.Name == "V1" || activeConfig.Name == "V4") ? 0.02f : ((activeConfig.Name == "V14") ? 0.15f : 0.05f);
                ScaleDownDeviceBuffer(d_fc1Weights, fc1.OutFeatures * fc1.InFeatures, scale, isHalf);
            }
        }

        var fc2 = activeConfig.GetParam("fc2");
        InitializeParameters(d_fc2Weights, d_fc2Biases, fc2.OutFeatures, fc2.InFeatures, seed, isHalf);
        if (activeConfig.Name == "V1" || activeConfig.Name == "V4")
        {
            ScaleDownDeviceBuffer(d_fc2Weights, fc2.OutFeatures * fc2.InFeatures, 0.05f, isHalf);
        }
    }

    static unsafe void ScaleDownDeviceBuffer(CUdeviceptr ptr, int size, float scale, bool isHalf)
    {
        if (isHalf)
        {
            Half[] host = new Half[size];
            cuMemcpyDtoH((IntPtr)Unsafe.AsPointer(ref host[0]), ptr, (nuint)(size * sizeof(Half))).Ok();
            for (var i = 0; i < size; i++)
            {
                host[i] = (Half)((float)host[i] * scale);
            }
            cuMemcpyHtoD(ptr, (IntPtr)Unsafe.AsPointer(ref host[0]), (nuint)(size * sizeof(Half))).Ok();
        }
        else
        {
            var host = new float[size];
            cuMemcpyDtoH((IntPtr)Unsafe.AsPointer(ref host[0]), ptr, (nuint)(size * sizeof(float))).Ok();
            for (var i = 0; i < size; i++)
            {
                host[i] = host[i] * scale;
            }
            cuMemcpyHtoD(ptr, (IntPtr)Unsafe.AsPointer(ref host[0]), (nuint)(size * sizeof(float))).Ok();
        }
    }

    static void EnsureDatasetFile(string filePath, string url)
    {
        if (File.Exists(filePath)) return;

        Console.WriteLine($"[DOWNLOAD] MNIST dataset file missing. Fetching from: {url}");
        using var client = new HttpClient();
        var response = client.GetAsync(url).GetAwaiter().GetResult();
        response.EnsureSuccessStatusCode();

        using var fs = File.Create(filePath);
        response.Content.CopyToAsync(fs).GetAwaiter().GetResult();
        Console.WriteLine($"[DOWNLOAD] Download complete. Saved to: {filePath}");
    }

    static (uint[] images, int count) ParseImagesGz(string filePath, int maxCount)
    {
        using var fileStream = File.OpenRead(filePath);
        using var gzStream = new GZipStream(fileStream, CompressionMode.Decompress);
        using var ms = new MemoryStream();
        gzStream.CopyTo(ms);
        var bytes = ms.ToArray();

        var magic = BinaryPrimitives.ReadUInt32BigEndian(bytes.AsSpan(0 * sizeof(uint), sizeof(uint)));
        if (magic != 0x00000803)
            throw new InvalidOperationException($"Invalid images magic number: {magic:X}");

        var count = BinaryPrimitives.ReadUInt32BigEndian(bytes.AsSpan(1 * sizeof(uint), sizeof(uint)));
        var rows = BinaryPrimitives.ReadUInt32BigEndian(bytes.AsSpan(2 * sizeof(uint), sizeof(uint)));
        var cols = BinaryPrimitives.ReadUInt32BigEndian(bytes.AsSpan(3 * sizeof(uint), sizeof(uint)));

        if (rows != 28 || cols != 28)
            throw new InvalidOperationException($"Expected 28x28 images, but got {rows}x{cols}");

        var imageCountToLoad = maxCount;
        var packedImages = new uint[imageCountToLoad * 28];

        for (var i = 0; i < imageCountToLoad; i++)
        {
            var sourceImageIdx = i % count;
            var sourcePixelOffset = 16 + sourceImageIdx * 28 * 28;

            for (var r = 0; r < 28; r++)
            {
                uint rowBits = 0;
                for (var c = 0; c < 28; c++)
                {
                    var pixelVal = bytes[sourcePixelOffset++];
                    if (pixelVal > 127)
                    {
                        rowBits |= (1u << c);
                    }
                }
                packedImages[i * 28 + r] = rowBits;
            }
        }

        return (packedImages, imageCountToLoad);
    }

    static int[] ParseLabelsGz(string filePath, int maxCount)
    {
        using var fileStream = File.OpenRead(filePath);
        using var gzStream = new GZipStream(fileStream, CompressionMode.Decompress);
        using var ms = new MemoryStream();
        gzStream.CopyTo(ms);
        var bytes = ms.ToArray();

        var magic = (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
        if (magic != 0x00000801)
            throw new InvalidOperationException($"Invalid labels magic number: {magic:X}");

        var count = (bytes[4] << 24) | (bytes[5] << 16) | (bytes[6] << 8) | bytes[7];

        var labelCountToLoad = maxCount;
        var labels = new int[labelCountToLoad];

        for (var i = 0; i < labelCountToLoad; i++)
        {
            labels[i] = bytes[8 + (i % count)];
        }

        return labels;
    }

    static void WriteMarkdownReport(
        NetworkConfig config,
        string deviceName,
        string cc,
        double warmupAcc,
        double warmupTime,
        double[] measuredAccs,
        double[] measuredTimes,
        double meanAcc,
        double meanTime,
        System.Collections.Generic.Dictionary<string, System.Collections.Generic.List<float>>? profileTimes = null)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"# MNIST GPU Training Report: {config.Name}");
        sb.AppendLine();
        sb.AppendLine("> [!NOTE]");
        sb.AppendLine($"> Programmatic report automatically generated by CudaSharp on {DateTime.Now:yyyy-MM-dd HH:mm:ss}.");
        sb.AppendLine();
        sb.AppendLine("## 1. Hardware Environment");
        sb.AppendLine();
        sb.AppendLine($"* **Active GPU Device**: {deviceName}");
        sb.AppendLine($"* **Compute Capability**: sm_{cc}");
        sb.AppendLine();
        sb.AppendLine("## 2. Network Architecture");
        sb.AppendLine();
        sb.AppendLine("| Component / Layer | Details | Rationale / Settings |");
        sb.AppendLine("| :--- | :--- | :--- |");
        sb.AppendLine($"| **Model Name** | `{config.Name}` | Target identifier |");
        sb.AppendLine($"| **Element Precision** | {(config.IsHalf ? "FP16 (Half Precision)" : "FP32 (Single Precision)")} | Low-precision bandwidth optimization |");
        sb.AppendLine($"| **Activation Type** | `{config.ActivationType}` | GELU preventing dead neurons vs ReLU |");
        sb.AppendLine($"| **Input Map** | 28x28 1-Bit CPU Packed Register | Packed 32 pixels/register for Layer 1 efficiency |");
        if (config.Name == "V5")
        {
            sb.AppendLine($"| **Conv1 (Layer 1)** | Bypassed (Pure MLP Mode) | Direct input-to-dense projection |");
            sb.AppendLine($"| **Conv2 (Layer 2)** | Bypassed (Pure MLP Mode) | Direct input-to-dense projection |");
        }
        else
        {
            sb.AppendLine($"| **Conv1 (Layer 1)** | {config.Conv1FilterSize}x{config.Conv1FilterSize} Conv ({config.Conv1FilterCount} channels), Pool (2x2) | Spatial feature extraction |");
            sb.AppendLine($"| **Conv2 (Layer 2)** | {config.Conv2FilterSize}x{config.Conv2FilterSize} Conv ({config.Conv2FilterCount} channels), Pool (2x2) | Deep spatial channel representation |");
        }
        if (config.HasFC1)
        {
            sb.AppendLine($"| **FC1 (Layer 3)** | {config.FC1Inputs} -> {config.FC1Outputs} dense hidden projection | Hidden feature scaling |");
        }
        sb.AppendLine($"| **FC2 (Output Layer)**| {config.FC2Inputs} -> 10 classification logits | Final softmax classification |");
        sb.AppendLine();
        sb.AppendLine("## 3. Weight Initialization Strategy");
        sb.AppendLine();
        sb.AppendLine($"* **Weight Standard Deviation Formula**: `Math.Sqrt(2.0 / inFeatures)` (He Normal initialization)");
        if (config.HasFC1)
        {
            var scale = (config.Name == "V1" || config.Name == "V4") ? "0.02f" : ((config.Name == "V14") ? "0.15f" : "0.05f");
            sb.AppendLine($"* **FC1 Dense Weights Scale Factor**: `{scale}` (Prevents early pre-activation explosion)");
        }
        var fc2Scale = (config.Name == "V1" || config.Name == "V4") ? "0.05f" : "1.0f";
        sb.AppendLine($"* **FC2 Dense Weights Scale Factor**: `{fc2Scale}` (Stabilizes output logit distribution)");
        var biasInit = (config.Name == "V1" || config.Name == "V4") ? "0.1f positive constant (Prevents dead ReLUs in FP32)" : "0.0f flat";
        sb.AppendLine($"* **FC1 Bias Initialization**: `{biasInit}`");
        sb.AppendLine();
        sb.AppendLine("## 4. Optimizer & Schedules");
        sb.AppendLine();
        sb.AppendLine("* **Optimizer**: Unified Adam / AdamW with Weight Decay");
        sb.AppendLine($"* **Momentum Hyperparameters**: $\\beta_1 = {config.Beta1}$, $\\beta_2 = {config.Beta2}$, $\\epsilon = 1e-8$");
        sb.AppendLine($"* **Learning Rate Schedule**: Cosine Annealing (OneCycleLR) peaked at **{config.MaxLR:F5}** (Start = LR/25, End = LR/1000)");
        sb.AppendLine();
        sb.AppendLine("## 5. Training Hyperparameters");
        sb.AppendLine();
        sb.AppendLine($"* **Batch Size**: {config.BatchSize}");
        sb.AppendLine($"* **Total Training Steps**: {config.TotalSteps}");
        sb.AppendLine($"* **Batches per Epoch**: {config.BatchesPerEpoch}");
        sb.AppendLine();
        sb.AppendLine("## 6. Performance Results");
        sb.AppendLine();
        sb.AppendLine("| Phase / Run | Test Accuracy (%) | GPU Training Time (ms) |");
        sb.AppendLine("| :--- | :---: | :---: |");
        sb.AppendLine($"| **Warmup Run** | {warmupAcc:F2}% | {warmupTime:F3} ms |");
        for (var i = 0; i < measuredAccs.Length; i++)
        {
            sb.AppendLine($"| **Measured Run #{i + 1}** | {measuredAccs[i]:F2}% | {measuredTimes[i]:F3} ms |");
        }
        sb.AppendLine($"| **MEAN METRIC** | **{meanAcc:F2}%** | **{meanTime:F3} ms** |");
        sb.AppendLine();

        if (profileTimes != null && profileTimes.Count > 0)
        {
            sb.AppendLine("## 7. Microsecond-Level Per-Kernel Timing Breakdown");
            sb.AppendLine();
            sb.AppendLine("The table below breaks down the measured step execution times for the different layers/operations within a single training iteration under profiling mode (host event synchronization active).");
            sb.AppendLine();
            sb.AppendLine("| Kernel / Operation | Min (ms) | Mean (ms) | Max (ms) | Total (ms) | Description / Role |");
            sb.AppendLine("| :--- | :---: | :---: | :---: | :---: | :--- |");

            void AddRow(string name, string desc)
            {
                if (profileTimes.TryGetValue(name, out var list) && list.Count > 0)
                {
                    float min = float.MaxValue, max = float.MinValue, sum = 0;
                    for (var i = 0; i < list.Count; i++)
                    {
                        var t = list[i];
                        if (t < min) min = t;
                        if (t > max) max = t;
                        sum += t;
                    }
                    var mean = sum / list.Count;
                    sb.AppendLine($"| `{name}` | {min:F3} | {mean:F3} | {max:F3} | {sum:F2} | {desc} |");
                }
            }

            AddRow("clear_gradient", "Resets weight/bias gradient buffers before step");
            AddRow("conv1_forward", "1-bit input spatial convolution, ReLU + MaxPool");
            AddRow("conv2_forward", "Channel-wise convolution, ReLU + MaxPool");
            AddRow("fc1_forward", "Fully connected hidden layer forward projection");
            AddRow("fc2_forward", "Fully connected final projection (logits)");
            AddRow("fc2_backward", "Logits Softmax backprop and activation gradient");
            AddRow("fc2_bwd_weights", "Parallel shared reduction weights gradient (Zero-Atomics)");
            AddRow("fc1_backward", "Backpropagation of FC1 hidden layer activation gradients");
            AddRow("fc1_bwd_weights", "Backpropagation of FC1 filter & bias gradients");
            AddRow("conv2_backward", "Backpropagation of Conv2 filter & bias gradients");
            AddRow("conv1_backward", "Backpropagation of Conv1 filter & bias gradients");
            AddRow("adam_update", "Parameter update step with dynamic Cosine Annealing");
            sb.AppendLine();
        }

        sb.AppendLine("---");

        var dir = AppDomain.CurrentDomain.BaseDirectory;
        while (!string.IsNullOrEmpty(dir) && !File.Exists(Path.Combine(dir, "CudaSharp.slnx")))
        {
            dir = Path.GetDirectoryName(dir);
        }
        var reportsDir = string.IsNullOrEmpty(dir)
            ? Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "reports")
            : Path.Combine(dir, "reports");

        Directory.CreateDirectory(reportsDir);
        var fileName = config.Name.StartsWith("V", StringComparison.OrdinalIgnoreCase) ? $"{config.Name}_report.md" : $"V{config.Name}_report.md";
        var reportPath = Path.Combine(reportsDir, fileName);
        File.WriteAllText(reportPath, sb.ToString());
        Console.WriteLine($"[REPORT] Programmatic markdown report written to reports/{fileName} successfully!");
    }

    static string? ResolveCudaIncludePath()
    {
        var cudaPath = Environment.GetEnvironmentVariable("CUDA_PATH");
        if (!string.IsNullOrEmpty(cudaPath))
        {
            var includePath = Path.Combine(cudaPath, "include");
            if (File.Exists(Path.Combine(includePath, "cuda_fp16.h")))
            {
                return includePath;
            }
        }

        // Search in common directories on J:\ and C:\
        var searchRoots = new[]
        {
            @"J:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
            @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        };
        foreach (var root in searchRoots)
        {
            if (Directory.Exists(root))
            {
                var subdirs = Directory.GetDirectories(root);
                // Sort subdirs to get the latest version if multiple exist
                Array.Sort(subdirs, StringComparer.OrdinalIgnoreCase);
                for (var i = subdirs.Length - 1; i >= 0; i--)
                {
                    var includePath = Path.Combine(subdirs[i], "include");
                    if (File.Exists(Path.Combine(includePath, "cuda_fp16.h")))
                    {
                        Console.WriteLine($"[JIT] Located CUDA include folder: {includePath}");
                        return includePath;
                    }
                }
            }
        }

        return null;
    }
}
