using System;
using System.Text;
using BenchmarkDotNet.Attributes;
using static CudaSharp.nvcuda;
using static CudaSharp.nvrtc;

namespace CudaSharp.ComparisonBenchmarks;

[BenchmarkCategory("LaunchKernel")]
public unsafe class LaunchKernelBench
{
    const string KernelName = "noop";
    const string KernelSource =
        """
        extern "C" __global__ void noop(int a, int b)
        {
        }
        """;

    CUcontext _context;
    CUmodule _module;
    CUfunction _function;

    public LaunchKernelBench() => CuInit.EnsureInit();

    [GlobalSetup]
    public void Setup()
    {
        cuDeviceGet(out var device, 0).Ok();
        cuCtxCreate(out _context, CUctx_flags.CU_CTX_SCHED_AUTO, device).Ok();
        cuCtxSetCurrent(_context).Ok();

        var ptx = CompileKernel(KernelSource, KernelName);
        cuModuleLoadData(out _module, ptx).Ok();
        cuModuleGetFunction(out _function, _module, KernelName).Ok();
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        if (_context.Value == IntPtr.Zero) { return; }
        cuCtxSetCurrent(_context).Ok();
        if (_module.Value != IntPtr.Zero)
        { cuModuleUnload(_module).Ok(); }
        cuCtxDestroy(_context).Ok();
    }

    [Benchmark(Baseline = true)]
    public void cuLaunchKernel_Raw_CtxSync()
    {
        var args = stackalloc int[] { 1, 2 };
        var kernelParams = stackalloc void*[] { &args[0], &args[1] };

        cuLaunchKernel(_function,
            1, 1, 1,
            1, 1, 1,
            0, default,
            kernelParams, null).Ok();

        cuCtxSynchronize().Ok();
    }

    [Benchmark]
    public void cuLaunchKernel_Overload_CtxSync()
    {
        cuLaunchKernel(_function,
            1, 1, 1,
            1, 1, 1,
            0, default,
            1, 2).Ok();

        cuCtxSynchronize().Ok();
    }

    [Benchmark]
    public void cuLaunchKernelEx_CtxSync()
    {
        var args = stackalloc int[] { 1, 2 };
        var kernelParams = stackalloc void*[] { &args[0], &args[1] };

        var config = new CUlaunchConfig
        {
            gridDimX = 1,
            gridDimY = 1,
            gridDimZ = 1,
            blockDimX = 1,
            blockDimY = 1,
            blockDimZ = 1,
            sharedMemBytes = 0,
            hStream = default,
            attrs = null,
            numAttrs = 0,
        };

        cuLaunchKernelEx(config,
            _function,
            kernelParams,
            null).Ok();

        cuCtxSynchronize().Ok();
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
