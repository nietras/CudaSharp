using System;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using static CudaSharp.nvcuda;
using static CudaSharp.nvrtc;

namespace CudaSharp.Test;

[TestClass]
public class DeviceKernelLaunchTest
{
    public DeviceKernelLaunchTest()
    {
        try
        {
            cuInit().Ok();
        }
        catch (Exception ex)
        {
            Assert.Inconclusive($"CUDA initialization failed: {ex.Message}");
        }
    }

    [TestMethod]
    public unsafe void CompileDeviceKernelLaunch_Blackwell()
    {
        var kernelSource = @"
            __global__ void childKernel(int* data) {
                int tid = threadIdx.x;
                data[tid] += 1;
            }

            __global__ void parentKernel(int* data) {
                // Device-side kernel launch
                childKernel<<<1, 32>>>(data);
            }
        ";

        nvrtcCreateProgram(out var prog, kernelSource, "device_launch.cu", 0, [], []).Ok();

        // Target Blackwell GPU compute capability 100/120.
        // For RTX 50 series, we check either compute_120 or compute_100.
        string[] options = [
            "--gpu-architecture=compute_100", 
            "-rdc=true"
        ];

        nvrtcResult compileResult;
        var optionPointers = stackalloc byte*[options.Length];
        var allocatedOptions = new IntPtr[options.Length];
        try
        {
            for (int i = 0; i < options.Length; i++)
            {
                var optionBytes = Encoding.UTF8.GetBytes(options[i] + '\0');
                allocatedOptions[i] = System.Runtime.InteropServices.Marshal.AllocHGlobal(optionBytes.Length);
                System.Runtime.InteropServices.Marshal.Copy(optionBytes, 0, allocatedOptions[i], optionBytes.Length);
                optionPointers[i] = (byte*)allocatedOptions[i];
            }
            
            compileResult = nvrtcCompileProgram(prog, options.Length, optionPointers);
        }
        finally
        {
            for (var i = 0; i < allocatedOptions.Length; i++)
            {
                if (allocatedOptions[i] != IntPtr.Zero)
                    System.Runtime.InteropServices.Marshal.FreeHGlobal(allocatedOptions[i]);
            }
        }

        if (compileResult != nvrtcResult.NVRTC_SUCCESS)
        {
            nvrtcGetProgramLogSize(prog, out var logSize).Ok();
            var logBuffer = new byte[logSize];
            nvrtcGetProgramLog(prog, logBuffer).Ok();
            var log = Encoding.UTF8.GetString(logBuffer).TrimEnd('\0');
            
            if (log.Contains("bad architecture") || log.Contains("unrecognized architecture") || log.Contains("unsupported architecture"))
            {
                Assert.Inconclusive($"NVRTC doesn't support Blackwell architecture on this machine. Log: {log}");
            }
            else
            {
                Assert.Fail($"Compilation failed:\n{log}");
            }
        }

        nvrtcGetPTXSize(prog, out var ptxSize).Ok();
        Assert.IsTrue(ptxSize > 0);

        nvrtcDestroyProgram(ref prog).Ok();
    }

    [TestMethod]
    public unsafe void CompileDeviceKernelLaunchFunctionPointer_Blackwell()
    {
        var kernelSource = @"
            __global__ void childKernel(int* data) {
                int tid = threadIdx.x;
                data[tid] += 1;
            }

            typedef void (*child_kernel_t)(int*);

            __global__ void parentKernel(child_kernel_t child, int* data) {
                // Device-side kernel launch using function pointer
                child<<<1, 32>>>(data);
            }
        ";

        nvrtcCreateProgram(out var prog, kernelSource, "device_launch_ptr.cu", 0, [], []).Ok();

        string[] options = [
            "--gpu-architecture=compute_100", 
            "-rdc=true"
        ];

        nvrtcResult compileResult;
        var optionPointers = stackalloc byte*[options.Length];
        var allocatedOptions = new IntPtr[options.Length];
        try
        {
            for (int i = 0; i < options.Length; i++)
            {
                var optionBytes = Encoding.UTF8.GetBytes(options[i] + '\0');
                allocatedOptions[i] = System.Runtime.InteropServices.Marshal.AllocHGlobal(optionBytes.Length);
                System.Runtime.InteropServices.Marshal.Copy(optionBytes, 0, allocatedOptions[i], optionBytes.Length);
                optionPointers[i] = (byte*)allocatedOptions[i];
            }
            
            compileResult = nvrtcCompileProgram(prog, options.Length, optionPointers);
        }
        finally
        {
            for (var i = 0; i < allocatedOptions.Length; i++)
            {
                if (allocatedOptions[i] != IntPtr.Zero)
                    System.Runtime.InteropServices.Marshal.FreeHGlobal(allocatedOptions[i]);
            }
        }

        if (compileResult != nvrtcResult.NVRTC_SUCCESS)
        {
            nvrtcGetProgramLogSize(prog, out var logSize).Ok();
            var logBuffer = new byte[logSize];
            nvrtcGetProgramLog(prog, logBuffer).Ok();
            var log = Encoding.UTF8.GetString(logBuffer).TrimEnd('\0');
            
            if (log.Contains("bad architecture") || log.Contains("unrecognized architecture") || log.Contains("unsupported architecture"))
            {
                Assert.Inconclusive($"NVRTC doesn't support Blackwell architecture on this machine. Log: {log}");
            }
            else
            {
                Assert.Fail($"Compilation failed:\n{log}");
            }
        }

        nvrtcGetPTXSize(prog, out var ptxSize).Ok();
        Assert.IsTrue(ptxSize > 0);

        nvrtcDestroyProgram(ref prog).Ok();
    }
}
