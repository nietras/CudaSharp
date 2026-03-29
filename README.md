# CudaSharp
![.NET](https://img.shields.io/badge/net10.0-5C2D91?logo=.NET&labelColor=gray)
![C#](https://img.shields.io/badge/C%23-14.0-239120?labelColor=gray)
[![Build Status](https://github.com/nietras/CudaSharp/actions/workflows/dotnet.yml/badge.svg?branch=main)](https://github.com/nietras/CudaSharp/actions/workflows/dotnet.yml)
[![Super-Linter](https://github.com/nietras/CudaSharp/actions/workflows/super-linter.yml/badge.svg)](https://github.com/marketplace/actions/super-linter)
[![codecov](https://codecov.io/gh/nietras/CudaSharp/branch/main/graph/badge.svg?token=WN56CR3X0D)](https://codecov.io/gh/nietras/CudaSharp)
[![CodeQL](https://github.com/nietras/CudaSharp/workflows/CodeQL/badge.svg)](https://github.com/nietras/CudaSharp/actions?query=workflow%3ACodeQL)
[![Nuget](https://img.shields.io/nuget/v/CudaSharp?color=purple)](https://www.nuget.org/packages/CudaSharp/)
[![Release](https://img.shields.io/github/v/release/nietras/CudaSharp)](https://github.com/nietras/CudaSharp/releases/)
[![downloads](https://img.shields.io/nuget/dt/CudaSharp)](https://www.nuget.org/packages/CudaSharp)
![Size](https://img.shields.io/github/repo-size/nietras/CudaSharp.svg)
[![License](https://img.shields.io/github/license/nietras/CudaSharp)](https://github.com/nietras/CudaSharp/blob/main/LICENSE)
[![Blog](https://img.shields.io/badge/blog-nietras.com-4993DD)](https://nietras.com)
![GitHub Repo stars](https://img.shields.io/github/stars/nietras/CudaSharp?style=flat)

Low-level CUDA interop in modern C#. Cross-platform, trimmable and
AOT/NativeAOT compatible.

⚠ WORK IN PROGRESS - definitions and interop may be wrong and untested.

⭐ Please star this project if you like it. ⭐

[Example](#example) | [Example Catalogue](#example-catalogue) | [Public API Reference](#public-api-reference)

## Example
```csharp
cuInit().Ok();

cuDeviceGet(out var device, 0).Ok();
cuCtxCreate(out var context, CUctx_flags.CU_CTX_SCHED_AUTO, device).Ok();

var kernelSource =
    """
    extern "C" __global__ void saxpy(float a, float *x, float *y, float *out, size_t n)
    {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < n) {
            out[tid] = a * x[tid] + y[tid];
        }
    }
    """;
nvrtcCreateProgram(out var prog, kernelSource, "saxpy.cu", 0, [], []).Ok();

var compileResult = nvrtcCompileProgram(prog, 0, []);
if (compileResult != nvrtcResult.NVRTC_SUCCESS)
{
    nvrtcGetProgramLogSize(prog, out var logSize).Ok();
    var logBuffer = new byte[logSize];
    nvrtcGetProgramLog(prog, logBuffer).Ok();
    var log = Encoding.UTF8.GetString(logBuffer).TrimEnd('\0');
    Assert.Fail($"Compilation failed:\n{log}");
}

nvrtcGetPTXSize(prog, out var ptxSize).Ok();
var ptxBuffer = new byte[ptxSize];
nvrtcGetPTX(prog, ptxBuffer).Ok();

nvrtcDestroyProgram(ref prog).Ok();

cuModuleLoadData(out var module, ptxBuffer).Ok();
cuModuleGetFunction(out var function, module, "saxpy").Ok();

var n = 1024;
var a = 2.5f;
var bytes = (nuint)(n * sizeof(float));

// Allocate Device Memory
cuMemAlloc(out var d_x, bytes).Ok();
cuMemAlloc(out var d_y, bytes).Ok();
cuMemAlloc(out var d_out, bytes).Ok();

// Allocate Host Memory (Pinned)
cuMemHostAlloc(out var h_x_ptr, bytes, 0).Ok();
cuMemHostAlloc(out var h_y_ptr, bytes, 0).Ok();
cuMemHostAlloc(out var h_out_ptr, bytes, 0).Ok();

// Initialize Host Data
var h_x = new Span<float>((void*)h_x_ptr, n);
var h_y = new Span<float>((void*)h_y_ptr, n);
var h_out = new Span<float>((void*)h_out_ptr, n);

for (var i = 0; i < n; i++)
{
    h_x[i] = i;
    h_y[i] = i * 2;
}

cuMemcpyHtoD(d_x, h_x_ptr, bytes).Ok();
cuMemcpyHtoD(d_y, h_y_ptr, bytes).Ok();

// Kernel params
void*[] args = [&a, &d_x, &d_y, &d_out, &n];
var argsPtrs = new IntPtr[args.Length];
for (var i = 0; i < args.Length; i++)
{
    argsPtrs[i] = (IntPtr)args[i];
}

cuLaunchKernel(
    function,
    (uint)((n + 255) / 256), 1, 1, // Grid
    256, 1, 1, // Block
    0, new CUstream(IntPtr.Zero),
    new ReadOnlySpan<IntPtr>(argsPtrs),
    []
).Ok();

cuCtxSynchronize().Ok();

cuMemcpyDtoH(h_out_ptr, d_out, bytes).Ok();

for (var i = 0; i < n; i++)
{
    var actual = h_out[i];
    var expected = a * h_x[i] + h_y[i];
    Assert.AreEqual(expected, actual, 1e-5);
}

// Cleanup
cuMemFreeHost(h_x_ptr);
cuMemFreeHost(h_y_ptr);
cuMemFreeHost(h_out_ptr);

cuMemFree(d_x);
cuMemFree(d_y);
cuMemFree(d_out);
cuCtxDestroy(context);

// Above example code is for demonstration purposes only.
// Short names and repeated constants are only for demonstration.
```

For more examples see [Example Catalogue](#example-catalogue).

## Benchmarks
Benchmarks.

### Detailed Benchmarks

#### Comparison Benchmarks

##### TestBench Benchmark Results

###### AMD.Ryzen.9.9950X - TestBench Benchmark Results (CudaSharp 0.0.0.0, System 10.0.326.7603)

| Method          | Scope | Count | Mean      | Ratio | Allocated | Alloc Ratio |
|---------------- |------ |------ |----------:|------:|----------:|------------:|
| CudaSharp______ | Test  | 25000 | 0.0004 ns |     ? |         - |           ? |


## Example Catalogue
The following examples are available in [ReadMeTest.cs](src/CudaSharp.XyzTest/ReadMeTest.cs).

### Example - Empty
```csharp
cuInit().Ok();

cuDeviceGet(out var device, 0).Ok();
cuCtxCreate(out var context, CUctx_flags.CU_CTX_SCHED_AUTO, device).Ok();

var kernelSource =
    """
    extern "C" __global__ void saxpy(float a, float *x, float *y, float *out, size_t n)
    {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < n) {
            out[tid] = a * x[tid] + y[tid];
        }
    }
    """;
nvrtcCreateProgram(out var prog, kernelSource, "saxpy.cu", 0, [], []).Ok();

var compileResult = nvrtcCompileProgram(prog, 0, []);
if (compileResult != nvrtcResult.NVRTC_SUCCESS)
{
    nvrtcGetProgramLogSize(prog, out var logSize).Ok();
    var logBuffer = new byte[logSize];
    nvrtcGetProgramLog(prog, logBuffer).Ok();
    var log = Encoding.UTF8.GetString(logBuffer).TrimEnd('\0');
    Assert.Fail($"Compilation failed:\n{log}");
}

nvrtcGetPTXSize(prog, out var ptxSize).Ok();
var ptxBuffer = new byte[ptxSize];
nvrtcGetPTX(prog, ptxBuffer).Ok();

nvrtcDestroyProgram(ref prog).Ok();

cuModuleLoadData(out var module, ptxBuffer).Ok();
cuModuleGetFunction(out var function, module, "saxpy").Ok();

var n = 1024;
var a = 2.5f;
var bytes = (nuint)(n * sizeof(float));

// Allocate Device Memory
cuMemAlloc(out var d_x, bytes).Ok();
cuMemAlloc(out var d_y, bytes).Ok();
cuMemAlloc(out var d_out, bytes).Ok();

// Allocate Host Memory (Pinned)
cuMemHostAlloc(out var h_x_ptr, bytes, 0).Ok();
cuMemHostAlloc(out var h_y_ptr, bytes, 0).Ok();
cuMemHostAlloc(out var h_out_ptr, bytes, 0).Ok();

// Initialize Host Data
var h_x = new Span<float>((void*)h_x_ptr, n);
var h_y = new Span<float>((void*)h_y_ptr, n);
var h_out = new Span<float>((void*)h_out_ptr, n);

for (var i = 0; i < n; i++)
{
    h_x[i] = i;
    h_y[i] = i * 2;
}

cuMemcpyHtoD(d_x, h_x_ptr, bytes).Ok();
cuMemcpyHtoD(d_y, h_y_ptr, bytes).Ok();

// Kernel params
void*[] args = [&a, &d_x, &d_y, &d_out, &n];
var argsPtrs = new IntPtr[args.Length];
for (var i = 0; i < args.Length; i++)
{
    argsPtrs[i] = (IntPtr)args[i];
}

cuLaunchKernel(
    function,
    (uint)((n + 255) / 256), 1, 1, // Grid
    256, 1, 1, // Block
    0, new CUstream(IntPtr.Zero),
    new ReadOnlySpan<IntPtr>(argsPtrs),
    []
).Ok();

cuCtxSynchronize().Ok();

cuMemcpyDtoH(h_out_ptr, d_out, bytes).Ok();

for (var i = 0; i < n; i++)
{
    var actual = h_out[i];
    var expected = a * h_x[i] + h_y[i];
    Assert.AreEqual(expected, actual, 1e-5);
}

// Cleanup
cuMemFreeHost(h_x_ptr);
cuMemFreeHost(h_y_ptr);
cuMemFreeHost(h_out_ptr);

cuMemFree(d_x);
cuMemFree(d_y);
cuMemFree(d_out);
cuCtxDestroy(context);

// Above example code is for demonstration purposes only.
// Short names and repeated constants are only for demonstration.
```

## Public API Reference
```csharp
[assembly: System.CLSCompliant(false)]
[assembly: System.Reflection.AssemblyMetadata("IsAotCompatible", "True")]
[assembly: System.Reflection.AssemblyMetadata("IsTrimmable", "True")]
[assembly: System.Reflection.AssemblyMetadata("RepositoryUrl", "https://github.com/nietras/CudaSharp/")]
[assembly: System.Resources.NeutralResourcesLanguage("en")]
[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("CudaSharp.Benchmarks")]
[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("CudaSharp.ComparisonBenchmarks")]
[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("CudaSharp.Test")]
[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("CudaSharp.XyzTest")]
[assembly: System.Runtime.Versioning.TargetFramework(".NETCoreApp,Version=v10.0", FrameworkDisplayName=".NET 10.0")]
namespace CudaSharp
{
    public class CudaException : System.Exception
    {
        public CudaException(string message) { }
    }
    public sealed class CudaException<TResult> : System.Exception
        where TResult :  unmanaged, System.Enum
    {
        public CudaException(TResult result, string message) { }
        public TResult Result { get; }
    }
    public static class DllResolver
    {
        public static void Register() { }
    }
    public static class nvcuda
    {
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuCtxCreate(out CudaSharp.nvcuda.CUcontext pctx, CudaSharp.nvcuda.CUctx_flags flags, CudaSharp.nvcuda.CUdevice dev) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuCtxDestroy(CudaSharp.nvcuda.CUcontext ctx) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuCtxDisablePeerAccess(CudaSharp.nvcuda.CUcontext peerContext) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuCtxEnablePeerAccess(CudaSharp.nvcuda.CUcontext peerContext, uint Flags) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuCtxGetCacheConfig(out CudaSharp.nvcuda.CUfunc_cache pconfig) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuCtxGetCurrent(out CudaSharp.nvcuda.CUcontext pctx) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuCtxGetDevice(out CudaSharp.nvcuda.CUdevice device) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuCtxGetLimit(out System.UIntPtr pvalue, CudaSharp.nvcuda.CUlimit limit) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuCtxGetSharedMemConfig(out CudaSharp.nvcuda.CUsharedconfig pConfig) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuCtxPopCurrent(out CudaSharp.nvcuda.CUcontext pctx) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuCtxPushCurrent(CudaSharp.nvcuda.CUcontext ctx) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuCtxSetCacheConfig(CudaSharp.nvcuda.CUfunc_cache config) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuCtxSetCurrent(CudaSharp.nvcuda.CUcontext ctx) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuCtxSetLimit(CudaSharp.nvcuda.CUlimit limit, nuint value) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuCtxSetSharedMemConfig(CudaSharp.nvcuda.CUsharedconfig config) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuCtxSynchronize() { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuDestroyExternalMemory(CudaSharp.nvcuda.CUexternalMemory extMem) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuDestroyExternalSemaphore(CudaSharp.nvcuda.CUexternalSemaphore extSem) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuDeviceCanAccessPeer(out int canAccessPeer, CudaSharp.nvcuda.CUdevice dev, CudaSharp.nvcuda.CUdevice peerDev) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuDeviceComputeCapability(out int major, out int minor, CudaSharp.nvcuda.CUdevice dev) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuDeviceGet(out CudaSharp.nvcuda.CUdevice device, int ordinal) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuDeviceGetAttribute(out int pi, CudaSharp.nvcuda.CUdevice_attribute attrib, CudaSharp.nvcuda.CUdevice dev) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuDeviceGetCount(out int count) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static unsafe CudaSharp.nvcuda.CUresult cuDeviceGetLuid(byte* luid, out uint deviceNodeMask, CudaSharp.nvcuda.CUdevice dev) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuDeviceGetName(System.Span<byte> name, int len, CudaSharp.nvcuda.CUdevice dev) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuDeviceGetProperties(out CudaSharp.nvcuda.CUdevprop pProp, CudaSharp.nvcuda.CUdevice dev) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuDeviceGetUuid(out CudaSharp.nvcuda.CUuuid uuid, CudaSharp.nvcuda.CUdevice dev) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuDevicePrimaryCtxGetState(CudaSharp.nvcuda.CUdevice dev, out uint flags, out int active) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuDevicePrimaryCtxRelease(CudaSharp.nvcuda.CUdevice dev) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuDevicePrimaryCtxReset(CudaSharp.nvcuda.CUdevice dev) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuDevicePrimaryCtxRetain(out CudaSharp.nvcuda.CUcontext pctx, CudaSharp.nvcuda.CUdevice dev) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuDevicePrimaryCtxSetFlags(CudaSharp.nvcuda.CUdevice dev, uint flags) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuDeviceTotalMem(out System.UIntPtr bytes, CudaSharp.nvcuda.CUdevice dev) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuDriverGetVersion(out int driverVersion) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuEventCreate(out CudaSharp.nvcuda.CUevent phEvent, uint Flags) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuEventDestroy(CudaSharp.nvcuda.CUevent hEvent) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuEventElapsedTime(out float pMilliseconds, CudaSharp.nvcuda.CUevent hStart, CudaSharp.nvcuda.CUevent hEnd) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuEventQuery(CudaSharp.nvcuda.CUevent hEvent) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuEventRecord(CudaSharp.nvcuda.CUevent hEvent, CudaSharp.nvcuda.CUstream hStream) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuEventSynchronize(CudaSharp.nvcuda.CUevent hEvent) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuExternalMemoryGetMappedBuffer(out CudaSharp.nvcuda.CUdeviceptr devPtr, CudaSharp.nvcuda.CUexternalMemory extMem, in CudaSharp.nvcuda.CUDA_EXTERNAL_MEMORY_BUFFER_DESC bufferDesc) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuExternalMemoryGetMappedMipmappedArray(out System.IntPtr mipmappedArray, CudaSharp.nvcuda.CUexternalMemory extMem, in CudaSharp.nvcuda.CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC mipmapDesc) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuGraphAddKernelNode(out CudaSharp.nvcuda.CUgraphNode phGraphNode, CudaSharp.nvcuda.CUgraph hGraph, System.ReadOnlySpan<CudaSharp.nvcuda.CUgraphNode> dependencies, nuint numDependencies, in CudaSharp.nvcuda.CUDA_KERNEL_NODE_PARAMS nodeParams) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuGraphCreate(out CudaSharp.nvcuda.CUgraph phGraph, uint flags) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuGraphDestroy(CudaSharp.nvcuda.CUgraph hGraph) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuGraphInstantiate(out CudaSharp.nvcuda.CUgraphExec phGraphExec, CudaSharp.nvcuda.CUgraph hGraph, out CudaSharp.nvcuda.CUgraphNode phErrorNode, System.Span<byte> logBuffer, nuint bufferSize) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuGraphLaunch(CudaSharp.nvcuda.CUgraphExec hGraphExec, CudaSharp.nvcuda.CUstream hStream) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuImportExternalMemory(out CudaSharp.nvcuda.CUexternalMemory extMem, in CudaSharp.nvcuda.CUDA_EXTERNAL_MEMORY_HANDLE_DESC memHandleDesc) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuImportExternalSemaphore(out CudaSharp.nvcuda.CUexternalSemaphore extSem, in CudaSharp.nvcuda.CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC semHandleDesc) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuInit(uint flags = 0) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuLaunchKernel(CudaSharp.nvcuda.CUfunction f, uint gridDimX, uint gridDimY, uint gridDimZ, uint blockDimX, uint blockDimY, uint blockDimZ, uint sharedMemBytes, CudaSharp.nvcuda.CUstream hStream, System.ReadOnlySpan<System.IntPtr> kernelParams, System.ReadOnlySpan<System.IntPtr> extra) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuMemAlloc(out CudaSharp.nvcuda.CUdeviceptr dptr, nuint bytesize) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuMemAllocPitch(out CudaSharp.nvcuda.CUdeviceptr dptr, out System.UIntPtr pPitch, nuint WidthInBytes, nuint Height, uint ElementSizeBytes) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuMemFree(CudaSharp.nvcuda.CUdeviceptr dptr) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuMemFreeHost(nint p) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuMemGetInfo(out System.UIntPtr free, out System.UIntPtr total) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuMemHostAlloc(out System.IntPtr pp, nuint bytesize, uint Flags) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuMemcpy2D(in CudaSharp.nvcuda.CUDA_MEMCPY2D pCopy) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuMemcpy2DAsync(in CudaSharp.nvcuda.CUDA_MEMCPY2D pCopy, CudaSharp.nvcuda.CUstream hStream) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuMemcpy3D(in CudaSharp.nvcuda.CUDA_MEMCPY3D pCopy) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuMemcpy3DAsync(in CudaSharp.nvcuda.CUDA_MEMCPY3D pCopy, CudaSharp.nvcuda.CUstream hStream) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuMemcpyDtoDAsync(CudaSharp.nvcuda.CUdeviceptr dstDevice, CudaSharp.nvcuda.CUdeviceptr srcDevice, nuint bytesize, CudaSharp.nvcuda.CUstream hStream) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuMemcpyDtoH(nint dstHost, CudaSharp.nvcuda.CUdeviceptr srcDevice, nuint bytesize) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuMemcpyDtoHAsync(nint dstHost, CudaSharp.nvcuda.CUdeviceptr srcDevice, nuint bytesize, CudaSharp.nvcuda.CUstream hStream) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuMemcpyHtoD(CudaSharp.nvcuda.CUdeviceptr dstDevice, nint srcHost, nuint bytesize) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuMemcpyHtoDAsync(CudaSharp.nvcuda.CUdeviceptr dstDevice, nint srcHost, nuint bytesize, CudaSharp.nvcuda.CUstream hStream) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuMemsetD16(CudaSharp.nvcuda.CUdeviceptr dstDevice, ushort us, nuint N) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuMemsetD16Async(CudaSharp.nvcuda.CUdeviceptr dstDevice, ushort us, nuint N, CudaSharp.nvcuda.CUstream hStream) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuMemsetD2D16(CudaSharp.nvcuda.CUdeviceptr dstDevice, nuint dstPitch, ushort us, nuint Width, nuint Height) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuMemsetD2D16Async(CudaSharp.nvcuda.CUdeviceptr dstDevice, nuint dstPitch, ushort us, nuint Width, nuint Height, CudaSharp.nvcuda.CUstream hStream) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuMemsetD2D32(CudaSharp.nvcuda.CUdeviceptr dstDevice, nuint dstPitch, uint ui, nuint Width, nuint Height) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuMemsetD2D32Async(CudaSharp.nvcuda.CUdeviceptr dstDevice, nuint dstPitch, uint ui, nuint Width, nuint Height, CudaSharp.nvcuda.CUstream hStream) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuMemsetD2D8(CudaSharp.nvcuda.CUdeviceptr dstDevice, nuint dstPitch, byte uc, nuint Width, nuint Height) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuMemsetD2D8Async(CudaSharp.nvcuda.CUdeviceptr dstDevice, nuint dstPitch, byte uc, nuint Width, nuint Height, CudaSharp.nvcuda.CUstream hStream) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuMemsetD32(CudaSharp.nvcuda.CUdeviceptr dstDevice, uint ui, nuint N) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuMemsetD32Async(CudaSharp.nvcuda.CUdeviceptr dstDevice, uint ui, nuint N, CudaSharp.nvcuda.CUstream hStream) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuMemsetD8(CudaSharp.nvcuda.CUdeviceptr dstDevice, byte uc, nuint N) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuMemsetD8Async(CudaSharp.nvcuda.CUdeviceptr dstDevice, byte uc, nuint N, CudaSharp.nvcuda.CUstream hStream) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda", StringMarshalling=System.Runtime.InteropServices.StringMarshalling.Utf8)]
        public static CudaSharp.nvcuda.CUresult cuModuleGetFunction(out CudaSharp.nvcuda.CUfunction hfunc, CudaSharp.nvcuda.CUmodule hmod, string name) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda", StringMarshalling=System.Runtime.InteropServices.StringMarshalling.Utf8)]
        public static CudaSharp.nvcuda.CUresult cuModuleLoad(out CudaSharp.nvcuda.CUmodule module, string fname) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuModuleLoadData(out CudaSharp.nvcuda.CUmodule module, System.ReadOnlySpan<byte> image) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuModuleUnload(CudaSharp.nvcuda.CUmodule hmod) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(out int numBlocks, CudaSharp.nvcuda.CUfunction func, int blockSize, nuint dynamicSMemSize) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(out int numBlocks, CudaSharp.nvcuda.CUfunction func, int blockSize, nuint dynamicSMemSize, uint flags) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda", StringMarshalling=System.Runtime.InteropServices.StringMarshalling.Utf8)]
        public static CudaSharp.nvcuda.CUresult cuProfilerInitialize(string configFile, string outputMode, CudaSharp.nvcuda.CUprofiler_outputMode mode) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuProfilerStart() { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuProfilerStop() { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuSignalExternalSemaphoresAsync(System.ReadOnlySpan<CudaSharp.nvcuda.CUexternalSemaphore> extSemArray, System.ReadOnlySpan<CudaSharp.nvcuda.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS> paramsArray, uint numSemaphores, CudaSharp.nvcuda.CUstream stream) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuStreamBatchMemOp(CudaSharp.nvcuda.CUstream stream, uint count, System.ReadOnlySpan<CudaSharp.nvcuda.CUstreamBatchMemOpParams> paramArray, uint flags) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuStreamCreate(out CudaSharp.nvcuda.CUstream pStream, uint Flags) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuStreamDestroy(CudaSharp.nvcuda.CUstream hStream) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuStreamSynchronize(CudaSharp.nvcuda.CUstream hStream) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuStreamWaitValue32(CudaSharp.nvcuda.CUstream stream, CudaSharp.nvcuda.CUdeviceptr addr, uint value, uint flags) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuStreamWaitValue64(CudaSharp.nvcuda.CUstream stream, CudaSharp.nvcuda.CUdeviceptr addr, ulong value, uint flags) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuStreamWriteValue32(CudaSharp.nvcuda.CUstream stream, CudaSharp.nvcuda.CUdeviceptr addr, uint value, uint flags) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuStreamWriteValue64(CudaSharp.nvcuda.CUstream stream, CudaSharp.nvcuda.CUdeviceptr addr, ulong value, uint flags) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuSurfObjectCreate(out CudaSharp.nvcuda.CUsurfObject pSurfObject, in CudaSharp.nvcuda.CUDA_RESOURCE_DESC pResDesc) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuSurfObjectDestroy(CudaSharp.nvcuda.CUsurfObject surfObject) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuSurfObjectGetResourceDesc(out CudaSharp.nvcuda.CUDA_RESOURCE_DESC pResDesc, CudaSharp.nvcuda.CUsurfObject surfObject) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuSurfRefCreate(out CudaSharp.nvcuda.CUsurfref pSurfRef) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuSurfRefDestroy(CudaSharp.nvcuda.CUsurfref hSurfRef) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuSurfRefGetArray(out CudaSharp.nvcuda.CUarray phArray, CudaSharp.nvcuda.CUsurfref hSurfRef) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuSurfRefSetArray(CudaSharp.nvcuda.CUsurfref hSurfRef, CudaSharp.nvcuda.CUarray hArray, uint Flags) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuTexObjectCreate(out CudaSharp.nvcuda.CUtexObject pTexObject, in CudaSharp.nvcuda.CUDA_RESOURCE_DESC pResDesc, in CudaSharp.nvcuda.CUDA_TEXTURE_DESC pTexDesc, in CudaSharp.nvcuda.CUDA_RESOURCE_VIEW_DESC pResViewDesc) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuTexObjectDestroy(CudaSharp.nvcuda.CUtexObject texObject) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuTexObjectGetResourceDesc(out CudaSharp.nvcuda.CUDA_RESOURCE_DESC pResDesc, CudaSharp.nvcuda.CUtexObject texObject) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuTexObjectGetResourceViewDesc(out CudaSharp.nvcuda.CUDA_RESOURCE_VIEW_DESC pResViewDesc, CudaSharp.nvcuda.CUtexObject texObject) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuTexObjectGetTextureDesc(out CudaSharp.nvcuda.CUDA_TEXTURE_DESC pTexDesc, CudaSharp.nvcuda.CUtexObject texObject) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuTexRefCreate(out CudaSharp.nvcuda.CUtexref pTexRef) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuTexRefDestroy(CudaSharp.nvcuda.CUtexref hTexRef) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuTexRefGetAddress(out CudaSharp.nvcuda.CUdeviceptr pdptr, CudaSharp.nvcuda.CUtexref hTexRef) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuTexRefGetAddressMode(out CudaSharp.nvcuda.CUaddress_mode pam, CudaSharp.nvcuda.CUtexref hTexRef, int dim) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuTexRefGetArray(out CudaSharp.nvcuda.CUarray phArray, CudaSharp.nvcuda.CUtexref hTexRef) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuTexRefGetFilterMode(out CudaSharp.nvcuda.CUfilter_mode pfm, CudaSharp.nvcuda.CUtexref hTexRef) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuTexRefGetFlags(out uint pFlags, CudaSharp.nvcuda.CUtexref hTexRef) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuTexRefGetFormat(out CudaSharp.nvcuda.CUarray_format pFormat, out int pNumPackedComponents, CudaSharp.nvcuda.CUtexref hTexRef) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuTexRefSetAddress(out System.UIntPtr ByteOffset, CudaSharp.nvcuda.CUtexref hTexRef, CudaSharp.nvcuda.CUdeviceptr dptr, nuint bytes) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuTexRefSetAddress2D(CudaSharp.nvcuda.CUtexref hTexRef, in CudaSharp.nvcuda.CUDA_ARRAY_DESCRIPTOR desc, CudaSharp.nvcuda.CUdeviceptr dptr, nuint Pitch) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuTexRefSetAddressMode(CudaSharp.nvcuda.CUtexref hTexRef, int dim, CudaSharp.nvcuda.CUaddress_mode am) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuTexRefSetArray(CudaSharp.nvcuda.CUtexref hTexRef, CudaSharp.nvcuda.CUarray hArray, uint Flags) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuTexRefSetFilterMode(CudaSharp.nvcuda.CUtexref hTexRef, CudaSharp.nvcuda.CUfilter_mode fm) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuTexRefSetFlags(CudaSharp.nvcuda.CUtexref hTexRef, uint Flags) { }
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuTexRefSetFormat(CudaSharp.nvcuda.CUtexref hTexRef, CudaSharp.nvcuda.CUarray_format fmt, int NumPackedComponents) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvcuda")]
        public static CudaSharp.nvcuda.CUresult cuWaitExternalSemaphoresAsync(System.ReadOnlySpan<CudaSharp.nvcuda.CUexternalSemaphore> extSemArray, System.ReadOnlySpan<CudaSharp.nvcuda.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS> paramsArray, uint numSemaphores, CudaSharp.nvcuda.CUstream stream) { }
        public struct CUDA_ARRAY3D_DESCRIPTOR
        {
            public System.UIntPtr Depth;
            public uint Flags;
            public CudaSharp.nvcuda.CUarray_format Format;
            public System.UIntPtr Height;
            public uint NumChannels;
            public System.UIntPtr Width;
        }
        public struct CUDA_ARRAY_DESCRIPTOR
        {
            public CudaSharp.nvcuda.CUarray_format Format;
            public System.UIntPtr Height;
            public uint NumChannels;
            public System.UIntPtr Width;
        }
        public struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC
        {
            public uint flags;
            public ulong offset;
            public ulong size;
        }
        public struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC
        {
            public uint flags;
            public CudaSharp.nvcuda.CUDA_EXTERNAL_MEMORY_HANDLE_DESC_UNION handle;
            public ulong size;
            public CudaSharp.nvcuda.CUexternalMemoryHandleType type;
        }
        public struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC_UNION
        {
            public int fd;
            public System.IntPtr nvSciBufObject;
            public CudaSharp.nvcuda.CUDA_EXTERNAL_MEMORY_HANDLE_DESC_WIN32 win32;
        }
        public struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC_WIN32
        {
            public System.IntPtr handle;
            public System.IntPtr name;
        }
        public struct CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC
        {
            public CudaSharp.nvcuda.CUDA_ARRAY3D_DESCRIPTOR arrayDesc;
            public uint numLevels;
            public ulong offset;
        }
        public struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC
        {
            public uint flags;
            public CudaSharp.nvcuda.CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_UNION handle;
            public CudaSharp.nvcuda.CUexternalSemaphoreHandleType type;
        }
        public struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_UNION
        {
            public int fd;
            public System.IntPtr nvSciSyncObj;
            public CudaSharp.nvcuda.CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_WIN32 win32;
        }
        public struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_WIN32
        {
            public System.IntPtr handle;
            public System.IntPtr name;
        }
        public struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS
        {
            public uint flags;
            public CudaSharp.nvcuda.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_PARAMS params_;
            [System.Runtime.CompilerServices.FixedBuffer(typeof(uint), 16)]
            public CudaSharp.nvcuda.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS.<reserved>e__FixedBuffer reserved;
        }
        public struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_FENCE
        {
            public ulong value;
        }
        public struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_KEYED_MUTEX
        {
            public ulong key;
        }
        public struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_PARAMS
        {
            public CudaSharp.nvcuda.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_FENCE fence;
            public CudaSharp.nvcuda.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_KEYED_MUTEX keyedMutex;
            public System.IntPtr nvSciSyncObj;
        }
        public struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS
        {
            public uint flags;
            public CudaSharp.nvcuda.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_PARAMS params_;
            [System.Runtime.CompilerServices.FixedBuffer(typeof(uint), 16)]
            public CudaSharp.nvcuda.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS.<reserved>e__FixedBuffer reserved;
        }
        public struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_FENCE
        {
            public uint timeoutMs;
            public ulong value;
        }
        public struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_KEYED_MUTEX
        {
            public ulong key;
            public uint timeoutMs;
        }
        public struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_PARAMS
        {
            public CudaSharp.nvcuda.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_FENCE fence;
            public CudaSharp.nvcuda.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_KEYED_MUTEX keyedMutex;
            public System.IntPtr nvSciSyncObj;
        }
        public struct CUDA_KERNEL_NODE_PARAMS
        {
            public uint blockDimX;
            public uint blockDimY;
            public uint blockDimZ;
            public System.IntPtr extra;
            public CudaSharp.nvcuda.CUfunction func;
            public uint gridDimX;
            public uint gridDimY;
            public uint gridDimZ;
            public System.IntPtr kernelParams;
            public uint sharedMemBytes;
        }
        public struct CUDA_MEMCPY2D
        {
            public System.UIntPtr Height;
            public System.UIntPtr WidthInBytes;
            public CudaSharp.nvcuda.CUarray dstArray;
            public CudaSharp.nvcuda.CUdeviceptr dstDevice;
            public System.IntPtr dstHost;
            public CudaSharp.nvcuda.CUmemorytype dstMemoryType;
            public System.UIntPtr dstPitch;
            public System.UIntPtr dstXInBytes;
            public System.UIntPtr dstY;
            public CudaSharp.nvcuda.CUarray srcArray;
            public CudaSharp.nvcuda.CUdeviceptr srcDevice;
            public System.IntPtr srcHost;
            public CudaSharp.nvcuda.CUmemorytype srcMemoryType;
            public System.UIntPtr srcPitch;
            public System.UIntPtr srcXInBytes;
            public System.UIntPtr srcY;
        }
        public struct CUDA_MEMCPY3D
        {
            public System.UIntPtr Depth;
            public System.UIntPtr Height;
            public System.UIntPtr WidthInBytes;
            public CudaSharp.nvcuda.CUarray dstArray;
            public System.IntPtr dstContext;
            public CudaSharp.nvcuda.CUdeviceptr dstDevice;
            public System.UIntPtr dstHeight;
            public System.IntPtr dstHost;
            public System.UIntPtr dstLOD;
            public CudaSharp.nvcuda.CUmemorytype dstMemoryType;
            public System.UIntPtr dstPitch;
            public System.UIntPtr dstXInBytes;
            public System.UIntPtr dstY;
            public System.UIntPtr dstZ;
            public CudaSharp.nvcuda.CUarray srcArray;
            public System.IntPtr srcContext;
            public CudaSharp.nvcuda.CUdeviceptr srcDevice;
            public System.UIntPtr srcHeight;
            public System.IntPtr srcHost;
            public System.UIntPtr srcLOD;
            public CudaSharp.nvcuda.CUmemorytype srcMemoryType;
            public System.UIntPtr srcPitch;
            public System.UIntPtr srcXInBytes;
            public System.UIntPtr srcY;
            public System.UIntPtr srcZ;
        }
        public struct CUDA_MEMCPY_NODE_PARAMS
        {
            public CudaSharp.nvcuda.CUDA_MEMCPY3D copyParams;
        }
        public struct CUDA_MEMSET_NODE_PARAMS
        {
            public CudaSharp.nvcuda.CUdeviceptr dst;
            public uint elementSize;
            public System.UIntPtr height;
            public System.UIntPtr pitch;
            public uint value;
            public System.UIntPtr width;
        }
        public struct CUDA_RESOURCE_DESC
        {
            public uint flags;
            public CudaSharp.nvcuda.CUDA_RESOURCE_DESC_UNION res;
            public CudaSharp.nvcuda.CUresourcetype resType;
        }
        public struct CUDA_RESOURCE_DESC_ARRAY
        {
            public CudaSharp.nvcuda.CUarray hArray;
        }
        public struct CUDA_RESOURCE_DESC_LINEAR
        {
            public CudaSharp.nvcuda.CUdeviceptr devPtr;
            public CudaSharp.nvcuda.CUarray_format format;
            public uint numChannels;
            public System.UIntPtr sizeInBytes;
        }
        public struct CUDA_RESOURCE_DESC_MIPMAPPED_ARRAY
        {
            public System.IntPtr hMipmappedArray;
        }
        public struct CUDA_RESOURCE_DESC_PITCH2D
        {
            public CudaSharp.nvcuda.CUdeviceptr devPtr;
            public CudaSharp.nvcuda.CUarray_format format;
            public System.UIntPtr height;
            public uint numChannels;
            public System.UIntPtr pitchInBytes;
            public System.UIntPtr width;
        }
        public struct CUDA_RESOURCE_DESC_UNION
        {
            public CudaSharp.nvcuda.CUDA_RESOURCE_DESC_ARRAY array;
            public CudaSharp.nvcuda.CUDA_RESOURCE_DESC_LINEAR linear;
            public CudaSharp.nvcuda.CUDA_RESOURCE_DESC_MIPMAPPED_ARRAY mipmap;
            public CudaSharp.nvcuda.CUDA_RESOURCE_DESC_PITCH2D pitch2D;
            [System.Runtime.CompilerServices.FixedBuffer(typeof(int), 16)]
            public CudaSharp.nvcuda.CUDA_RESOURCE_DESC_UNION.<reserved>e__FixedBuffer reserved;
        }
        public struct CUDA_RESOURCE_VIEW_DESC
        {
            public System.UIntPtr depth;
            public uint firstLayer;
            public uint firstMipmapLevel;
            public CudaSharp.nvcuda.CUresourceViewFormat format;
            public System.UIntPtr height;
            public uint lastLayer;
            public uint lastMipmapLevel;
            [System.Runtime.CompilerServices.FixedBuffer(typeof(uint), 16)]
            public CudaSharp.nvcuda.CUDA_RESOURCE_VIEW_DESC.<reserved>e__FixedBuffer reserved;
            public System.UIntPtr width;
        }
        public struct CUDA_TEXTURE_DESC
        {
            public CudaSharp.nvcuda.CUaddress_mode addressMode0;
            public CudaSharp.nvcuda.CUaddress_mode addressMode1;
            public CudaSharp.nvcuda.CUaddress_mode addressMode2;
            public float borderColor0;
            public float borderColor1;
            public float borderColor2;
            public float borderColor3;
            public CudaSharp.nvcuda.CUfilter_mode filterMode;
            public uint flags;
            public uint maxAnisotropy;
            public float maxMipmapLevelClamp;
            public float minMipmapLevelClamp;
            public CudaSharp.nvcuda.CUfilter_mode mipmapFilterMode;
            public float mipmapLevelBias;
            [System.Runtime.CompilerServices.FixedBuffer(typeof(int), 15)]
            public CudaSharp.nvcuda.CUDA_TEXTURE_DESC.<reserved>e__FixedBuffer reserved;
        }
        public enum CUaddress_mode
        {
            CU_TR_ADDRESS_MODE_WRAP = 0,
            CU_TR_ADDRESS_MODE_CLAMP = 1,
            CU_TR_ADDRESS_MODE_MIRROR = 2,
            CU_TR_ADDRESS_MODE_BORDER = 3,
        }
        public readonly struct CUarray : System.IEquatable<CudaSharp.nvcuda.CUarray>
        {
            public CUarray(nint Value) { }
            public System.IntPtr Value { get; init; }
        }
        public enum CUarray_format
        {
            CU_AD_FORMAT_UNSIGNED_INT8 = 1,
            CU_AD_FORMAT_UNSIGNED_INT16 = 2,
            CU_AD_FORMAT_UNSIGNED_INT32 = 3,
            CU_AD_FORMAT_SIGNED_INT8 = 8,
            CU_AD_FORMAT_SIGNED_INT16 = 9,
            CU_AD_FORMAT_SIGNED_INT32 = 10,
            CU_AD_FORMAT_HALF = 16,
            CU_AD_FORMAT_FLOAT = 32,
        }
        public readonly struct CUcontext : System.IEquatable<CudaSharp.nvcuda.CUcontext>
        {
            public CUcontext(nint Value) { }
            public System.IntPtr Value { get; init; }
        }
        public enum CUctx_flags
        {
            CU_CTX_SCHED_AUTO = 0,
            CU_CTX_SCHED_SPIN = 1,
            CU_CTX_SCHED_YIELD = 2,
            CU_CTX_SCHED_BLOCKING_SYNC = 4,
            CU_CTX_MAP_HOST = 8,
            CU_CTX_LMEM_RESIZE_TO_MAX = 16,
        }
        public readonly struct CUdevice : System.IEquatable<CudaSharp.nvcuda.CUdevice>
        {
            public CUdevice(int Value) { }
            public int Value { get; init; }
        }
        public enum CUdevice_attribute
        {
            CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
            CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
            CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
            CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
            CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
            CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
            CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
            CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
            CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
            CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
            CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
            CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
            CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
            CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
            CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
            CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
            CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
            CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
        }
        public readonly struct CUdeviceptr : System.IEquatable<CudaSharp.nvcuda.CUdeviceptr>
        {
            public CUdeviceptr(nint Value) { }
            public System.IntPtr Value { get; init; }
        }
        public struct CUdevprop
        {
            public int SIMDWidth;
            public int clockRate;
            public int maxGridSize0;
            public int maxGridSize1;
            public int maxGridSize2;
            public int maxThreadsDim0;
            public int maxThreadsDim1;
            public int maxThreadsDim2;
            public int maxThreadsPerBlock;
            public int memPitch;
            public int regsPerBlock;
            public int sharedMemPerBlock;
            public int textureAlign;
            public int totalConstantMemory;
        }
        public readonly struct CUevent : System.IEquatable<CudaSharp.nvcuda.CUevent>
        {
            public CUevent(nint Value) { }
            public System.IntPtr Value { get; init; }
        }
        public enum CUevent_flags
        {
            CU_EVENT_DEFAULT = 0,
            CU_EVENT_BLOCKING_SYNC = 1,
            CU_EVENT_DISABLE_TIMING = 2,
            CU_EVENT_INTERPROCESS = 4,
        }
        public readonly struct CUexternalMemory : System.IEquatable<CudaSharp.nvcuda.CUexternalMemory>
        {
            public CUexternalMemory(nint Value) { }
            public System.IntPtr Value { get; init; }
        }
        public enum CUexternalMemoryHandleType
        {
            CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = 1,
            CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 = 2,
            CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3,
            CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP = 4,
            CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE = 5,
            CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE = 6,
            CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT = 7,
            CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF = 8,
        }
        public readonly struct CUexternalSemaphore : System.IEquatable<CudaSharp.nvcuda.CUexternalSemaphore>
        {
            public CUexternalSemaphore(nint Value) { }
            public System.IntPtr Value { get; init; }
        }
        public enum CUexternalSemaphoreHandleType
        {
            CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD = 1,
            CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32 = 2,
            CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3,
            CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE = 4,
            CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE = 5,
            CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC = 6,
            CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX = 7,
            CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT = 8,
            CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD = 9,
            CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32 = 10,
        }
        public enum CUfilter_mode
        {
            CU_TR_FILTER_MODE_POINT = 0,
            CU_TR_FILTER_MODE_LINEAR = 1,
        }
        public enum CUfunc_cache
        {
            CU_FUNC_CACHE_PREFER_NONE = 0,
            CU_FUNC_CACHE_PREFER_SHARED = 1,
            CU_FUNC_CACHE_PREFER_L1 = 2,
            CU_FUNC_CACHE_PREFER_EQUAL = 3,
        }
        public readonly struct CUfunction : System.IEquatable<CudaSharp.nvcuda.CUfunction>
        {
            public CUfunction(nint Value) { }
            public System.IntPtr Value { get; init; }
        }
        public readonly struct CUgraph : System.IEquatable<CudaSharp.nvcuda.CUgraph>
        {
            public CUgraph(nint Value) { }
            public System.IntPtr Value { get; init; }
        }
        public readonly struct CUgraphExec : System.IEquatable<CudaSharp.nvcuda.CUgraphExec>
        {
            public CUgraphExec(nint Value) { }
            public System.IntPtr Value { get; init; }
        }
        public readonly struct CUgraphNode : System.IEquatable<CudaSharp.nvcuda.CUgraphNode>
        {
            public CUgraphNode(nint Value) { }
            public System.IntPtr Value { get; init; }
        }
        public enum CUlimit
        {
            CU_LIMIT_STACK_SIZE = 0,
            CU_LIMIT_PRINTF_FIFO_SIZE = 1,
            CU_LIMIT_MALLOC_HEAP_SIZE = 2,
            CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 3,
            CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 4,
            CU_LIMIT_MAX_L2_FETCH_GRANULARITY = 5,
            CU_LIMIT_PERSISTENT_L2_CACHE_SIZE = 6,
        }
        public enum CUmemhostalloc_flags
        {
            CU_MEMHOSTALLOC_PORTABLE = 1,
            CU_MEMHOSTALLOC_DEVICEMAP = 2,
            CU_MEMHOSTALLOC_WRITECOMBINED = 4,
        }
        public enum CUmemorytype
        {
            CU_MEMORYTYPE_HOST = 1,
            CU_MEMORYTYPE_DEVICE = 2,
            CU_MEMORYTYPE_ARRAY = 3,
            CU_MEMORYTYPE_UNIFIED = 4,
        }
        public readonly struct CUmodule : System.IEquatable<CudaSharp.nvcuda.CUmodule>
        {
            public CUmodule(nint Value) { }
            public System.IntPtr Value { get; init; }
        }
        public enum CUprofiler_outputMode
        {
            CU_OUT_KEY_VALUE_PAIR = 0,
            CU_OUT_CSV = 1,
        }
        public enum CUresourceViewFormat
        {
            CU_RESVIEWFORMAT_NONE = 0,
            CU_RESVIEWFORMAT_UINT_1CHANNEL = 1,
            CU_RESVIEWFORMAT_UINT_2CHANNEL = 2,
            CU_RESVIEWFORMAT_UINT_4CHANNEL = 3,
            CU_RESVIEWFORMAT_SINT_1CHANNEL = 4,
            CU_RESVIEWFORMAT_SINT_2CHANNEL = 5,
            CU_RESVIEWFORMAT_SINT_4CHANNEL = 6,
            CU_RESVIEWFORMAT_FLOAT_1CHANNEL = 7,
            CU_RESVIEWFORMAT_FLOAT_2CHANNEL = 8,
            CU_RESVIEWFORMAT_FLOAT_4CHANNEL = 9,
            CU_RESVIEWFORMAT_HALF_1CHANNEL = 10,
            CU_RESVIEWFORMAT_HALF_2CHANNEL = 11,
            CU_RESVIEWFORMAT_HALF_4CHANNEL = 12,
        }
        public enum CUresourcetype
        {
            CU_RESOURCE_TYPE_ARRAY = 0,
            CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 1,
            CU_RESOURCE_TYPE_LINEAR = 2,
            CU_RESOURCE_TYPE_PITCH2D = 3,
        }
        public enum CUresult
        {
            CUDA_SUCCESS = 0,
            CUDA_ERROR_INVALID_VALUE = 1,
            CUDA_ERROR_OUT_OF_MEMORY = 2,
            CUDA_ERROR_NOT_INITIALIZED = 3,
            CUDA_ERROR_DEINITIALIZED = 4,
            CUDA_ERROR_NO_DEVICE = 100,
            CUDA_ERROR_INVALID_DEVICE = 101,
            CUDA_ERROR_INVALID_CONTEXT = 201,
            CUDA_ERROR_MAP_FAILED = 205,
            CUDA_ERROR_UNMAP_FAILED = 206,
            CUDA_ERROR_NOT_FOUND = 300,
            CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
            CUDA_ERROR_INVALID_IMAGE = 200,
            CUDA_ERROR_LAUNCH_FAILED = 702,
            CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
            CUDA_ERROR_LAUNCH_TIMEOUT = 704,
            CUDA_ERROR_LAUNCH_PARAM_COUNT_MISMATCH = 705,
            CUDA_ERROR_LAUNCH_PARAM_INVALID = 706,
            CUDA_ERROR_LAUNCH_PARAM_NOT_ADDRESSABLE = 707,
            CUDA_ERROR_LAUNCH_PARAM_UNKNOWN = 708,
            CUDA_ERROR_INVALID_DEVICE_FUNCTION = 709,
            CUDA_ERROR_NOT_READY = 600,
            CUDA_ERROR_MODULE_NOT_FOUND = 304,
            CUDA_ERROR_FILE_NOT_FOUND = 301,
            CUDA_ERROR_INVALID_DEVICE_POINTER = 400,
            CUDA_ERROR_INVALID_PITCH_VALUE = 210,
            CUDA_ERROR_INVALID_CUDAARRAY = 211,
            CUDA_ERROR_INVALID_TEXTURE = 212,
            CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 213,
            CUDA_ERROR_INVALID_SOURCE = 302,
            CUDA_ERROR_INVALID_ADDRESS = 401,
        }
        public enum CUsharedconfig
        {
            CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = 0,
            CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = 1,
            CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = 2,
        }
        public readonly struct CUstream : System.IEquatable<CudaSharp.nvcuda.CUstream>
        {
            public CUstream(nint Value) { }
            public System.IntPtr Value { get; init; }
        }
        public struct CUstreamBatchMemOpParams
        {
            public CudaSharp.nvcuda.CUstreamMemOpBarrier_params barrier;
            public CudaSharp.nvcuda.CUstreamFlushRemoteWrites_params flushRemoteWrites;
            public CudaSharp.nvcuda.CUstreamBatchMemOpType operation;
            [System.Runtime.CompilerServices.FixedBuffer(typeof(ulong), 6)]
            public CudaSharp.nvcuda.CUstreamBatchMemOpParams.<pad>e__FixedBuffer pad;
            public CudaSharp.nvcuda.CUstreamWaitValue_params waitValue;
            public CudaSharp.nvcuda.CUstreamWriteValue_params writeValue;
        }
        public enum CUstreamBatchMemOpType
        {
            CU_STREAM_MEM_OP_WAIT_VALUE_32 = 1,
            CU_STREAM_MEM_OP_WRITE_VALUE_32 = 2,
            CU_STREAM_MEM_OP_WAIT_VALUE_64 = 4,
            CU_STREAM_MEM_OP_WRITE_VALUE_64 = 5,
            CU_STREAM_MEM_OP_BARRIER = 6,
            CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = 3,
        }
        public struct CUstreamFlushRemoteWrites_params
        {
            public uint flags;
            public CudaSharp.nvcuda.CUstreamBatchMemOpType operation;
        }
        public struct CUstreamMemOpBarrier_params
        {
            public uint flags;
            public CudaSharp.nvcuda.CUstreamBatchMemOpType operation;
        }
        public enum CUstreamWaitValue_flags
        {
            CU_STREAM_WAIT_VALUE_GEQ = 0,
            CU_STREAM_WAIT_VALUE_EQ = 1,
            CU_STREAM_WAIT_VALUE_AND = 2,
            CU_STREAM_WAIT_VALUE_NOR = 3,
            CU_STREAM_WAIT_VALUE_FLUSH = 1073741824,
        }
        public struct CUstreamWaitValue_params
        {
            public CudaSharp.nvcuda.CUdeviceptr address;
            public CudaSharp.nvcuda.CUdeviceptr alias;
            public uint flags;
            public CudaSharp.nvcuda.CUstreamBatchMemOpType operation;
            public CudaSharp.nvcuda.CUstreamWaitValue_params_union value;
        }
        public struct CUstreamWaitValue_params_union
        {
            public uint value;
            public ulong value64;
        }
        public enum CUstreamWriteValue_flags
        {
            CU_STREAM_WRITE_VALUE_DEFAULT = 0,
            CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER = 1,
        }
        public struct CUstreamWriteValue_params
        {
            public CudaSharp.nvcuda.CUdeviceptr address;
            public CudaSharp.nvcuda.CUdeviceptr alias;
            public uint flags;
            public CudaSharp.nvcuda.CUstreamBatchMemOpType operation;
            public CudaSharp.nvcuda.CUstreamWriteValue_params_union value;
        }
        public struct CUstreamWriteValue_params_union
        {
            public uint value;
            public ulong value64;
        }
        public readonly struct CUsurfObject : System.IEquatable<CudaSharp.nvcuda.CUsurfObject>
        {
            public CUsurfObject(ulong Value) { }
            public ulong Value { get; init; }
        }
        public readonly struct CUsurfref : System.IEquatable<CudaSharp.nvcuda.CUsurfref>
        {
            public CUsurfref(nint Value) { }
            public System.IntPtr Value { get; init; }
        }
        public readonly struct CUtexObject : System.IEquatable<CudaSharp.nvcuda.CUtexObject>
        {
            public CUtexObject(ulong Value) { }
            public ulong Value { get; init; }
        }
        public readonly struct CUtexref : System.IEquatable<CudaSharp.nvcuda.CUtexref>
        {
            public CUtexref(nint Value) { }
            public System.IntPtr Value { get; init; }
        }
        public struct CUuuid
        {
            [System.Runtime.CompilerServices.FixedBuffer(typeof(byte), 16)]
            public CudaSharp.nvcuda.CUuuid.<bytes>e__FixedBuffer bytes;
        }
        extension(CudaSharp.nvcuda.CUresult result)
        {
            public void Ok() { }
            public bool IsOk() { }
            public bool IsError() { }
            public string ToStringFast() { }
        }
    }
    public static class nvrtc
    {
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvrtc", StringMarshalling=System.Runtime.InteropServices.StringMarshalling.Utf8)]
        public static CudaSharp.nvrtc.nvrtcResult nvrtcAddNameExpression(CudaSharp.nvrtc.nvrtcProgram prog, string name_expression) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvrtc", StringMarshalling=System.Runtime.InteropServices.StringMarshalling.Utf8)]
        public static CudaSharp.nvrtc.nvrtcResult nvrtcCompileProgram(CudaSharp.nvrtc.nvrtcProgram prog, int numOptions, in System.ReadOnlySpan<string> options) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvrtc", StringMarshalling=System.Runtime.InteropServices.StringMarshalling.Utf8)]
        public static CudaSharp.nvrtc.nvrtcResult nvrtcCreateProgram(out CudaSharp.nvrtc.nvrtcProgram prog, string src, string name, int numHeaders, in System.ReadOnlySpan<string> headers, in System.ReadOnlySpan<string> includeNames) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvrtc")]
        public static CudaSharp.nvrtc.nvrtcResult nvrtcDestroyProgram(ref CudaSharp.nvrtc.nvrtcProgram prog) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvrtc")]
        public static CudaSharp.nvrtc.nvrtcResult nvrtcGetCUBIN(CudaSharp.nvrtc.nvrtcProgram prog, System.Span<byte> cubin) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvrtc")]
        public static CudaSharp.nvrtc.nvrtcResult nvrtcGetCUBINSize(CudaSharp.nvrtc.nvrtcProgram prog, out System.UIntPtr cubinSizeRet) { }
        [System.Runtime.InteropServices.LibraryImport("nvrtc")]
        public static nint nvrtcGetErrorString(CudaSharp.nvrtc.nvrtcResult result) { }
        public static System.ReadOnlySpan<byte> nvrtcGetErrorStringSpan(CudaSharp.nvrtc.nvrtcResult result) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvrtc")]
        public static CudaSharp.nvrtc.nvrtcResult nvrtcGetLTOIR(CudaSharp.nvrtc.nvrtcProgram prog, System.Span<byte> ltoIR) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvrtc")]
        public static CudaSharp.nvrtc.nvrtcResult nvrtcGetLTOIRSize(CudaSharp.nvrtc.nvrtcProgram prog, out System.UIntPtr ltoIRSizeRet) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvrtc", StringMarshalling=System.Runtime.InteropServices.StringMarshalling.Utf8)]
        public static CudaSharp.nvrtc.nvrtcResult nvrtcGetLoweredName(CudaSharp.nvrtc.nvrtcProgram prog, string name_expression, out System.IntPtr lowered_name) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvrtc")]
        public static CudaSharp.nvrtc.nvrtcResult nvrtcGetNumSupportedArchs(out int numArchs) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvrtc")]
        public static CudaSharp.nvrtc.nvrtcResult nvrtcGetOptiXIR(CudaSharp.nvrtc.nvrtcProgram prog, System.Span<byte> optixIR) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvrtc")]
        public static CudaSharp.nvrtc.nvrtcResult nvrtcGetOptiXIRSize(CudaSharp.nvrtc.nvrtcProgram prog, out System.UIntPtr optixIRSizeRet) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvrtc")]
        public static CudaSharp.nvrtc.nvrtcResult nvrtcGetPTX(CudaSharp.nvrtc.nvrtcProgram prog, System.Span<byte> ptx) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvrtc")]
        public static CudaSharp.nvrtc.nvrtcResult nvrtcGetPTXSize(CudaSharp.nvrtc.nvrtcProgram prog, out System.UIntPtr ptxSizeRet) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvrtc")]
        public static CudaSharp.nvrtc.nvrtcResult nvrtcGetProgramLog(CudaSharp.nvrtc.nvrtcProgram prog, System.Span<byte> log) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvrtc", StringMarshalling=System.Runtime.InteropServices.StringMarshalling.Utf8)]
        public static CudaSharp.nvrtc.nvrtcResult nvrtcGetProgramLogSize(CudaSharp.nvrtc.nvrtcProgram prog, out System.UIntPtr logSizeRet) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvrtc")]
        public static CudaSharp.nvrtc.nvrtcResult nvrtcGetSupportedArchs(System.Span<int> supportedArchs) { }
        [System.Runtime.CompilerServices.SkipLocalsInit]
        [System.Runtime.InteropServices.LibraryImport("nvrtc")]
        public static CudaSharp.nvrtc.nvrtcResult nvrtcVersion(out int major, out int minor) { }
        public readonly struct nvrtcProgram : System.IEquatable<CudaSharp.nvrtc.nvrtcProgram>
        {
            public nvrtcProgram(nint Value) { }
            public System.IntPtr Value { get; init; }
        }
        public enum nvrtcResult
        {
            NVRTC_SUCCESS = 0,
            NVRTC_ERROR_OUT_OF_MEMORY = 1,
            NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
            NVRTC_ERROR_INVALID_INPUT = 3,
            NVRTC_ERROR_INVALID_PROGRAM = 4,
            NVRTC_ERROR_INVALID_OPTION = 5,
            NVRTC_ERROR_COMPILATION = 6,
            NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
            NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
            NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
            NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
            NVRTC_ERROR_INTERNAL_ERROR = 11,
            NVRTC_ERROR_TIME_FILE_WRITE_FAILED = 12,
        }
        extension(CudaSharp.nvrtc.nvrtcResult result)
        {
        }
    }
}
```
