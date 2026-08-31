using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Runtime.InteropServices;
using System.Text;
using static CudaSharp.nvcuda;

namespace CudaSharp.Test;

[TestClass]
public class nvcudaTest
{
    static readonly byte[] EmptyKernelPtx =
    [
        .. Encoding.UTF8.GetBytes("""
            .version 6.0
            .target sm_50
            .address_size 64

            .visible .entry empty()
            {
                ret;
            }
            """),
        0,
    ];

    public nvcudaTest()
    {
        try
        {
            cuInit().Ok();
        }
        catch (Exception ex)
        {
#pragma warning disable MSTEST0058 // Do not use asserts in catch blocks
            Assert.Inconclusive($"CUDA initialization failed: {ex.Message}");
#pragma warning restore MSTEST0058 // Do not use asserts in catch blocks
        }
    }

    [TestMethod]
    public void nvcudaTest_cuInit()
    {
        cuInit().Ok();
    }

    [TestMethod]
    public void nvcudaTest_CUresult_ToStringFast()
    {
        Assert.EnumValuesToString<CUresult>(r => r.ToStringFast());
        var unknown = (CUresult)(int.MaxValue - 1);
        Assert.AreEqual("CUDA_ERROR_UNKNOWN:2147483646", unknown.ToStringFast());
    }
    [TestMethod]
    public void nvcudaTest_CUresult_Ok()
    {
        Assert.EnumValuesOkThrows<CUresult>(r => r == CUresult.CUDA_SUCCESS, r => r.Ok());
    }
    [TestMethod]
    public void nvcudaTest_CUresult_IsOk()
    {
        Assert.IsTrue(CUresult.CUDA_SUCCESS.IsOk());
        Assert.IsFalse(CUresult.CUDA_ERROR_DEINITIALIZED.IsOk());
        Assert.IsFalse(CUresult.CUDA_ERROR_NOT_READY.IsOk());
    }
    [TestMethod]
    public void nvcudaTest_CUresult_IsError()
    {
        Assert.IsFalse(CUresult.CUDA_SUCCESS.IsError());
        Assert.IsTrue(CUresult.CUDA_ERROR_DEINITIALIZED.IsError());
        Assert.IsTrue(CUresult.CUDA_ERROR_NOT_READY.IsError());
    }

    [TestMethod]
    public void nvcudaTest_DriverAndErrorHandling()
    {
        cuDriverGetVersion(out var driverVersion).Ok();
        Assert.IsGreaterThan(0, driverVersion);

        cuGetErrorName(CUresult.CUDA_SUCCESS, out var name).Ok();
        cuGetErrorString(CUresult.CUDA_SUCCESS, out var description).Ok();

        Assert.AreEqual(nameof(CUresult.CUDA_SUCCESS), Marshal.PtrToStringUTF8(name));
        Assert.IsFalse(string.IsNullOrWhiteSpace(Marshal.PtrToStringUTF8(description)));
    }

    [TestMethod]
    public void nvcudaTest_DeviceManagement()
    {
        cuDeviceGetCount(out var count).Ok();
        Assert.IsGreaterThan(0, count);

        cuDeviceGet(out var device, 0).Ok();
        var name = new byte[256];
        cuDeviceGetName(name, name.Length, device).Ok();
        cuDeviceTotalMem(out var totalMemory, device).Ok();
        cuDeviceComputeCapability(out var major, out var minor, device).Ok();
        cuDeviceGetAttribute(out var multiprocessorCount,
            CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device).Ok();

        Assert.IsFalse(string.IsNullOrWhiteSpace(Encoding.UTF8.GetString(name).TrimEnd('\0')));
        Assert.IsGreaterThan((nuint)0, totalMemory);
        Assert.IsGreaterThan(0, major);
        Assert.IsGreaterThanOrEqualTo(0, minor);
        Assert.IsGreaterThan(0, multiprocessorCount);
    }

    [TestMethod]
    public void nvcudaTest_PrimaryContextAndContextManagement()
    {
        WithPrimaryContext((device, context) =>
        {
            cuCtxGetCurrent(out var current).Ok();
            cuCtxGetDevice(out var contextDevice).Ok();
            cuCtxGetApiVersion(context, out var apiVersion).Ok();
            cuCtxGetFlags(out _).Ok();
            cuCtxGetStreamPriorityRange(out var leastPriority, out var greatestPriority).Ok();
            cuCtxSynchronize().Ok();

            Assert.AreEqual(context, current);
            Assert.AreEqual(device, contextDevice);
            Assert.IsGreaterThan((uint)0, apiVersion);
            Assert.IsLessThanOrEqualTo(leastPriority, greatestPriority);
        });
    }

    [TestMethod]
    public unsafe void nvcudaTest_ModuleExecutionOccupancyAndLinker()
    {
        WithPrimaryContext((_, _) =>
        {
            cuModuleLoadData(out var module, EmptyKernelPtx).Ok();
            try
            {
                cuModuleGetFunction(out var function, module, "empty").Ok();
                cuFuncGetAttribute(out var maxThreads,
                    CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function).Ok();
                cuOccupancyMaxActiveBlocksPerMultiprocessor(
                    out var activeBlocks, function, 1, 0).Ok();
                cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, default, null, null).Ok();
                cuCtxSynchronize().Ok();

                Assert.IsGreaterThan(0, maxThreads);
                Assert.IsGreaterThan(0, activeBlocks);
            }
            finally
            {
                cuModuleUnload(module).Ok();
            }

            cuLinkCreate(0, null, null, out var linkState).Ok();
            cuLinkDestroy(linkState).Ok();
        });
    }

    [TestMethod]
    public void nvcudaTest_ArrayTextureAndSurfaceManagement()
    {
        WithPrimaryContext((_, _) =>
        {
            var descriptor = new CUDA_ARRAY_DESCRIPTOR
            {
                Width = 4,
                Height = 4,
                Format = CUarray_format.CU_AD_FORMAT_UNSIGNED_INT8,
                NumChannels = 1,
            };

            cuArrayCreate(out var array, in descriptor).Ok();
            try
            {
                cuArrayGetDescriptor(out var actual, array).Ok();
                Assert.AreEqual(descriptor.Width, actual.Width);
                Assert.AreEqual(descriptor.Height, actual.Height);
                Assert.AreEqual(descriptor.Format, actual.Format);
                Assert.AreEqual(descriptor.NumChannels, actual.NumChannels);
            }
            finally
            {
                cuArrayDestroy(array).Ok();
            }

            cuTexRefCreate(out var textureReference).Ok();
            cuTexRefDestroy(textureReference).Ok();

            const uint array3DSurfaceLoadStore = 0x02;
            var surfaceDescriptor = new CUDA_ARRAY3D_DESCRIPTOR
            {
                Width = descriptor.Width,
                Height = descriptor.Height,
                Depth = 0,
                Format = descriptor.Format,
                NumChannels = descriptor.NumChannels,
                Flags = array3DSurfaceLoadStore,
            };
            cuArray3DCreate(out array, in surfaceDescriptor).Ok();
            try
            {
                var resourceDescriptor = new CUDA_RESOURCE_DESC
                {
                    resType = CUresourcetype.CU_RESOURCE_TYPE_ARRAY,
                    res = new CUDA_RESOURCE_DESC_UNION
                    {
                        array = new CUDA_RESOURCE_DESC_ARRAY { hArray = array },
                    },
                };
                cuSurfObjectCreate(out var surfaceObject, in resourceDescriptor).Ok();
                try
                {
                    cuSurfObjectGetResourceDesc(out var actualResource, surfaceObject).Ok();
                    Assert.AreEqual(CUresourcetype.CU_RESOURCE_TYPE_ARRAY, actualResource.resType);
                    Assert.AreEqual(array, actualResource.res.array.hArray);
                }
                finally
                {
                    cuSurfObjectDestroy(surfaceObject).Ok();
                }
            }
            finally
            {
                cuArrayDestroy(array).Ok();
            }
        });
    }

    [TestMethod]
    public unsafe void nvcudaTest_MemoryManagement()
    {
        WithPrimaryContext((_, _) =>
        {
            const int elementCount = 16;
            const uint value = 0x12345678;
            cuMemAlloc_v2(out var devicePointer, elementCount * sizeof(uint)).Ok();
            try
            {
                cuMemsetD32_v2(devicePointer, value, elementCount).Ok();
                var host = stackalloc uint[elementCount];
                cuMemcpyDtoH_v2((IntPtr)host, devicePointer, elementCount * sizeof(uint)).Ok();

                for (var i = 0; i < elementCount; i++)
                    Assert.AreEqual(value, host[i]);
            }
            finally
            {
                cuMemFree_v2(devicePointer).Ok();
            }
        });
    }

    [TestMethod]
    public void nvcudaTest_StreamEventAndGraphManagement()
    {
        WithPrimaryContext((_device, context) =>
        {
            cuStreamCreate(out var stream, 0).Ok();
            try
            {
                cuStreamGetCtx(stream, out var streamContext).Ok();
                cuStreamGetFlags(stream, out var streamFlags).Ok();
                cuStreamGetPriority(stream, out _).Ok();
                Assert.AreEqual(context, streamContext);
                Assert.AreEqual((uint)0, streamFlags);

                cuEventCreate(out var start, 0).Ok();
                cuEventCreate(out var end, 0).Ok();
                try
                {
                    cuEventRecord(start, stream).Ok();
                    cuEventRecord(end, stream).Ok();
                    cuEventSynchronize(end).Ok();
                    cuEventQuery(end).Ok();
                    cuEventElapsedTime(out var milliseconds, start, end).Ok();
                    Assert.IsGreaterThanOrEqualTo(0, milliseconds);
                }
                finally
                {
                    cuEventDestroy(end).Ok();
                    cuEventDestroy(start).Ok();
                }

                cuStreamBeginCapture(stream, CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL).Ok();
                cuStreamEndCapture(stream, out var capturedGraph).Ok();
                cuGraphDestroy(capturedGraph).Ok();
                cuStreamSynchronize(stream).Ok();
            }
            finally
            {
                cuStreamDestroy(stream).Ok();
            }

            cuGraphCreate(out var graph, 0).Ok();
            cuGraphDestroy(graph).Ok();
        });
    }

    static void WithPrimaryContext(Action<CUdevice, CUcontext> action)
    {
        cuDeviceGet(out var device, 0).Ok();
        cuCtxGetCurrent(out var previousContext).Ok();
        cuDevicePrimaryCtxRetain(out var context, device).Ok();
        try
        {
            cuCtxSetCurrent(context).Ok();
            action(device, context);
        }
        finally
        {
            cuCtxSetCurrent(previousContext).Ok();
            cuDevicePrimaryCtxRelease(device).Ok();
        }
    }
}
