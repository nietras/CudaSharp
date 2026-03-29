using System.Collections.Generic;
using System.Linq;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using static CudaSharp.nvcuda;

namespace CudaSharp.ComparisonBenchmarks;

[HideColumns("InvocationCount")]
[GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByCategory, BenchmarkLogicalGroupRule.ByParams)]
public abstract class BaseBench
{
    const int FloatsCount = 20;
    readonly ReaderSpec[] _readers;

    protected BaseBench(string scope, int count)
    {
        Scope = scope;
        Count = count;
        _readers =
        [
            ReaderSpec.FromString("String", new(() => "TEST")),
            //ReaderSpec.FromBytes("Stream", new(() => T.GenerateBytes(Rows, FloatsCount))),
        ];
        Reader = _readers.First();
    }

    [ParamsSource(nameof(ScopeParams))] // Attributes for params is challenging 👇
    public string Scope { get; set; }
    public IEnumerable<string> ScopeParams() => [Scope];

    [ParamsSource(nameof(Readers))]
    public ReaderSpec Reader { get; set; }
    public IEnumerable<ReaderSpec> Readers() => _readers;

    [ParamsSource(nameof(CountParams))] // Attributes for params is challenging 👇
    public int Count { get; set; }
    public IEnumerable<int> CountParams() => [Count];
}

[BenchmarkCategory("0")]
public class TestBench : BaseBench
{
#if DEBUG
    const int DefaultCount = 10_000;
#else
    const int DefaultCount = 25_000;
#endif

    public TestBench() : base("Test", DefaultCount) { CuInit.EnsureInit(); }

    [Benchmark(Baseline = true)]
    public void CudaSharp_cuInit()
    {
        cuInit();
    }
    [Benchmark]
    public void CudaSharp_CuInit_EnsureInit()
    {
        CuInit.EnsureInit();
    }
    [Benchmark]
    public void CudaSharp_CtxCreateDestroy()
    {
        cuDeviceGet(out var device, 0).Ok();
        cuCtxCreate(out var context, CUctx_flags.CU_CTX_SCHED_AUTO, device).Ok();
        cuCtxDestroy(context);
    }
}
