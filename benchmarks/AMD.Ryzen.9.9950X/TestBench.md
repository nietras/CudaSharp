```

BenchmarkDotNet v0.15.8, Windows 10 (10.0.19045.7417/22H2/2022Update)
AMD Ryzen 9 9950X 4.30GHz, 1 CPU, 32 logical and 16 physical cores
.NET SDK 10.0.109
  [Host]    : .NET 10.0.9 (10.0.9, 10.0.926.27113), X64 RyuJIT x86-64-v4
  .NET 10.0 : .NET 10.0.9 (10.0.9, 10.0.926.27113), X64 RyuJIT x86-64-v4

Job=.NET 10.0  EnvironmentVariables=DOTNET_GCDynamicAdaptationMode=0  Runtime=.NET 10.0  
Toolchain=net10.0  InvocationCount=Default  IterationTime=350ms  
MaxIterationCount=15  MinIterationCount=5  WarmupCount=6  
Reader=String  

```
| Method                      | Scope | Count | Mean               | Ratio         | Allocated | Alloc Ratio |
|---------------------------- |------ |------ |-------------------:|--------------:|----------:|------------:|
| CudaSharp_cuInit            | Test  | 25000 |         27.8944 ns |         1.000 |         - |          NA |
| CudaSharp_CuInit_EnsureInit | Test  | 25000 |          0.0216 ns |         0.001 |         - |          NA |
| CudaSharp_CtxCreateDestroy  | Test  | 25000 | 57,439,150.0000 ns | 2,059,164.388 |         - |          NA |
