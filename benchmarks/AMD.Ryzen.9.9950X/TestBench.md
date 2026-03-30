```

BenchmarkDotNet v0.15.8, Windows 10 (10.0.19045.7058/22H2/2022Update)
AMD Ryzen 9 9950X 4.30GHz, 1 CPU, 32 logical and 16 physical cores
.NET SDK 10.0.104
  [Host]    : .NET 10.0.5 (10.0.5, 10.0.526.15411), X64 RyuJIT x86-64-v4
  .NET 10.0 : .NET 10.0.5 (10.0.5, 10.0.526.15411), X64 RyuJIT x86-64-v4

Job=.NET 10.0  EnvironmentVariables=DOTNET_GCDynamicAdaptationMode=0  Runtime=.NET 10.0  
Toolchain=net10.0  InvocationCount=Default  IterationTime=350ms  
MaxIterationCount=15  MinIterationCount=5  WarmupCount=6  
Reader=String  

```
| Method                      | Scope | Count | Mean               | Ratio         | Allocated | Alloc Ratio |
|---------------------------- |------ |------ |-------------------:|--------------:|----------:|------------:|
| CudaSharp_cuInit            | Test  | 25000 |         27.2133 ns |         1.000 |         - |          NA |
| CudaSharp_CuInit_EnsureInit | Test  | 25000 |          0.0097 ns |         0.000 |         - |          NA |
| CudaSharp_CtxCreateDestroy  | Test  | 25000 | 50,797,173.3333 ns | 1,866,632.010 |         - |          NA |
