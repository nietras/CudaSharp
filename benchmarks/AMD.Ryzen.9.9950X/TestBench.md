```

BenchmarkDotNet v0.15.8, Windows 10 (10.0.19045.6937/22H2/2022Update)
AMD Ryzen 9 9950X 4.30GHz, 1 CPU, 32 logical and 16 physical cores
.NET SDK 10.0.103
  [Host]    : .NET 10.0.3 (10.0.3, 10.0.326.7603), X64 RyuJIT x86-64-v4
  .NET 10.0 : .NET 10.0.3 (10.0.3, 10.0.326.7603), X64 RyuJIT x86-64-v4

Job=.NET 10.0  EnvironmentVariables=DOTNET_GCDynamicAdaptationMode=0  Runtime=.NET 10.0  
Toolchain=net10.0  InvocationCount=Default  IterationTime=350ms  
MaxIterationCount=15  MinIterationCount=5  WarmupCount=6  
Reader=String  Error=0.0024 ns  StdDev=0.0006 ns  
Median=0.0 ns  RatioSD=?  

```
| Method          | Scope | Count | Mean      | Ratio | Allocated | Alloc Ratio |
|---------------- |------ |------ |----------:|------:|----------:|------------:|
| CudaSharp______ | Test  | 25000 | 0.0004 ns |     ? |         - |           ? |
