```

BenchmarkDotNet v0.15.8, Windows 10 (10.0.19045.7058/22H2/2022Update)
AMD Ryzen 9 9950X 4.30GHz, 1 CPU, 32 logical and 16 physical cores
.NET SDK 10.0.104
  [Host]    : .NET 10.0.5 (10.0.5, 10.0.526.15411), X64 RyuJIT x86-64-v4
  .NET 10.0 : .NET 10.0.5 (10.0.5, 10.0.526.15411), X64 RyuJIT x86-64-v4

Job=.NET 10.0  EnvironmentVariables=DOTNET_GCDynamicAdaptationMode=0  Runtime=.NET 10.0  
Toolchain=net10.0  IterationTime=350ms  MaxIterationCount=15  
MinIterationCount=5  WarmupCount=6  

```
| Method                                                         | SerialLaunchCount | Mean     | Ratio | Allocated | Alloc Ratio |
|--------------------------------------------------------------- |------------------ |---------:|------:|----------:|------------:|
| cuLaunchKernel_Raw_SerialTripleBuffer_StreamSync               | 256               | 820.9 μs |  1.00 |         - |          NA |
| cuGraphLaunch_SerialTripleBuffer_StreamSync                    | 256               | 257.8 μs |  0.31 |         - |          NA |
| cuGraphLaunch_DeviceLaunchCapableSerialTripleBuffer_StreamSync | 256               | 216.3 μs |  0.26 |         - |          NA |
| cuGraphLaunch_CapturedSerialTripleBuffer_StreamSync            | 256               | 256.5 μs |  0.31 |         - |          NA |
