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
| Method                                           | SerialLaunchCount | Mean      | Ratio | Allocated | Alloc Ratio |
|------------------------------------------------- |------------------ |----------:|------:|----------:|------------:|
| cuLaunchKernel_Raw_SerialTripleBuffer_StreamSync | 1                 |  25.62 μs |  1.00 |         - |          NA |
| cuGraphLaunch_SerialTripleBuffer_StreamSync      | 1                 |  25.04 μs |  0.98 |         - |          NA |
|                                                  |                   |           |       |           |             |
| cuLaunchKernel_Raw_SerialTripleBuffer_StreamSync | 2                 |  26.43 μs |  1.00 |         - |          NA |
| cuGraphLaunch_SerialTripleBuffer_StreamSync      | 2                 |  25.13 μs |  0.95 |         - |          NA |
|                                                  |                   |           |       |           |             |
| cuLaunchKernel_Raw_SerialTripleBuffer_StreamSync | 4                 |  27.07 μs |  1.00 |         - |          NA |
| cuGraphLaunch_SerialTripleBuffer_StreamSync      | 4                 |  25.28 μs |  0.93 |         - |          NA |
|                                                  |                   |           |       |           |             |
| cuLaunchKernel_Raw_SerialTripleBuffer_StreamSync | 8                 |  40.21 μs |  1.00 |         - |          NA |
| cuGraphLaunch_SerialTripleBuffer_StreamSync      | 8                 |  25.87 μs |  0.64 |         - |          NA |
|                                                  |                   |           |       |           |             |
| cuLaunchKernel_Raw_SerialTripleBuffer_StreamSync | 16                |  66.81 μs |  1.00 |         - |          NA |
| cuGraphLaunch_SerialTripleBuffer_StreamSync      | 16                |  39.51 μs |  0.59 |         - |          NA |
|                                                  |                   |           |       |           |             |
| cuLaunchKernel_Raw_SerialTripleBuffer_StreamSync | 32                | 118.91 μs |  1.00 |         - |          NA |
| cuGraphLaunch_SerialTripleBuffer_StreamSync      | 32                |  54.49 μs |  0.46 |         - |          NA |
|                                                  |                   |           |       |           |             |
| cuLaunchKernel_Raw_SerialTripleBuffer_StreamSync | 64                | 222.55 μs |  1.00 |         - |          NA |
| cuGraphLaunch_SerialTripleBuffer_StreamSync      | 64                |  84.56 μs |  0.38 |         - |          NA |
|                                                  |                   |           |       |           |             |
| cuLaunchKernel_Raw_SerialTripleBuffer_StreamSync | 128               | 430.40 μs |  1.00 |         - |          NA |
| cuGraphLaunch_SerialTripleBuffer_StreamSync      | 128               | 144.96 μs |  0.34 |         - |          NA |
|                                                  |                   |           |       |           |             |
| cuLaunchKernel_Raw_SerialTripleBuffer_StreamSync | 256               | 834.78 μs |  1.00 |         - |          NA |
| cuGraphLaunch_SerialTripleBuffer_StreamSync      | 256               | 260.24 μs |  0.31 |         - |          NA |
