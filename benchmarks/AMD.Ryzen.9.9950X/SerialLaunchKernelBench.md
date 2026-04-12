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
| Method                                                          | SerialLaunchCount | Mean            | Ratio | Allocated | Alloc Ratio |
|---------------------------------------------------------------- |------------------ |----------------:|------:|----------:|------------:|
| cuLaunchKernel_Raw_SerialTripleBuffer_StreamSync                | 256               | 818,683.7740 ns | 1.000 |         - |          NA |
| cuGraphLaunch_SerialTripleBuffer_StreamSync                     | 256               | 259,743.3750 ns | 0.317 |         - |          NA |
| cuGraphLaunch_DeviceLaunchCapableSerialTripleBuffer_StreamSync  | 256               | 215,228.6757 ns | 0.263 |         - |          NA |
| cuGraphLaunch_TrueDeviceTailLaunchSerialTripleBuffer_StreamSync | 256               |       0.1799 ns | 0.000 |         - |          NA |
| cuGraphLaunch_CapturedSerialTripleBuffer_StreamSync             | 256               | 258,311.2403 ns | 0.316 |         - |          NA |
