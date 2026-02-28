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

⭐ Please star this project if you like it. ⭐

[Example](#example) | [Example Catalogue](#example-catalogue) | [Public API Reference](#public-api-reference)

## Example
```csharp
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
    public static class nvcuda
    {
        public static void Empty() { }
    }
}
```
