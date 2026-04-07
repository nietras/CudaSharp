using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading;

namespace CudaSharp;

public static class DllResolver
{
    static bool _registered = false;
    static readonly Lock _lock = new();

    public static void Register()
    {
        lock (_lock)
        {
            if (_registered) return;
            try
            {
                NativeLibrary.SetDllImportResolver(Assembly.GetExecutingAssembly(), OnDllImport);
            }
            catch (InvalidOperationException)
            {
                // Already set, ignore
            }
            _registered = true;
        }
    }

    static IntPtr OnDllImport(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName == "nvcuda")
        {
            var cudaPath = Environment.GetEnvironmentVariable("CUDA_PATH");
            if (!string.IsNullOrEmpty(cudaPath))
            {
                var dllPath = Path.Combine(cudaPath, "bin", "nvcuda.dll");
                if (NativeLibrary.TryLoad(dllPath, out var handle))
                {
                    return handle;
                }
            }

            var defaultPath = @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA";
            if (Directory.Exists(defaultPath))
            {
                var versions = Directory.GetDirectories(defaultPath, "v*.*")
                                        .Select(Path.GetFileName)
                                        .Where(v => v is not null)
                                        .OrderByDescending(v => v)
                                        .ToList();
                if (versions.Any())
                {
                    var latestVersion = versions.First();
                    var dllPath = Path.Combine(defaultPath, latestVersion!, "bin", "nvcuda.dll");
                    if (NativeLibrary.TryLoad(dllPath, out var handle))
                    {
                        return handle;
                    }
                }
            }
        }
        else if (libraryName == "nvrtc")
        {
            var cudaPath = Environment.GetEnvironmentVariable("CUDA_PATH");
            if (!string.IsNullOrEmpty(cudaPath))
            {
                foreach (var binPath in GetCudaNvrtcSearchPaths(cudaPath))
                {
                    if (!Directory.Exists(binPath))
                    {
                        continue;
                    }

                    var dll = Directory.GetFiles(binPath, "nvrtc64_*.dll")
                                       .OrderByDescending(f => f)
                                       .FirstOrDefault();
                    if (dll == null)
                    {
                        continue;
                    }

                    var builtins = Directory.GetFiles(binPath, "nvrtc-builtins64_*.dll")
                                            .OrderByDescending(f => f)
                                            .FirstOrDefault();
                    if (builtins != null)
                    {
                        NativeLibrary.TryLoad(builtins, out _);
                    }

                    if (NativeLibrary.TryLoad(dll, out var handle))
                    {
                        return handle;
                    }
                }
            }

            // Fallback to searching default paths if CUDA_PATH not set
            var defaultPath = @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA";
            if (Directory.Exists(defaultPath))
            {
                var versions = Directory.GetDirectories(defaultPath, "v*.*")
                                        .Select(Path.GetFileName)
                                        .Where(v => v is not null)
                                        .OrderByDescending(v => v)
                                        .ToList();
                foreach (var version in versions)
                {
                    foreach (var binPath in GetCudaNvrtcSearchPaths(Path.Combine(defaultPath, version!)))
                    {
                        if (!Directory.Exists(binPath))
                        {
                            continue;
                        }

                        var dll = Directory.GetFiles(binPath, "nvrtc64_*.dll")
                                           .OrderByDescending(f => f)
                                           .FirstOrDefault();
                        if (dll == null)
                        {
                            continue;
                        }

                        var builtins = Directory.GetFiles(binPath, "nvrtc-builtins64_*.dll")
                                                .OrderByDescending(f => f)
                                                .FirstOrDefault();
                        if (builtins != null)
                        {
                            NativeLibrary.TryLoad(builtins, out _);
                        }

                        if (NativeLibrary.TryLoad(dll, out var handle))
                        {
                            return handle;
                        }
                    }
                }
            }
        }
        else if (libraryName == "nvJitLink")
        {
            var cudaPath = Environment.GetEnvironmentVariable("CUDA_PATH");
            if (!string.IsNullOrEmpty(cudaPath))
            {
                var binPath = Path.Combine(cudaPath, "bin", "x64");
                if (Directory.Exists(binPath))
                {
                    var dll = Directory.GetFiles(binPath, "nvJitLink_*.dll")
                                       .OrderByDescending(f => f)
                                       .FirstOrDefault();
                    if (dll != null && NativeLibrary.TryLoad(dll, out var handle))
                    {
                        return handle;
                    }
                }
            }

            var defaultPath = @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA";
            if (Directory.Exists(defaultPath))
            {
                var versions = Directory.GetDirectories(defaultPath, "v*.*")
                                        .Select(Path.GetFileName)
                                        .Where(v => v is not null)
                                        .OrderByDescending(v => v)
                                        .ToList();
                foreach (var version in versions)
                {
                    var binPath = Path.Combine(defaultPath, version!, "bin", "x64");
                    if (!Directory.Exists(binPath))
                    {
                        continue;
                    }

                    var dll = Directory.GetFiles(binPath, "nvJitLink_*.dll")
                                       .OrderByDescending(f => f)
                                       .FirstOrDefault();
                    if (dll != null && NativeLibrary.TryLoad(dll, out var handle))
                    {
                        return handle;
                    }
                }
            }
        }
        return IntPtr.Zero;
    }

    static IEnumerable<string> GetCudaNvrtcSearchPaths(string cudaRoot)
    {
        yield return Path.Combine(cudaRoot, "bin", "x64");
        yield return Path.Combine(cudaRoot, "bin");
    }
}
