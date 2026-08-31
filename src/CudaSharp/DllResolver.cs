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
        var nativePathsData = AppContext.GetData("NATIVE_DLL_SEARCH_DIRECTORIES") as string ?? string.Empty;
        // Paths are separated by ';' on Windows and ':' on Unix/Linux/macOS
        var nativeDirectories = nativePathsData.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries);

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

            string[] defaultPaths = [@"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"];
            foreach (var defaultPath in defaultPaths)
            {
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

            if (NativeLibrary.TryLoad("nvcuda.dll", out var fallbackHandle))
            {
                return fallbackHandle;
            }
        }
        else if (libraryName == "nvrtc")
        {
            int maxCudaVersion = GetDriverVersion(assembly, searchPath);
            var cudaPath = Environment.GetEnvironmentVariable("CUDA_PATH") ?? string.Empty;
            {
                var directories = GetCudaNvrtcSearchPaths(cudaPath).Concat(nativeDirectories).ToArray();
                foreach (var binPath in directories)
                {
                    if (!Directory.Exists(binPath))
                    {
                        continue;
                    }

                    var dlls = Directory.GetFiles(binPath, "nvrtc64_*.dll")
                                       .OrderByDescending(f => f);
                    foreach (var dll in dlls)
                    {
                        var builtins = Directory.GetFiles(binPath, "nvrtc-builtins64_*.dll")
                                                .OrderByDescending(f => f)
                                                .FirstOrDefault();
                        if (builtins != null)
                        {
                            NativeLibrary.TryLoad(builtins, out _);
                        }

                        var jitLink = Directory.GetFiles(binPath, "nvJitLink*.dll")
                                               .OrderByDescending(f => f)
                                               .FirstOrDefault();
                        if (jitLink != null)
                        {
                            NativeLibrary.TryLoad(jitLink, out _);
                        }

                        if (NativeLibrary.TryLoad(dll, out var handle))
                        {
                            //if (IsNvrtcCompatible(handle, maxCudaVersion))
                            {
                                return handle;
                            }
                            //NativeLibrary.Free(handle);
                        }
                    }
                }
            }

            // Fallback to searching default paths if CUDA_PATH not set
            string[] defaultPaths = [@"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"];
            foreach (var defaultPath in defaultPaths)
            {
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

                            var dlls = Directory.GetFiles(binPath, "nvrtc64_*.dll")
                                               .OrderByDescending(f => f);
                            foreach (var dll in dlls)
                            {
                                var builtins = Directory.GetFiles(binPath, "nvrtc-builtins64_*.dll")
                                                        .OrderByDescending(f => f)
                                                        .FirstOrDefault();
                                if (builtins != null)
                                {
                                    NativeLibrary.TryLoad(builtins, out _);
                                }

                                var jitLink = Directory.GetFiles(binPath, "nvJitLink*.dll")
                                                       .OrderByDescending(f => f)
                                                       .FirstOrDefault();
                                if (jitLink != null)
                                {
                                    NativeLibrary.TryLoad(jitLink, out _);
                                }

                                if (NativeLibrary.TryLoad(dll, out var handle))
                                {
                                    if (IsNvrtcCompatible(handle, maxCudaVersion))
                                    {
                                        return handle;
                                    }
                                    NativeLibrary.Free(handle);
                                }
                            }
                        }
                    }
                }
            }

            // Final fallback to searching .nuget packages in user profile
            var userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            var nugetPath = Path.Combine(userProfile, ".nuget", "packages");
            if (Directory.Exists(nugetPath))
            {
                var dlls = Directory.GetFiles(nugetPath, "nvrtc64_*.dll", SearchOption.AllDirectories)
                                   .OrderByDescending(Path.GetFileName);
                foreach (var dll in dlls)
                {
                    var binDir = Path.GetDirectoryName(dll);
                    var builtins = Directory.GetFiles(binDir!, "nvrtc-builtins64_*.dll")
                                            .OrderByDescending(f => f)
                                            .FirstOrDefault();
                    if (builtins != null)
                    {
                        NativeLibrary.TryLoad(builtins, out _);
                    }

                    var jitLink = Directory.GetFiles(binDir!, "nvJitLink*.dll")
                                           .OrderByDescending(f => f)
                                           .FirstOrDefault();
                    if (jitLink != null)
                    {
                        NativeLibrary.TryLoad(jitLink, out _);
                    }

                    if (NativeLibrary.TryLoad(dll, out var handle))
                    {
                        if (IsNvrtcCompatible(handle, maxCudaVersion))
                        {
                            return handle;
                        }
                        NativeLibrary.Free(handle);
                    }
                }
            }
        }
        else if (libraryName == "nvJitLink")
        {
            foreach (var nativeDirectory in nativeDirectories)
            {
                if (!Directory.Exists(nativeDirectory))
                {
                    continue;
                }

                var dll = Directory.GetFiles(nativeDirectory, "nvJitLink*.dll", SearchOption.AllDirectories)
                                   .OrderByDescending(Path.GetFileName)
                                   .FirstOrDefault();
                if (dll != null && NativeLibrary.TryLoad(dll, out var handle))
                {
                    return handle;
                }
            }

            var cudaPath = Environment.GetEnvironmentVariable("CUDA_PATH");
            if (!string.IsNullOrEmpty(cudaPath))
            {
                foreach (var binPath in GetCudaNvrtcSearchPaths(cudaPath))
                {
                    if (Directory.Exists(binPath))
                    {
                        var dll = Directory.GetFiles(binPath, "nvJitLink*.dll")
                                           .OrderByDescending(f => f)
                                           .FirstOrDefault();
                        if (dll != null && NativeLibrary.TryLoad(dll, out var handle))
                        {
                            return handle;
                        }
                    }
                }
            }

            string[] defaultPaths = [@"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"];
            foreach (var defaultPath in defaultPaths)
            {
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

                            var dll = Directory.GetFiles(binPath, "nvJitLink*.dll")
                                               .OrderByDescending(f => f)
                                               .FirstOrDefault();
                            if (dll != null && NativeLibrary.TryLoad(dll, out var handle))
                            {
                                return handle;
                            }
                        }
                    }
                }
            }

            // Final fallback to searching .nuget packages in user profile
            var userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            var nugetPath = Path.Combine(userProfile, ".nuget", "packages");
            if (Directory.Exists(nugetPath))
            {
                var dll = Directory.GetFiles(nugetPath, "nvJitLink*.dll", SearchOption.AllDirectories)
                                   .OrderByDescending(Path.GetFileName)
                                   .FirstOrDefault();
                if (dll != null && NativeLibrary.TryLoad(dll, out var handle))
                {
                    return handle;
                }
            }
        }
        else if (libraryName == "cudart")
        {
            var cudaPath = Environment.GetEnvironmentVariable("CUDA_PATH");
            if (!string.IsNullOrEmpty(cudaPath))
            {
                foreach (var probePath in new[] { Path.Combine(cudaPath, "bin", "x64"), Path.Combine(cudaPath, "bin") })
                {
                    if (Directory.Exists(probePath))
                    {
                        var dll = Directory.GetFiles(probePath, "cudart64_*.dll")
                                           .OrderByDescending(f => f)
                                           .FirstOrDefault();
                        if (dll != null && NativeLibrary.TryLoad(dll, out var handle))
                        {
                            return handle;
                        }
                    }
                }
            }

            string[] defaultPaths = [@"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"];
            foreach (var defaultPath in defaultPaths)
            {
                if (Directory.Exists(defaultPath))
                {
                    var versions = Directory.GetDirectories(defaultPath, "v*.*")
                                            .Select(Path.GetFileName)
                                            .Where(v => v is not null)
                                            .OrderByDescending(v => v)
                                            .ToList();
                    foreach (var version in versions)
                    {
                        foreach (var probePath in new[] { Path.Combine(defaultPath, version!, "bin", "x64"), Path.Combine(defaultPath, version!, "bin") })
                        {
                            if (Directory.Exists(probePath))
                            {
                                var dll = Directory.GetFiles(probePath, "cudart64_*.dll")
                                                   .OrderByDescending(f => f)
                                                   .FirstOrDefault();
                                if (dll != null && NativeLibrary.TryLoad(dll, out var handle))
                                {
                                    return handle;
                                }
                            }
                        }
                    }
                }
            }

            // Final fallback to searching .nuget packages in user profile
            var userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            var nugetPath = Path.Combine(userProfile, ".nuget", "packages");
            if (Directory.Exists(nugetPath))
            {
                var dll = Directory.GetFiles(nugetPath, "cudart64_*.dll", SearchOption.AllDirectories)
                                   .OrderByDescending(Path.GetFileName)
                                   .FirstOrDefault();
                if (dll != null && NativeLibrary.TryLoad(dll, out var handle))
                {
                    return handle;
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

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    delegate int cuDriverGetVersionDelegate(out int version);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    delegate int nvrtcVersionDelegate(out int major, out int minor);

    static int GetDriverVersion(Assembly assembly, DllImportSearchPath? searchPath)
    {
        var nvcudaHandle = OnDllImport("nvcuda", assembly, searchPath);
        if (nvcudaHandle != IntPtr.Zero)
        {
            if (NativeLibrary.TryGetExport(nvcudaHandle, "cuDriverGetVersion", out var exportHandle))
            {
                var cuDriverGetVersion = Marshal.GetDelegateForFunctionPointer<cuDriverGetVersionDelegate>(exportHandle);
                if (cuDriverGetVersion(out var version) == 0)
                {
                    return version;
                }
            }
        }
        return 99999;
    }

    static bool IsNvrtcCompatible(IntPtr nvrtcHandle, int maxCudaVersion)
    {
        if (NativeLibrary.TryGetExport(nvrtcHandle, "nvrtcVersion", out var nvrtcVersionExport))
        {
            var nvrtcVersion = Marshal.GetDelegateForFunctionPointer<nvrtcVersionDelegate>(nvrtcVersionExport);
            if (nvrtcVersion(out var major, out var minor) == 0)
            {
                var nvrtcCudaVersion = major * 1000 + minor * 10;
                if (nvrtcCudaVersion > maxCudaVersion)
                {
                    return false;
                }
            }
        }
        return true;
    }
}
