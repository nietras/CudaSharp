# CUDA Tile C++ kernel sources

This directory contains CUDA Tile C++ kernel definitions used to exercise the C# runtime compilation and autotuning APIs.

The files under `tilegym/` are copied from [NVIDIA TileGym](https://github.com/NVIDIA/TileGym/tree/main/src/tilegym/ops/tilecpp) and retain their original SPDX copyright and MIT license notices.

They require CUDA Tile C++ support from CUDA 13.3 or later. CudaSharp compiles them through NVRTC and its NuGet-distributed bundled headers; no machine-wide CUDA Toolkit installation is assumed.

The TileGym console shows compilation and driver-JIT progress before each blocking operation, and prints the absolute report directory at startup. Add `--trace-compilation` to also show the detailed NVRTC stage timings in the console; without that switch, those traces remain in Visual Studio's Debug Output window.

NVRTC compilation and CUDA driver JIT loading are separate costs. Changing from individual specializations to shared modules can require new JIT cache entries, even when the old individual modules were cached. Compare both implementations with equally cold or equally warm driver caches rather than comparing the first batched run with a previously warmed individual run.
