# CUDA Tile C++ kernel sources

This directory contains CUDA Tile C++ kernel definitions used to exercise the C# runtime compilation and autotuning APIs.

The files under `tilegym/` are copied from [NVIDIA TileGym](https://github.com/NVIDIA/TileGym/tree/main/src/tilegym/ops/tilecpp) and retain their original SPDX copyright and MIT license notices.

They require CUDA Tile C++ support from CUDA 13.3 or later. CudaSharp compiles them through NVRTC and its NuGet-distributed bundled headers; no machine-wide CUDA Toolkit installation is assumed.
