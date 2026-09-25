#pragma once

#include "convolution_common.cuh"

template<int N, int CI, int CO, int D, int H, int W, int OD, int OH, int OW,
    int KD, int KH, int KW, int SD, int SH, int SW, int PD, int PH, int PW,
    int DD, int DH, int DW, int GROUPS, int BLOCK>
__tile_global__ void conv3d_implicit_gemm_kernel(const float* input, const float* weights,
    float* output) {
    convolution_body<false, false, true, false, N, CI, CO, D, H, W, OD, OH, OW,
        KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW, GROUPS, BLOCK>(
            input, weights, nullptr, nullptr, output);
}