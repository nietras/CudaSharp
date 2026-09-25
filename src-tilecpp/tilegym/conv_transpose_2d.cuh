#pragma once

#include "convolution_common.cuh"

template<int N, int CI, int CO, int H, int W, int OH, int OW,
    int KH, int KW, int SH, int SW, int PH, int PW, int DH, int DW,
    int GROUPS, int BLOCK>
__tile_global__ void conv_transpose_2d_implicit_gemm_kernel(const float* input, const float* weights,
    const float* conv_bias, const float* model_bias, float* output) {
    convolution_body<true, true, false, true, N, CI, CO, 1, H, W, 1, OH, OW,
        1, KH, KW, 1, SH, SW, 0, PH, PW, 1, DH, DW, GROUPS, BLOCK>(
            input, weights, conv_bias, model_bias, output);
}