#pragma once

#include <cuda_tile.h>

// NCDHW input/output; weights are packed as [output channel, input channel per group, kernel depth, height, width].
template<bool TRANSPOSED, bool BIAS, bool RELU, bool MODEL_BIAS,
    int N, int CI, int CO, int D, int H, int W, int OD, int OH, int OW,
    int KD, int KH, int KW, int SD, int SH, int SW, int PD, int PH, int PW,
    int DD, int DH, int DW, int GROUPS, int BLOCK>
__tile__ void convolution_body(const float* input, const float* weights,
    const float* conv_bias, const float* model_bias, float* output) {
    namespace ct = cuda::tiles;
    using IntTile = ct::tile<int, ct::shape<BLOCK>>;
    using FloatTile = ct::tile<float, ct::shape<BLOCK>>;
    constexpr int CI_GROUP = CI / GROUPS;
    constexpr int CO_GROUP = CO / GROUPS;
    constexpr int K = CI_GROUP * KD * KH * KW;
    constexpr int TOTAL = N * CO * OD * OH * OW;

    auto index = ct::iota<IntTile>() + static_cast<int>(ct::bid().x) * BLOCK;
    auto valid_output = index < TOTAL;
    auto ow = index % OW;
    auto oh = (index / OW) % OH;
    auto od = (index / (OW * OH)) % OD;
    auto oc = (index / (OW * OH * OD)) % CO;
    auto batch = index / (OW * OH * OD * CO);
    auto group = oc / CO_GROUP;
    auto acc = ct::zeros<FloatTile>();

    for (int k = 0; k < K; ++k) {
        int ic = k / (KD * KH * KW);
        int kd = (k / (KH * KW)) % KD;
        int kh = (k / KW) % KH;
        int kw = k % KW;
        auto in_d = od * SD - PD + kd * DD;
        auto in_h = oh * SH - PH + kh * DH;
        auto in_w = ow * SW - PW + kw * DW;
        auto valid = valid_output;
        if constexpr (TRANSPOSED) {
            auto d_num = od + PD - kd * DD;
            auto h_num = oh + PH - kh * DH;
            auto w_num = ow + PW - kw * DW;
            in_d = d_num / SD;
            in_h = h_num / SH;
            in_w = w_num / SW;
            valid = valid & (d_num % SD == 0) & (h_num % SH == 0) & (w_num % SW == 0);
        }
        valid = valid & (in_d >= 0) & (in_d < D) & (in_h >= 0) & (in_h < H)
            & (in_w >= 0) & (in_w < W);
        auto input_index = ((((batch * CI + group * CI_GROUP + ic) * D + in_d) * H + in_h) * W + in_w);
        auto weight_index = oc * K + k;
        auto a = ct::load_masked(input + input_index, valid, 0.0f);
        auto b = ct::load_masked(weights + weight_index, valid_output, 0.0f);
        acc = acc + a * b;
    }
    if constexpr (BIAS) {
        acc = acc + ct::load_masked(conv_bias + oc, valid_output, 0.0f);
    }
    if constexpr (RELU) {
        acc = ct::max(acc, ct::zeros<FloatTile>());
    }
    if constexpr (MODEL_BIAS) {
        acc = acc + ct::load_masked(model_bias + oc, valid_output, 0.0f);
    }
    ct::store_masked(output + index, acc, valid_output);
}