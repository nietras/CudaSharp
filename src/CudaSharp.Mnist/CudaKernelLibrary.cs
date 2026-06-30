namespace CudaSharp.Mnist;

public static class CudaKernelLibrary
{
    public static readonly string Preamble =
        """
        #include <cuda_fp16.h>

        typedef unsigned int uint32_t;

        #ifndef BATCH_SIZE
        #define BATCH_SIZE 256
        #endif
        #ifndef FILTER1_SIZE
        #define FILTER1_SIZE 5
        #endif
        #ifndef FILTER2_SIZE
        #define FILTER2_SIZE 5
        #endif
        #define INPUT_SIZE 28
        #ifndef POOL1_SIZE
        #define POOL1_SIZE 12
        #endif
        #ifndef POOL2_SIZE
        #define POOL2_SIZE 4
        #endif
        #ifndef FC1_INPUTS
        #define FC1_INPUTS 256
        #endif
        #ifndef FC1_OUTPUTS
        #define FC1_OUTPUTS 120
        #endif
        #define FC2_OUTPUTS 10

        #ifndef BATCHES_PER_EPOCH
        #define BATCHES_PER_EPOCH 200
        #endif
        #ifndef TOTAL_STEPS
        #define TOTAL_STEPS 300
        #endif
        """;

    public static readonly string ClearGradient =
        """

        extern "C" __global__ void clear_gradient(
            __half* __restrict__ d_grad, int num_elements)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            __half zero = __float2half(0.0f);
            for (int i = tid; i < num_elements; i += stride)
            {
                d_grad[i] = zero;
            }
        }
        """;

    public static readonly string ActivationGelu =
        """

        __device__ inline __half htanh(__half x)
        {
            __half clamped_x = __hmin(__hmax(x, __float2half(-5.5f)), __float2half(5.5f));
            __half two_x = __hmul(__float2half(2.0f), clamped_x);
            __half exp_two_x = hexp(two_x);
            return __hdiv(__hsub(exp_two_x, __float2half(1.0f)), __hadd(exp_two_x, __float2half(1.0f)));
        }

        __device__ inline __half gelu(__half x)
        {
            __half x3 = __hmul(__hmul(x, x), x);
            __half inner = __hadd(x, __hmul(__float2half(0.044715f), x3));
            __half tanh_arg = __hmul(__float2half(0.79788456f), inner);
            __half t = htanh(tanh_arg);
            return __hmul(__float2half(0.5f), __hmul(x, __hadd(__float2half(1.0f), t)));
        }

        __device__ inline __half d_gelu(__half x, __half dy)
        {
            __half x2 = __hmul(x, x);
            __half x3 = __hmul(x2, x);
            __half inner = __hadd(x, __hmul(__float2half(0.044715f), x3));
            __half tanh_arg = __hmul(__float2half(0.79788456f), inner);
            __half t = htanh(tanh_arg);
            __half t2 = __hmul(t, t);
            __half sech2 = __hsub(__float2half(1.0f), t2);
            __half dtanh_coef = __hmul(__float2half(0.79788456f), __hadd(__float2half(1.0f), __hmul(__float2half(0.134145f), x2)));
            __half dtanh = __hmul(dtanh_coef, sech2);
            __half term1 = __hmul(__float2half(0.5f), __hadd(__float2half(1.0f), t));
            __half term2 = __hmul(__hmul(__float2half(0.5f), x), dtanh);
            __half derivative = __hadd(term1, term2);
            return __hmul(dy, derivative);
        }
        """;


    public static readonly string ActivationSilu =
        """

        __device__ inline __half silu(__half x)
        {
            __half clamped_x = __hmin(__hmax(x, __float2half(-10.0f)), __float2half(10.0f));
            __half exp_neg_x = hexp(__hneg(clamped_x));
            __half sigmoid = __hdiv(__float2half(1.0f), __hadd(__float2half(1.0f), exp_neg_x));
            return __hmul(x, sigmoid);
        }

        __device__ inline __half d_silu(__half x, __half dy)
        {
            __half clamped_x = __hmin(__hmax(x, __float2half(-10.0f)), __float2half(10.0f));
            __half exp_neg_x = hexp(__hneg(clamped_x));
            __half sigmoid = __hdiv(__float2half(1.0f), __hadd(__float2half(1.0f), exp_neg_x));
            __half silu_val = __hmul(x, sigmoid);
            __half one_minus_sig = __hsub(__float2half(1.0f), sigmoid);
            __half derivative = __hadd(sigmoid, __hmul(silu_val, one_minus_sig));
            return __hmul(dy, derivative);
        }
        """;

    public static readonly string DummyFusedKernels =
        """

        extern "C" __global__ void fused_forward(
            const uint32_t* d_inputs, const __half* d_conv1_filters,
            const __half* d_conv1_biases, const __half* d_conv2_filters,
            const __half* d_conv2_biases, const __half* d_fc2_weights,
            const __half* d_fc2_biases, __half* d_conv1_out,
            __half* d_conv1_unpooled, __half* d_conv2_out,
            __half* d_conv2_unpooled, __half* d_fc2_out,
            const int* d_step, int is_training) {}

        extern "C" __global__ void fused_backward(
            const __half* d_fc2_out, const unsigned char* d_labels,
            const __half* d_conv2_out, const __half* d_fc2_weights,
            __half* d_fc2_weights_grad, __half* d_fc2_biases_grad,
            const __half* d_conv2_unpooled, const __half* d_conv1_out,
            const __half* d_conv2_filters, __half* d_conv2_filters_grad,
            __half* d_conv2_biases_grad, const __half* d_conv1_unpooled,
            const uint32_t* d_inputs, __half* d_conv1_filters_grad,
            __half* d_conv1_biases_grad, const int* d_step) {}
        """;

    public static readonly string Conv1Forward =
        """

        extern "C" __global__ void conv1_forward(
            const uint32_t* __restrict__ d_inputs,
            const __half* __restrict__ d_filters,
            const __half* __restrict__ d_biases,
            __half* __restrict__ d_outputs,
            __half* __restrict__ d_unpooled_vals,
            const int* __restrict__ d_step,
            int is_training)
        {
            const int batch_idx = blockIdx.x;
            const int filter_idx = blockIdx.y;
            const int out_x = threadIdx.x;
            const int out_y = threadIdx.y;

            if (out_x >= POOL1_SIZE || out_y >= POOL1_SIZE) return;

            int batchOffset = ((*d_step) % BATCHES_PER_EPOCH) * BATCH_SIZE;

            __shared__ __half s_filter[FILTER1_SIZE][FILTER1_SIZE];
            int tid_flat = threadIdx.y * POOL1_SIZE + threadIdx.x;
            if (tid_flat < FILTER1_SIZE * FILTER1_SIZE)
            {
                s_filter[tid_flat / FILTER1_SIZE][tid_flat % FILTER1_SIZE] =
                    d_filters[filter_idx * FILTER1_SIZE * FILTER1_SIZE + tid_flat];
            }

            __shared__ uint32_t s_image[28];
            if (tid_flat < 28)
            {
                s_image[tid_flat] =
                    d_inputs[(batchOffset + batch_idx) * 28 + tid_flat];
            }
            __syncthreads();

            const int conv_x_base = out_x * 2;
            const int conv_y_base = out_y * 2;

            int seed = batch_idx + *d_step;
            int dx = (is_training == 1)
                ? ((seed * 1103515245 + 12345) % 3 - 1) : 0;
            int dy = (is_training == 1)
                ? (((seed * 1103515245 + 12345) / 3) % 3 - 1) : 0;

            __half max_val = __float2half(-1e9f);

            #pragma unroll
            for (int py = 0; py < 2; py++)
            {
                #pragma unroll
                for (int px = 0; px < 2; px++)
                {
                    const int cx = conv_x_base + px;
                    const int cy = conv_y_base + py;

                    __half sum = d_biases[filter_idx];

                    #pragma unroll
                    for (int fy = 0; fy < FILTER1_SIZE; fy++)
                    {
                        int shift_y = cy + fy + dy;
                        uint32_t row_bits = 0;
                        if (shift_y >= 0 && shift_y < 28)
                        {
                            row_bits = s_image[shift_y];
                        }
                        #pragma unroll
                        for (int fx = 0; fx < FILTER1_SIZE; fx++)
                        {
                            int img_x = cx + fx + dx;
                            uint32_t pixel = 0;
                            if (img_x >= 0 && img_x < 28)
                            {
                                pixel = (row_bits >> img_x) & 1u;
                            }
                            if (pixel != 0)
                            {
                                sum = __hadd(sum, s_filter[fy][fx]);
                            }
                        }
                    }

                    int conv_out_w = POOL1_SIZE * 2;
                    int unpooled_idx = (batch_idx * conv_out_w * conv_out_w
                        + cy * conv_out_w + cx) * CONV1_FILTER_COUNT + filter_idx;
                    d_unpooled_vals[unpooled_idx] = sum;

                    __half activated = ACTIVATION_FWD(sum);
                    if (__hgt(activated, max_val))
                    {
                        max_val = activated;
                    }
                }
            }

            const int out_idx = batch_idx * POOL1_SIZE * POOL1_SIZE
                * CONV1_FILTER_COUNT
                + (out_y * POOL1_SIZE + out_x) * CONV1_FILTER_COUNT
                + filter_idx;
            d_outputs[out_idx] = max_val;
        }
        """;

    public static readonly string Conv2Forward =
        """

        extern "C" __global__ void conv2_forward(
            const __half* __restrict__ d_inputs,
            const __half* __restrict__ d_filters,
            const __half* __restrict__ d_biases,
            __half* __restrict__ d_outputs,
            __half* __restrict__ d_unpooled_vals)
        {
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x;

            const int conv1_out_per_sample =
                POOL1_SIZE * POOL1_SIZE * CONV1_FILTER_COUNT;
            const int filter_weights =
                CONV2_FILTER_COUNT * CONV1_FILTER_COUNT
                * FILTER2_SIZE * FILTER2_SIZE;

            __shared__ __half s_input[POOL1_SIZE * POOL1_SIZE
                * CONV1_FILTER_COUNT];
            __shared__ __half s_filters[CONV2_FILTER_COUNT
                * CONV1_FILTER_COUNT * FILTER2_SIZE * FILTER2_SIZE];

            for (int i = tid; i < conv1_out_per_sample; i += 256)
            {
                s_input[i] = d_inputs[batch_idx * conv1_out_per_sample + i];
            }

            for (int i = tid; i < filter_weights; i += 256)
            {
                s_filters[i] = d_filters[i];
            }
            __syncthreads();

            #if IS_GAP == 1
            if (tid < CONV2_FILTER_COUNT)
            {
                int filter_idx = tid;
                float gap_sum = 0.0f;

                for (int cy = 0; cy < 8; cy++)
                {
                    for (int cx = 0; cx < 8; cx++)
                    {
                        __half sum = d_biases[filter_idx];

                        #pragma unroll
                        for (int c = 0; c < CONV1_FILTER_COUNT; c++)
                        {
                            #pragma unroll
                            for (int fy = 0; fy < FILTER2_SIZE; fy++)
                            {
                                #pragma unroll
                                for (int fx = 0; fx < FILTER2_SIZE; fx++)
                                {
                                    int in_x = cx + fx;
                                    int in_y = cy + fy;
                                    sum += s_input[
                                        (in_y * POOL1_SIZE + in_x)
                                        * CONV1_FILTER_COUNT + c]
                                        * s_filters[
                                        filter_idx * CONV1_FILTER_COUNT
                                        * FILTER2_SIZE * FILTER2_SIZE
                                        + (fy * FILTER2_SIZE + fx)
                                        * CONV1_FILTER_COUNT + c];
                                }
                            }
                        }

                        int unpooled_idx = (batch_idx * 64 + cy * 8 + cx)
                            * CONV2_FILTER_COUNT + filter_idx;
                        d_unpooled_vals[unpooled_idx] = sum;

                        gap_sum += __half2float(ACTIVATION_FWD(sum));
                    }
                }

                gap_sum /= 64.0f;
                d_outputs[batch_idx * CONV2_FILTER_COUNT + filter_idx] = __float2half(gap_sum);
            }
            #else
            const int conv2_spatial = POOL2_SIZE * POOL2_SIZE;
            const int total_outputs = CONV2_FILTER_COUNT * conv2_spatial;

            #pragma unroll
            for (int out_idx = tid; out_idx < total_outputs; out_idx += 256)
            {
                int filter_idx = out_idx / conv2_spatial;
                int spatial_idx = out_idx % conv2_spatial;
                int out_x = spatial_idx % POOL2_SIZE;
                int out_y = spatial_idx / POOL2_SIZE;

                const int conv_x_base = out_x * 2;
                const int conv_y_base = out_y * 2;

                __half max_val = __float2half(-1e9f);

                #pragma unroll
                for (int py = 0; py < 2; py++)
                {
                    #pragma unroll
                    for (int px = 0; px < 2; px++)
                    {
                        const int cx = conv_x_base + px;
                        const int cy = conv_y_base + py;

                        __half sum = d_biases[filter_idx];

                        #pragma unroll
                        for (int c = 0; c < CONV1_FILTER_COUNT; c++)
                        {
                            #pragma unroll
                            for (int fy = 0; fy < FILTER2_SIZE; fy++)
                            {
                                #pragma unroll
                                for (int fx = 0; fx < FILTER2_SIZE; fx++)
                                {
                                    int in_x = cx + fx;
                                    int in_y = cy + fy;
                                    sum += s_input[
                                        (in_y * POOL1_SIZE + in_x)
                                        * CONV1_FILTER_COUNT + c]
                                        * s_filters[
                                        (filter_idx * CONV1_FILTER_COUNT + c) * FILTER2_SIZE * FILTER2_SIZE + fy * FILTER2_SIZE + fx];
                                }
                            }
                        }

                        int conv2_out_w = POOL2_SIZE * 2;
                        int unpooled_idx = (batch_idx * conv2_out_w
                            * conv2_out_w + cy * conv2_out_w + cx)
                            * CONV2_FILTER_COUNT + filter_idx;
                        d_unpooled_vals[unpooled_idx] = sum;

                        __half activated = ACTIVATION_FWD(sum);
                        if (__hgt(activated, max_val))
                        {
                            max_val = activated;
                        }
                    }
                }

                const int out_idx_global = batch_idx
                    * CONV2_FILTER_COUNT * conv2_spatial
                    + (out_y * POOL2_SIZE + out_x)
                    * CONV2_FILTER_COUNT + filter_idx;
                d_outputs[out_idx_global] = max_val;
            }
            #endif
        }
        """;

    public static readonly string Fc1Forward =
        """

        #if HAS_DROPOUT == 1
        extern "C" __global__ void fc1_forward(
            const __half* __restrict__ d_inputs,
            const __half* __restrict__ d_weights,
            const __half* __restrict__ d_biases,
            __half* __restrict__ d_outputs,
            __half* __restrict__ d_unpooled_vals,
            const int* __restrict__ d_step,
            const int is_training)
        #else
        extern "C" __global__ void fc1_forward(
            const __half* __restrict__ d_inputs,
            const __half* __restrict__ d_weights,
            const __half* __restrict__ d_biases,
            __half* __restrict__ d_outputs,
            __half* __restrict__ d_unpooled_vals)
        #endif
        {
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x;

            __shared__ __half s_input[FC1_INPUTS];
            if (tid < FC1_INPUTS)
            {
                s_input[tid] = d_inputs[batch_idx * FC1_INPUTS + tid];
            }
            __syncthreads();

            #if USE_HALF2 == 1
            if (tid < FC1_OUTPUTS / 2)
            {
                const __half2* d_weights_h2 = (const __half2*)d_weights;
                __half2 sum2 = ((const __half2*)d_biases)[tid];
                #pragma unroll
                for (int i = 0; i < FC1_INPUTS; i++)
                {
                    __half2 in2 = __halves2half2(s_input[i], s_input[i]);
                    sum2 = __hfma2(in2, d_weights_h2[i * (FC1_OUTPUTS / 2) + tid], sum2);
                }
                
                #if HAS_DROPOUT == 1
                if (is_training == 1)
                {
                    uint32_t state = (batch_idx + *d_step) * 1103515245 + tid * 12345 + 12345;
                    state = (state * 1103515245 + 12345);
                    float rand_val = (float)(state & 0x7fffffff) / 2147483648.0f;
                    
                    __half2* d_unpooled_h2 = (__half2*)d_unpooled_vals;
                    __half2* d_outputs_h2 = (__half2*)d_outputs;
                    
                    if (rand_val < DROPOUT_RATE)
                    {
                        d_unpooled_h2[batch_idx * (FC1_OUTPUTS / 2) + tid] = __halves2half2(__float2half(-1e9f), __float2half(-1e9f));
                        d_outputs_h2[batch_idx * (FC1_OUTPUTS / 2) + tid] = __halves2half2(__float2half(0.0f), __float2half(0.0f));
                    }
                    else
                    {
                        d_unpooled_h2[batch_idx * (FC1_OUTPUTS / 2) + tid] = sum2;
                        float scale = 1.0f / (1.0f - DROPOUT_RATE);
                        __half act0 = __float2half(__half2float(ACTIVATION_FWD(sum2.x)) * scale);
                        __half act1 = __float2half(__half2float(ACTIVATION_FWD(sum2.y)) * scale);
                        d_outputs_h2[batch_idx * (FC1_OUTPUTS / 2) + tid] = __halves2half2(act0, act1);
                    }
                }
                else
                {
                    ((__half2*)d_unpooled_vals)[batch_idx * (FC1_OUTPUTS / 2) + tid] = sum2;
                    ((__half2*)d_outputs)[batch_idx * (FC1_OUTPUTS / 2) + tid] = __halves2half2(ACTIVATION_FWD(sum2.x), ACTIVATION_FWD(sum2.y));
                }
                #else
                ((__half2*)d_unpooled_vals)[batch_idx * (FC1_OUTPUTS / 2) + tid] = sum2;
                ((__half2*)d_outputs)[batch_idx * (FC1_OUTPUTS / 2) + tid] = __halves2half2(ACTIVATION_FWD(sum2.x), ACTIVATION_FWD(sum2.y));
                #endif
            }
            #else
            if (tid < FC1_OUTPUTS)
            {
                __half sum = d_biases[tid];
                #pragma unroll
                for (int i = 0; i < FC1_INPUTS; i++)
                {
                    sum += s_input[i] * d_weights[i * FC1_OUTPUTS + tid];
                }
                
                #if HAS_DROPOUT == 1
                if (is_training == 1)
                {
                    uint32_t state = (batch_idx + *d_step) * 1103515245 + tid * 12345 + 12345;
                    state = (state * 1103515245 + 12345);
                    float rand_val = (float)(state & 0x7fffffff) / 2147483648.0f;
                    
                    if (rand_val < DROPOUT_RATE)
                    {
                        d_unpooled_vals[batch_idx * FC1_OUTPUTS + tid] = __float2half(-1e9f);
                        d_outputs[batch_idx * FC1_OUTPUTS + tid] = __float2half(0.0f);
                    }
                    else
                    {
                        d_unpooled_vals[batch_idx * FC1_OUTPUTS + tid] = sum;
                        float scale = 1.0f / (1.0f - DROPOUT_RATE);
                        d_outputs[batch_idx * FC1_OUTPUTS + tid] = __float2half(__half2float(ACTIVATION_FWD(sum)) * scale);
                    }
                }
                else
                {
                    d_unpooled_vals[batch_idx * FC1_OUTPUTS + tid] = sum;
                    d_outputs[batch_idx * FC1_OUTPUTS + tid] = ACTIVATION_FWD(sum);
                }
                #else
                d_unpooled_vals[batch_idx * FC1_OUTPUTS + tid] = sum;
                d_outputs[batch_idx * FC1_OUTPUTS + tid] = ACTIVATION_FWD(sum);
                #endif
            }
            #endif
        }
        """;

    public static readonly string Fc2Forward =
        """

        extern "C" __global__ void fc2_forward(
            const __half* __restrict__ d_inputs,
            const __half* __restrict__ d_weights,
            const __half* __restrict__ d_biases,
            __half* __restrict__ d_outputs)
        {
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x;

            __shared__ __half s_input[FC2_INPUTS];
            if (tid < FC2_INPUTS)
            {
                s_input[tid] = d_inputs[batch_idx * FC2_INPUTS + tid];
            }
            __syncthreads();

            #if USE_HALF2 == 1
            if (tid < 5)
            {
                const __half2* d_weights_h2 = (const __half2*)d_weights;
                __half2 sum2 = ((const __half2*)d_biases)[tid];
                #pragma unroll
                for (int i = 0; i < FC2_INPUTS; i++)
                {
                    __half2 in2 = __halves2half2(s_input[i], s_input[i]);
                    sum2 = __hfma2(in2, d_weights_h2[i * 5 + tid], sum2);
                }
                ((__half2*)d_outputs)[batch_idx * 5 + tid] = sum2;
            }
            #else
            if (tid < 10)
            {
                __half sum = d_biases[tid];
                #pragma unroll
                for (int i = 0; i < FC2_INPUTS; i++)
                {
                    sum += s_input[i] * d_weights[i * 10 + tid];
                }
                d_outputs[batch_idx * 10 + tid] = sum;
            }
            #endif
        }
        """;

    public static readonly string Fc2Backward =
        """

        extern "C" __global__ void fc2_backward(
            const __half* __restrict__ d_fc2_outputs,
            const unsigned char* __restrict__ d_labels,
            const __half* __restrict__ d_fc1_outputs,
            const __half* __restrict__ d_fc2_weights,
            __half* __restrict__ d_fc2_weights_grad,
            __half* __restrict__ d_fc2_biases_grad,
            __half* __restrict__ d_fc1_out_grad,
            const int* __restrict__ d_step,
            const __half* __restrict__ d_fc1_unpooled)
        {
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x;

            __shared__ __half s_grad[10];

            int batchOffset = ((*d_step) % BATCHES_PER_EPOCH) * BATCH_SIZE;

            if (tid < 10)
            {
                float max_logit = -1e9f;
                for (int i = 0; i < 10; i++)
                {
                    float logit = __half2float(
                        d_fc2_outputs[batch_idx * 10 + i]);
                    if (logit > max_logit) max_logit = logit;
                }

                float sum_exp = 0.0f;
                for (int i = 0; i < 10; i++)
                {
                    sum_exp += expf(__half2float(
                        d_fc2_outputs[batch_idx * 10 + i]) - max_logit);
                }

                float prob = expf(__half2float(
                    d_fc2_outputs[batch_idx * 10 + tid]) - max_logit)
                    / sum_exp;
                int correct_label = d_labels[batchOffset + batch_idx];
                s_grad[tid] = __float2half(
                    prob - (tid == correct_label ? 1.0f : 0.0f));
            }
            __syncthreads();

            if (tid < FC2_INPUTS)
            {
                #if USE_HALF2 == 1
                __half2 sum2 = __float2half2_rn(0.0f);
                const __half2* d_fc2_weights_h2 = (const __half2*)&d_fc2_weights[tid * 10];
                const __half2* s_grad_h2 = (const __half2*)s_grad;
                #pragma unroll
                for (int c2 = 0; c2 < 5; c2++)
                {
                    sum2 = __hfma2(s_grad_h2[c2], d_fc2_weights_h2[c2], sum2);
                }
                __half sum_input_grad = sum2.x + sum2.y;
                #else
                __half sum_input_grad = __float2half(0.0f);
                #pragma unroll
                for (int c = 0; c < 10; c++)
                {
                    sum_input_grad += s_grad[c] * d_fc2_weights[tid * 10 + c];
                }
                #endif
                
                #if HAS_FC1 == 1
                __half fc1_unpooled = d_fc1_unpooled[batch_idx * FC2_INPUTS + tid];
                __half out_grad = ACTIVATION_BWD(fc1_unpooled, sum_input_grad);
                #if HAS_DROPOUT == 1
                if (__half2float(fc1_unpooled) > -1e8f) {
                    out_grad = __float2half(__half2float(out_grad) * (1.0f / (1.0f - DROPOUT_RATE)));
                } else {
                    out_grad = __float2half(0.0f);
                }
                #endif
                d_fc1_out_grad[batch_idx * FC2_INPUTS + tid] = out_grad;
                #else
                d_fc1_out_grad[batch_idx * FC2_INPUTS + tid] = sum_input_grad;
                #endif
            }

            if (tid < 10)
            {
                __half b_grad = s_grad[tid];
                if (__half2float(b_grad) != 0.0f)
                {
                    atomicAdd(&d_fc2_biases_grad[tid], b_grad);
                }
            }
        }
        """;

    public static readonly string Fc2BackwardWeights =
        """

        extern "C" __global__ void fc2_backward_weights(
            const __half* __restrict__ d_fc2_outputs,
            const unsigned char* __restrict__ d_labels,
            const __half* __restrict__ d_fc1_outputs,
            __half* __restrict__ d_fc2_weights_grad,
            const int* __restrict__ d_step)
        {
            const int input_idx = blockIdx.x;
            const int tid = threadIdx.x;

            int batchOffset = ((*d_step) % BATCHES_PER_EPOCH) * BATCH_SIZE;

            __shared__ __half s_fc2_outputs[BATCH_SIZE][10];
            for (int i = tid; i < BATCH_SIZE; i += 128)
            {
                #pragma unroll
                for (int c = 0; c < 10; c++)
                {
                    s_fc2_outputs[i][c] = d_fc2_outputs[i * 10 + c];
                }
            }

            __shared__ __half s_weight_grads[128][10];

            #pragma unroll
            for (int c = 0; c < 10; c++)
            {
                s_weight_grads[tid][c] = __float2half(0.0f);
            }
            __syncthreads();

            #pragma unroll
            for (int i = 0; i < (BATCH_SIZE / 128); i++)
            {
                int b = i * 128 + tid;

                float max_logit = -1e9f;
                for (int c = 0; c < 10; c++)
                {
                    float logit = __half2float(s_fc2_outputs[b][c]);
                    if (logit > max_logit) max_logit = logit;
                }

                float sum_exp = 0.0f;
                for (int c = 0; c < 10; c++)
                {
                    sum_exp += expf(__half2float(s_fc2_outputs[b][c]) - max_logit);
                }

                int correct_label = d_labels[batchOffset + b];
                __half x_val = d_fc1_outputs[b * FC2_INPUTS + input_idx];

                #pragma unroll
                for (int c = 0; c < 10; c++)
                {
                    float prob = expf(__half2float(s_fc2_outputs[b][c]) - max_logit) / sum_exp;
                    __half g_val = __float2half(prob - (c == correct_label ? 1.0f : 0.0f));
                    s_weight_grads[tid][c] = __hfma(g_val, x_val, s_weight_grads[tid][c]);
                }
            }
            __syncthreads();

            for (int stride = 64; stride > 0; stride >>= 1)
            {
                if (tid < stride)
                {
                    #pragma unroll
                    for (int c = 0; c < 10; c++)
                    {
                        s_weight_grads[tid][c] = __hadd(s_weight_grads[tid][c], s_weight_grads[tid + stride][c]);
                    }
                }
                __syncthreads();
            }

            if (tid == 0)
            {
                #pragma unroll
                for (int c = 0; c < 10; c++)
                {
                    d_fc2_weights_grad[input_idx * 10 + c] = s_weight_grads[0][c];
                }
            }
        }
        """;

    public static readonly string Fc1Backward =
        """

        extern "C" __global__ void fc1_backward(
            const __half* __restrict__ d_fc1_out_grad,
            const __half* __restrict__ d_fc1_outputs,
            const __half* __restrict__ d_fc2_inputs,
            const __half* __restrict__ d_fc1_weights,
            __half* __restrict__ d_fc1_biases_grad,
            __half* __restrict__ d_conv2_out_grad)
        {
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x;

            __shared__ __half s_grad[FC1_OUTPUTS];

            if (tid < FC1_OUTPUTS)
            {
                s_grad[tid] = d_fc1_out_grad[
                    batch_idx * FC1_OUTPUTS + tid];
            }
            __syncthreads();

            if (tid < FC1_INPUTS)
            {
                __half sum_grad = __float2half(0.0f);
                #pragma unroll 8
                for (int c = 0; c < FC1_OUTPUTS; c++)
                {
                    sum_grad = __hfma(s_grad[c], d_fc1_weights[tid * FC1_OUTPUTS + c], sum_grad);
                }
                d_conv2_out_grad[batch_idx * FC1_INPUTS + tid] = sum_grad;
            }

            if (tid < FC1_OUTPUTS)
            {
                __half b_grad = s_grad[tid];
                if (__half2float(b_grad) != 0.0f)
                {
                    atomicAdd(&d_fc1_biases_grad[tid], b_grad);
                }
            }
        }
        """;

    public static readonly string Fc1BackwardWeights =
        """

        extern "C" __global__ void fc1_backward_weights(
            const __half* __restrict__ d_fc1_out_grad,
            const __half* __restrict__ d_fc1_outputs,
            const __half* __restrict__ d_fc1_inputs,
            __half* __restrict__ d_fc1_weights_grad)
        {
            const int input_idx = blockIdx.x;
            const int tid = threadIdx.x;

            __shared__ __half s_accum[64][FC1_OUTPUTS];

            #pragma unroll
            for (int c = 0; c < FC1_OUTPUTS; c++)
            {
                s_accum[tid][c] = __float2half(0.0f);
            }
            __syncthreads();

            #pragma unroll
            for (int i = 0; i < (BATCH_SIZE / 64); i++)
            {
                int b = i * 64 + tid;
                __half x_val = d_fc1_inputs[b * FC1_INPUTS + input_idx];
                #pragma unroll
                for (int c = 0; c < FC1_OUTPUTS; c++)
                {
                    s_accum[tid][c] = __hfma(x_val, d_fc1_out_grad[b * FC1_OUTPUTS + c], s_accum[tid][c]);
                }
            }
            __syncthreads();

            for (int stride = 32; stride > 0; stride >>= 1)
            {
                if (tid < stride)
                {
                    #pragma unroll
                    for (int c = 0; c < FC1_OUTPUTS; c++)
                    {
                        s_accum[tid][c] = __hadd(s_accum[tid][c], s_accum[tid + stride][c]);
                    }
                }
                __syncthreads();
            }

            if (tid == 0)
            {
                #pragma unroll
                for (int c = 0; c < FC1_OUTPUTS; c++)
                {
                    d_fc1_weights_grad[input_idx * FC1_OUTPUTS + c] = s_accum[0][c];
                }
            }
        }
        """;

    public static readonly string Conv2Backward =
        """

        extern "C" __global__ void conv2_backward(
            const __half* __restrict__ d_conv2_out_grad,
            const __half* __restrict__ d_conv2_out_val,
            const __half* __restrict__ d_conv2_unpooled_vals,
            const __half* __restrict__ d_conv1_out,
            const __half* __restrict__ d_conv2_filters,
            __half* __restrict__ d_conv2_filters_grad,
            __half* __restrict__ d_conv2_biases_grad,
            __half* __restrict__ d_conv1_out_grad)
        {
            #define CONV2_BATCH_PER_CHUNK (BATCH_SIZE / CONV2_CHUNKS)

            const int filter_idx = blockIdx.x / CONV2_CHUNKS;
            const int chunk_idx = blockIdx.x % CONV2_CHUNKS;
            const int tid = threadIdx.x;

            const int conv1_out_per_sample =
                POOL1_SIZE * POOL1_SIZE * CONV1_FILTER_COUNT;
            const int filter_wt_per_filter =
                CONV1_FILTER_COUNT * FILTER2_SIZE * FILTER2_SIZE;
            #if IS_GAP == 1
            const int conv2_out_w = 8;
            #else
            const int conv2_out_w = POOL2_SIZE * 2;
            #endif

            __shared__ __half s_conv1_out[POOL1_SIZE * POOL1_SIZE
                * CONV1_FILTER_COUNT];
            __shared__ __half s_filter_grad[CONV1_FILTER_COUNT
                * FILTER2_SIZE * FILTER2_SIZE];
            __shared__ __half s_bias_grad;
            #if IS_GAP == 1
            __shared__ __half s_grad[8][8];
            #else
            __shared__ __half s_grad[POOL2_SIZE * 2][POOL2_SIZE * 2];
            #endif

            __half zero = __float2half(0.0f);

            for (int i = tid; i < filter_wt_per_filter; i += 128)
            {
                s_filter_grad[i] = zero;
            }
            if (tid == 0)
            {
                s_bias_grad = zero;
            }
            __syncthreads();

            const int cx = (tid < conv2_out_w * conv2_out_w)
                ? (tid % conv2_out_w) : 0;
            const int cy = (tid < conv2_out_w * conv2_out_w)
                ? (tid / conv2_out_w) : 0;
            const int px = cx / 2;
            const int py = cy / 2;

            const int start_b = chunk_idx * CONV2_BATCH_PER_CHUNK;
            const int end_b = start_b + CONV2_BATCH_PER_CHUNK;

            int c_arr[4], fx_arr[4], fy_arr[4];
            int i_count = 0;
            for (int i = tid; i < filter_wt_per_filter; i += 128)
            {
                fx_arr[i_count] = i % FILTER2_SIZE;
                fy_arr[i_count] = (i / FILTER2_SIZE) % FILTER2_SIZE;
                c_arr[i_count] = i / (FILTER2_SIZE * FILTER2_SIZE);
                i_count++;
            }

            for (int b = start_b; b < end_b; b++)
            {
                for (int i = tid; i < conv1_out_per_sample; i += 128)
                {
                    s_conv1_out[i] =
                        d_conv1_out[b * conv1_out_per_sample + i];
                }

                #if IS_GAP == 1
                __half out_grad = d_conv2_out_grad[b * CONV2_FILTER_COUNT + filter_idx];
                out_grad = __hmul(out_grad, __float2half(0.015625f));
                
                __half my_val = zero;
                if (tid < 64)
                {
                    my_val = d_conv2_unpooled_vals[
                        (b * 64 + tid)
                        * CONV2_FILTER_COUNT + filter_idx];
                }

                __half grad = zero;
                if (tid < 64)
                {
                    grad = ACTIVATION_BWD(my_val, out_grad);
                }
                #else
                __half out_grad = zero;
                __half out_val = zero;
                if (tid < conv2_out_w * conv2_out_w
                    && px < POOL2_SIZE && py < POOL2_SIZE)
                {
                    int pool_idx = (py * POOL2_SIZE + px)
                        * CONV2_FILTER_COUNT + filter_idx;
                    out_grad = d_conv2_out_grad[
                        b * CONV2_FILTER_COUNT * POOL2_SIZE * POOL2_SIZE
                        + pool_idx];
                    out_val = d_conv2_out_val[
                        b * CONV2_FILTER_COUNT * POOL2_SIZE * POOL2_SIZE
                        + pool_idx];
                }

                __half my_val = zero;
                if (tid < conv2_out_w * conv2_out_w)
                {
                    my_val = d_conv2_unpooled_vals[
                        (b * conv2_out_w * conv2_out_w + tid)
                        * CONV2_FILTER_COUNT + filter_idx];
                }

                __half grad = zero;
                if (tid < conv2_out_w * conv2_out_w
                    && __heq(ACTIVATION_FWD(my_val), out_val))
                {
                    grad = ACTIVATION_BWD(my_val, out_grad);
                }
                #endif
                if (tid < conv2_out_w * conv2_out_w)
                {
                    s_grad[cy][cx] = grad;
                }
                __syncthreads();

                int idx = 0;
                for (int i = tid; i < filter_wt_per_filter; i += 128)
                {
                    int c = c_arr[idx];
                    int fx = fx_arr[idx];
                    int fy = fy_arr[idx];
                    idx++;

                    __half w_grad = zero;
                    #pragma unroll
                    for (int y = 0; y < conv2_out_w; y++)
                    {
                        #pragma unroll
                        for (int x = 0; x < conv2_out_w; x++)
                        {
                            __half g = s_grad[y][x];
                            int in_x = x + fx;
                            int in_y = y + fy;
                            w_grad += g * s_conv1_out[
                                (in_y * POOL1_SIZE + in_x)
                                * CONV1_FILTER_COUNT + c];
                        }
                    }
                    s_filter_grad[i] += w_grad;
                }

                // Parallel bias gradient reduction across 64 threads
                __shared__ float s_bias_reduce[64];
                if (tid < 64)
                {
                    int rx = tid % conv2_out_w;
                    int ry = tid / conv2_out_w;
                    s_bias_reduce[tid] = (rx < conv2_out_w && ry < conv2_out_w) 
                        ? __half2float(s_grad[ry][rx]) 
                        : 0.0f;
                }
                __syncthreads();

                for (int stride = 32; stride > 0; stride >>= 1)
                {
                    if (tid < stride)
                    {
                        s_bias_reduce[tid] += s_bias_reduce[tid + stride];
                    }
                    __syncthreads();
                }

                if (tid == 0)
                {
                    s_bias_grad += __float2half(s_bias_reduce[0]);
                }

                for (int i = tid; i < conv1_out_per_sample; i += 128)
                {
                    int c = i % CONV1_FILTER_COUNT;
                    int spatial_idx = i / CONV1_FILTER_COUNT;
                    int ix = spatial_idx % POOL1_SIZE;
                    int iy = spatial_idx / POOL1_SIZE;

                    __half sum_grad = zero;
                    #pragma unroll
                    for (int fy = 0; fy < FILTER2_SIZE; fy++)
                    {
                        #pragma unroll
                        for (int fx = 0; fx < FILTER2_SIZE; fx++)
                        {
                            int x = ix - fx;
                            int y = iy - fy;
                            if (x >= 0 && x < conv2_out_w
                                && y >= 0 && y < conv2_out_w)
                            {
                                int f_idx = filter_idx
                                    * filter_wt_per_filter
                                    + c * FILTER2_SIZE * FILTER2_SIZE
                                    + fy * FILTER2_SIZE + fx;
                                sum_grad += s_grad[y][x]
                                    * d_conv2_filters[f_idx];
                            }
                        }
                    }
                    if (__hne(sum_grad, zero))
                    {
                        atomicAdd(&d_conv1_out_grad[
                            b * conv1_out_per_sample + i], sum_grad);
                    }
                }
                __syncthreads();
            }

            for (int i = tid; i < filter_wt_per_filter; i += 128)
            {
                atomicAdd(&d_conv2_filters_grad[
                    filter_idx * filter_wt_per_filter + i],
                    s_filter_grad[i]);
            }
            if (tid == 0)
            {
                atomicAdd(&d_conv2_biases_grad[filter_idx], s_bias_grad);
            }
        }
        """;

    public static readonly string Conv1Backward =
        """

        extern "C" __global__ void conv1_backward(
            const __half* __restrict__ d_conv1_out_grad,
            const __half* __restrict__ d_conv1_out_val,
            const __half* __restrict__ d_conv1_unpooled_vals,
            const uint32_t* __restrict__ d_inputs,
            __half* __restrict__ d_conv1_filters_grad,
            __half* __restrict__ d_conv1_biases_grad,
            const int* __restrict__ d_step,
            int is_training)
        {
            #define CONV1_BATCH_PER_CHUNK (BATCH_SIZE / CONV1_CHUNKS)

            const int filter_idx = blockIdx.x / CONV1_CHUNKS;
            const int chunk_idx = blockIdx.x % CONV1_CHUNKS;
            const int tid = threadIdx.y * 16 + threadIdx.x;

            const int conv1_out_w = POOL1_SIZE * 2;
            const int conv1_out_per_sample =
                POOL1_SIZE * POOL1_SIZE * CONV1_FILTER_COUNT;
            const int filter_size_sq =
                FILTER1_SIZE * FILTER1_SIZE;

            __shared__ __half s_filter_grad[FILTER1_SIZE * FILTER1_SIZE];
            __shared__ __half s_bias_grad;
            __shared__ __half s_grad[POOL1_SIZE * 2][POOL1_SIZE * 2];
            __shared__ uint32_t s_image[28];
            __shared__ __half s_reduce[256];

            __half zero = __float2half(0.0f);

            if (tid < filter_size_sq)
            {
                s_filter_grad[tid] = zero;
            }
            if (tid == 0)
            {
                s_bias_grad = zero;
            }
            __syncthreads();

            const int batchOffset =
                ((*d_step) % BATCHES_PER_EPOCH) * BATCH_SIZE;

            const int start_b = chunk_idx * CONV1_BATCH_PER_CHUNK;
            const int end_b = start_b + CONV1_BATCH_PER_CHUNK;

            // Mapping for weight gradient parallel calculation
            int f = tid / 8;
            int s = tid % 8;
            int fy = f / FILTER1_SIZE;
            int fx = f % FILTER1_SIZE;

            for (int b = start_b; b < end_b; b++)
            {
                if (tid < 28)
                {
                    s_image[tid] =
                        d_inputs[(batchOffset + b) * 28 + tid];
                }

                for (int i = tid; i < conv1_out_w * conv1_out_w; i += 256)
                {
                    int gy = i / conv1_out_w;
                    int gx = i % conv1_out_w;
                    int px2 = gx / 2;
                    int py2 = gy / 2;
                    int pool_idx = (py2 * POOL1_SIZE + px2)
                        * CONV1_FILTER_COUNT + filter_idx;

                    __half out_grad = d_conv1_out_grad[
                        b * conv1_out_per_sample + pool_idx];
                    __half out_val = d_conv1_out_val[
                        b * conv1_out_per_sample + pool_idx];
                    __half my_val = d_conv1_unpooled_vals[
                        (b * conv1_out_w * conv1_out_w
                        + gy * conv1_out_w + gx)
                        * CONV1_FILTER_COUNT + filter_idx];

                    __half grad = zero;
                    if (__heq(ACTIVATION_FWD(my_val), out_val))
                    {
                        grad = ACTIVATION_BWD(my_val, out_grad);
                    }
                    s_grad[gy][gx] = grad;
                }
                __syncthreads();

                int seed = b + *d_step;
                int dx = (is_training == 1)
                    ? ((seed * 1103515245 + 12345) % 3 - 1) : 0;
                int dy = (is_training == 1)
                    ? (((seed * 1103515245 + 12345) / 3) % 3 - 1) : 0;

                __half w_grad = zero;
                if (f < 25)
                {
                    int y_start = s * 3;
                    int y_end = y_start + 3;
                    #pragma unroll
                    for (int y = y_start; y < y_end; y++)
                    {
                        int shift_y = y + fy + dy;
                        uint32_t row_bits = 0;
                        if (shift_y >= 0 && shift_y < 28)
                        {
                            row_bits = s_image[shift_y];
                        }
                        #pragma unroll
                        for (int x = 0; x < conv1_out_w; x++)
                        {
                            int img_x = x + fx + dx;
                            uint32_t pixel = 0;
                            if (img_x >= 0 && img_x < 28)
                            {
                                pixel = (row_bits >> img_x) & 1u;
                            }
                            if (pixel != 0)
                            {
                                w_grad = __hadd(w_grad, s_grad[y][x]);
                            }
                        }
                    }
                }
                s_reduce[tid] = w_grad;
                __syncthreads();

                if (tid < 25)
                {
                    __half final_w_grad = zero;
                    #pragma unroll
                    for (int j = 0; j < 8; j++)
                    {
                        final_w_grad = __hadd(final_w_grad, s_reduce[tid * 8 + j]);
                    }
                    s_filter_grad[tid] += final_w_grad;
                }

                // Parallel bias gradient calculation
                __half my_bias = zero;
                for (int i = tid; i < conv1_out_w * conv1_out_w; i += 256)
                {
                    my_bias = __hadd(my_bias, s_grad[i / conv1_out_w][i % conv1_out_w]);
                }
                s_reduce[tid] = my_bias;
                __syncthreads();

                for (int stride = 128; stride > 0; stride >>= 1)
                {
                    if (tid < stride)
                    {
                        s_reduce[tid] = __hadd(s_reduce[tid], s_reduce[tid + stride]);
                    }
                    __syncthreads();
                }

                if (tid == 0)
                {
                    s_bias_grad = __hadd(s_bias_grad, s_reduce[0]);
                }
                __syncthreads();
            }

            if (tid < filter_size_sq)
            {
                atomicAdd(&d_conv1_filters_grad[
                    filter_idx * filter_size_sq + tid],
                    s_filter_grad[tid]);
            }
            if (tid == 0)
            {
                atomicAdd(&d_conv1_biases_grad[filter_idx],
                    s_bias_grad);
            }
        }
        """;

    public static readonly string AdamUpdate =
        """

        extern "C" __global__ void adam_update(
            __half* __restrict__ d_param,
            __half* __restrict__ d_grad,
            __half* __restrict__ d_m,
            __half* __restrict__ d_v,
            int num_elements,
            int* __restrict__ d_step)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;

            int step_val = *d_step + 1;

            #ifndef MAX_LR
            #define MAX_LR 0.006f
            #endif
            __half max_lr = __float2half(MAX_LR);
            __half beta1 = __float2half(0.7f);
            __half beta2 = __float2half(0.9f);
            __half epsilon = __float2half(1e-4f);

            int total_steps = TOTAL_STEPS;

            __shared__ __half s_lr;
            __shared__ __half s_beta1_t;
            __shared__ __half s_beta2_t;

            if (threadIdx.x == 0)
            {
                float f_beta1_t = powf(0.7f, (float)step_val);
                float f_beta2_t = powf(0.9f, (float)step_val);

                float pct = (float)step_val / total_steps;
                float warmup_pct = 0.20f;
                float local_lr = 0.0f;
                if (pct < warmup_pct)
                {
                    float alpha = pct / warmup_pct;
                    local_lr = __half2float(max_lr) * (0.1f + 0.9f * alpha);
                }
                else
                {
                    float alpha = (pct - warmup_pct) / (1.0f - warmup_pct);
                    float cos_val = cosf(3.14159265f * alpha);
                    local_lr = __half2float(max_lr) * 0.5f * (1.0f + cos_val);
                }
                s_lr = __float2half(local_lr);
                s_beta1_t = __float2half(f_beta1_t);
                s_beta2_t = __float2half(f_beta2_t);
            }
            __syncthreads();

            __half lr = s_lr;
            __half beta1_t = s_beta1_t;
            __half beta2_t = s_beta2_t;

            __half h_one_minus_beta1 = __float2half(0.3f);
            __half h_one_minus_beta2 = __float2half(0.1f);
            __half h_one_minus_beta1_t = __hsub(__float2half(1.0f), beta1_t);
            __half h_one_minus_beta2_t = __hsub(__float2half(1.0f), beta2_t);
            __half h_batch_size = __float2half((float)BATCH_SIZE);

            for (int i = tid; i < num_elements; i += stride)
            {
                __half g = __hdiv(d_grad[i], h_batch_size);
                __half m = __hadd(__hmul(beta1, d_m[i]), __hmul(h_one_minus_beta1, g));
                __half v = __hadd(__hmul(beta2, d_v[i]), __hmul(h_one_minus_beta2, __hmul(g, g)));

                d_m[i] = m;
                d_v[i] = v;

                __half m_hat = __hdiv(m, h_one_minus_beta1_t);
                __half v_hat = __hdiv(v, h_one_minus_beta2_t);

                __half param_val = d_param[i];
                
                #ifndef WEIGHT_DECAY
                #define WEIGHT_DECAY 0.0f
                #endif
                
                if (WEIGHT_DECAY > 0.0f)
                {
                    __half wd = __float2half(WEIGHT_DECAY);
                    param_val = __hsub(param_val, __hmul(__hmul(lr, wd), param_val));
                }
                
                __half denom = __hadd(hsqrt(v_hat), epsilon);
                param_val = __hsub(param_val, __hdiv(__hmul(lr, m_hat), denom));
                
                d_param[i] = param_val;
                d_grad[i] = __float2half(0.0f);
            }

            if (threadIdx.x == 0 && blockIdx.x == 0)
            {
                *d_step = step_val;
            }
        }
        """;

    public static readonly string QuantizeAllWeights =
        """

        __device__ __forceinline__ __half quantize_to_fp4_val(__half h_val, __half scale)
        {
            __half val = __hmul(h_val, scale);
            __half abs_val = __habs(val);
            __half quant_val;
            __half h_2_5 = __float2half(2.5f);
            __half h_2_0 = __float2half(2.0f);
            __half h_0_5 = __float2half(0.5f);
            __half h_5_0 = __float2half(5.0f);
            __half h_3_5 = __float2half(3.5f);
            __half h_3_0 = __float2half(3.0f);
            __half h_4_0 = __float2half(4.0f);
            __half h_6_0 = __float2half(6.0f);

            if (__hlt(abs_val, h_2_5))
            {
                quant_val = __hmul(hrint(__hmul(abs_val, h_2_0)), h_0_5);
            }
            else if (__hlt(abs_val, h_5_0))
            {
                quant_val = __hlt(abs_val, h_3_5) ? h_3_0 : h_4_0;
            }
            else
            {
                quant_val = h_6_0;
            }
            __half sign = __hlt(h_val, __float2half(0.0f)) ? __float2half(-1.0f) : __float2half(1.0f);
            return __hdiv(__hmul(sign, quant_val), scale);
        }

        extern "C" __global__ void quantize_all_weights(
            const __half* __restrict__ d_params,
            __half* __restrict__ d_quant_params)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;

            for (int i = tid; i < 29066; i += stride)
            {
                __half h_val = d_params[i];
                if (i < 200) // conv1 weights
                {
                    d_quant_params[i] = quantize_to_fp4_val(h_val, __float2half(8.0f));
                }
                else if (i < 208) // conv1 biases
                {
                    d_quant_params[i] = h_val;
                }
                else if (i < 3408) // conv2 weights
                {
                    d_quant_params[i] = quantize_to_fp4_val(h_val, __float2half(32.0f));
                }
                else if (i < 3424) // conv2 biases
                {
                    d_quant_params[i] = h_val;
                }
                else if (i < 28000) // fc1 weights
                {
                    d_quant_params[i] = quantize_to_fp4_val(h_val, __float2half(256.0f));
                }
                else if (i < 28096) // fc1 biases
                {
                    d_quant_params[i] = h_val;
                }
                else if (i < 29056) // fc2 weights
                {
                    d_quant_params[i] = quantize_to_fp4_val(h_val, __float2half(16.0f));
                }
                else // fc2 biases
                {
                    d_quant_params[i] = h_val;
                }
            }
        }
        """;

    public static string BuildLeNetSource(NetworkConfig config)
    {
        var sb = new System.Text.StringBuilder(32768);
        sb.AppendLine(Preamble);

        if (config.ActivationType == "SILU")
        {
            sb.AppendLine("#define ACTIVATION_FWD(x) silu(x)");
            sb.AppendLine("#define ACTIVATION_BWD(x, dy) d_silu(x, dy)");
        }
        else
        {
            sb.AppendLine("#define ACTIVATION_FWD(x) gelu(x)");
            sb.AppendLine("#define ACTIVATION_BWD(x, dy) d_gelu(x, dy)");
        }

        // Architecture macros
        sb.AppendLine($"#define CONV1_FILTER_COUNT {config.Conv1FilterCount}");
        sb.AppendLine($"#define CONV2_FILTER_COUNT {config.Conv2FilterCount}");
        sb.AppendLine($"#define HAS_DROPOUT {(config.HasDropout ? 1 : 0)}");
        sb.AppendLine($"#define DROPOUT_RATE {config.DropoutRate}f");
        sb.AppendLine($"#define USE_HALF2 {(config.IsHalf ? 1 : 0)}");

        if (config.IsGlobalAveragePooling)
        {
            sb.AppendLine("#define IS_GAP 1");
        }
        else
        {
            sb.AppendLine("#define IS_GAP 0");
        }

        int fc2Inputs = config.HasFC1 ? config.FC1Outputs : (config.Conv2FilterCount * config.Pool2OutSize * config.Pool2OutSize);
        sb.AppendLine($"#define FC2_INPUTS {fc2Inputs}");

        sb.Append(ClearGradient);

        if (config.ActivationType == "SILU")
        {
            sb.Append(ActivationSilu);
        }
        else
        {
            sb.Append(ActivationGelu);
        }
        sb.Append(DummyFusedKernels);
        if (config.IsFusedForward)
        {
            sb.Append(CudaSourceFused.FusedForward);
        }
        else
        {
            sb.Append(Conv1Forward);
            sb.Append(Conv2Forward);
            if (config.HasFC1) sb.Append(Fc1Forward);
            sb.Append(Fc2Forward);
        }
        sb.Append(Fc2Backward);
        sb.Append(Fc2BackwardWeights);
        if (config.HasFC1)
        {
            sb.Append(Fc1Backward);
            sb.Append(Fc1BackwardWeights);
        }
        sb.Append(Conv2Backward);
        sb.Append(Conv1Backward);
        sb.Append(AdamUpdate);
        if (config.Name == "FP4")
        {
            sb.Append(QuantizeAllWeights);
        }

        return sb.ToString();
    }
}
