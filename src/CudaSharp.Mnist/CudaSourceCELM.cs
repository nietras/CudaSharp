namespace CudaSharp.Mnist;

public static partial class Program
{
    public static string CudaSourceCELM =>
        """
        #include <cuda_fp16.h>
        #include <mma.h>
        using namespace nvcuda;

        typedef unsigned int uint32_t;

        #ifndef BATCH_SIZE
        #define BATCH_SIZE 128
        #endif
        #define FILTER1_SIZE 5
        #define FILTER2_SIZE 5
        #define INPUT_SIZE 28
        #define POOL1_SIZE 12
        #define POOL2_SIZE 4
        #define FC1_INPUTS 256
        #define FC1_OUTPUTS 128
        #define FC2_OUTPUTS 10

        #ifndef BATCHES_PER_EPOCH
        #define BATCHES_PER_EPOCH 400
        #endif
        #ifndef TOTAL_STEPS
        #define TOTAL_STEPS 155
        #endif

        extern "C" __global__ void clear_gradient(__half* __restrict__ d_grad, int num_elements)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            __half zero = __float2half(0.0f);
            for (int i = tid; i < num_elements; i += stride)
            {
                d_grad[i] = zero;
            }
        }

        __device__ inline __half gelu(__half x)
        {
            float val = __half2float(x);
            float c = 0.79788456f;
            float tanh_arg = c * (val + 0.044715f * val * val * val);
            float g = 0.5f * val * (1.0f + tanhf(tanh_arg));
            return __float2half(g);
        }

        __device__ inline __half d_gelu(__half x, __half dy)
        {
            float val = __half2float(x);
            float g_dy = __half2float(dy);
            float c = 0.79788456f;
            float tanh_arg = c * (val + 0.044715f * val * val * val);
            float t = tanhf(tanh_arg);
            float sech2 = 1.0f - t * t;
            float dtanh = c * (1.0f + 3.0f * 0.044715f * val * val) * sech2;
            float derivative = 0.5f * (1.0f + t) + 0.5f * val * dtanh;
            return __float2half(g_dy * derivative);
        }

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

            if (out_x >= 12 || out_y >= 12) return;

            int batchOffset = ((*d_step) % BATCHES_PER_EPOCH) * BATCH_SIZE;

            __shared__ __half s_filter[5][5];
            int tid_flat = threadIdx.y * 12 + threadIdx.x;
            if (tid_flat < 25)
            {
                s_filter[tid_flat / 5][tid_flat % 5] = d_filters[filter_idx * 25 + tid_flat];
            }
            
            __shared__ uint32_t s_image[28];
            if (tid_flat < 28)
            {
                s_image[tid_flat] = d_inputs[(batchOffset + batch_idx) * 28 + tid_flat];
            }
            __syncthreads();

            const int conv_x_base = out_x * 2;
            const int conv_y_base = out_y * 2;

            int seed = batch_idx + *d_step;
            int dx = (is_training == 1) ? ((seed * 1103515245 + 12345) % 3 - 1) : 0;
            int dy = (is_training == 1) ? (((seed * 1103515245 + 12345) / 3) % 3 - 1) : 0;

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
                    for (int fy = 0; fy < 5; fy++)
                    {
                        int shift_y = cy + fy + dy;
                        uint32_t row_bits = 0;
                        if (shift_y >= 0 && shift_y < 28)
                        {
                            row_bits = s_image[shift_y];
                        }
                        #pragma unroll
                        for (int fx = 0; fx < 5; fx++)
                        {
                            int img_x = cx + fx + dx;
                            uint32_t pixel = 0;
                            if (img_x >= 0 && img_x < 28)
                            {
                                pixel = (row_bits >> img_x) & 1u;
                            }
                            sum += __half((float)pixel) * s_filter[fy][fx];
                        }
                    }

                    int unpooled_idx = (batch_idx * 576 + cy * 24 + cx) * 16 + filter_idx;
                    d_unpooled_vals[unpooled_idx] = sum;

                    __half activated = gelu(sum);
                    if (__hgt(activated, max_val))
                    {
                        max_val = activated;
                    }
                }
            }

            const int out_idx = batch_idx * 2304 + (out_y * 12 + out_x) * 16 + filter_idx;
            d_outputs[out_idx] = max_val;
        }

        extern "C" __global__ void conv2_forward(
            const __half* __restrict__ d_inputs,
            const __half* __restrict__ d_filters,
            const __half* __restrict__ d_biases,
            __half* __restrict__ d_outputs,
            __half* __restrict__ d_unpooled_vals)
        {
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x;

            __shared__ __half s_input[2304]; 
            __shared__ __half s_filters[6400]; // 16 * 16 * 25 = 6400

            for (int i = tid; i < 2304; i += 256)
            {
                s_input[i] = d_inputs[batch_idx * 2304 + i];
            }

            for (int i = tid; i < 6400; i += 256)
            {
                s_filters[i] = d_filters[i];
            }
            __syncthreads();

            #pragma unroll
            for (int out_idx = tid; out_idx < 256; out_idx += 256)
            {
                int filter_idx = out_idx / 16;
                int spatial_idx = out_idx % 16;
                int out_x = spatial_idx % 4;
                int out_y = spatial_idx / 4;

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
                        for (int c = 0; c < 16; c++)
                        {
                            #pragma unroll
                            for (int fy = 0; fy < 5; fy++)
                            {
                                #pragma unroll
                                for (int fx = 0; fx < 5; fx++)
                                {
                                    int in_x = cx + fx;
                                    int in_y = cy + fy;
                                    sum += s_input[(in_y * 12 + in_x) * 16 + c] * s_filters[filter_idx * 400 + (fy * 5 + fx) * 16 + c];
                                }
                            }
                        }

                        int unpooled_idx = (batch_idx * 64 + cy * 8 + cx) * 16 + filter_idx;
                        d_unpooled_vals[unpooled_idx] = sum;

                        __half activated = gelu(sum);
                        if (__hgt(activated, max_val))
                        {
                            max_val = activated;
                        }
                    }
                }

                const int out_idx_global = batch_idx * 256 + (out_y * 4 + out_x) * 16 + filter_idx;
                d_outputs[out_idx_global] = max_val;
            }
        }

        // TENSOR CORE WMMA FC1 Forward
        // 256 -> 128
        extern "C" __global__ void fc1_forward(
            const __half* __restrict__ d_inputs,
            const __half* __restrict__ d_weights,
            const __half* __restrict__ d_biases,
            __half* __restrict__ d_outputs,
            __half* __restrict__ d_unpooled_vals)
        {
            const int batch_idx = blockIdx.x; 
            const int tid = threadIdx.x;

            __shared__ __half s_input[256];
            
            if (tid < 256) {
                s_input[tid] = d_inputs[batch_idx * 256 + tid];
            }
            __syncthreads();

            for (int out_idx = tid; out_idx < FC1_OUTPUTS; out_idx += blockDim.x)
            {
                __half sum = d_biases[out_idx];
                #pragma unroll 4
                for (int i = 0; i < 256; i++)
                {
                    sum += s_input[i] * d_weights[i * FC1_OUTPUTS + out_idx];
                }
                d_unpooled_vals[batch_idx * FC1_OUTPUTS + out_idx] = sum;
                d_outputs[batch_idx * FC1_OUTPUTS + out_idx] = gelu(sum);
            }
        }

        extern "C" __global__ void fc2_forward(
            const __half* __restrict__ d_inputs,
            const __half* __restrict__ d_weights,
            const __half* __restrict__ d_biases,
            __half* __restrict__ d_outputs)
        {
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x;

            __shared__ __half s_input[4096]; // Max FC1_OUTPUTS
            for (int i = tid; i < FC1_OUTPUTS; i += blockDim.x)
            {
                s_input[i] = d_inputs[batch_idx * FC1_OUTPUTS + i];
            }
            __syncthreads();

            if (tid < 10)
            {
                __half sum = d_biases[tid];
                #pragma unroll 4
                for (int i = 0; i < FC1_OUTPUTS; i++)
                {
                    sum += s_input[i] * d_weights[i * 10 + tid];
                }
                d_outputs[batch_idx * 10 + tid] = sum;
            }
        }

        extern "C" __global__ void fc2_backward(
            const __half* __restrict__ d_fc2_outputs,
            const int* __restrict__ d_labels,
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
                    float logit = __half2float(d_fc2_outputs[batch_idx * 10 + i]);
                    if (logit > max_logit) max_logit = logit;
                }

                float sum_exp = 0.0f;
                for (int i = 0; i < 10; i++)
                {
                    sum_exp += expf(__half2float(d_fc2_outputs[batch_idx * 10 + i]) - max_logit);
                }

                float prob = expf(__half2float(d_fc2_outputs[batch_idx * 10 + tid]) - max_logit) / sum_exp;
                int correct_label = d_labels[batchOffset + batch_idx];
                s_grad[tid] = __float2half(prob - (tid == correct_label ? 1.0f : 0.0f));
            }
            __syncthreads();

            if (tid < 128)
            {
                __half sum_input_grad = __float2half(0.0f);
                #pragma unroll
                for (int c = 0; c < 10; c++)
                {
                    sum_input_grad += s_grad[c] * d_fc2_weights[tid * 10 + c];
                }
                __half fc1_unpooled = d_fc1_unpooled[batch_idx * 128 + tid];
                d_fc1_out_grad[batch_idx * 128 + tid] = d_gelu(fc1_unpooled, sum_input_grad);
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

        extern "C" __global__ void fc2_backward_weights(
            const __half* __restrict__ d_fc2_outputs,
            const int* __restrict__ d_labels,
            const __half* __restrict__ d_fc1_outputs,
            __half* __restrict__ d_fc2_weights_grad,
            const int* __restrict__ d_step)
        {
            const int input_idx = blockIdx.x; // 0..127
            const int tid = threadIdx.x;      // 0..127

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

            __shared__ float s_weight_grads[128][10];

            #pragma unroll
            for (int c = 0; c < 10; c++)
            {
                s_weight_grads[tid][c] = 0.0f;
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
                float x_val = __half2float(d_fc1_outputs[b * 128 + input_idx]);

                #pragma unroll
                for (int c = 0; c < 10; c++)
                {
                    float prob = expf(__half2float(s_fc2_outputs[b][c]) - max_logit) / sum_exp;
                    float g_val = prob - (c == correct_label ? 1.0f : 0.0f);
                    s_weight_grads[tid][c] += g_val * x_val;
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
                        s_weight_grads[tid][c] += s_weight_grads[tid + stride][c];
                    }
                }
                __syncthreads();
            }

            if (tid == 0)
            {
                #pragma unroll
                for (int c = 0; c < 10; c++)
                {
                    d_fc2_weights_grad[input_idx * 10 + c] = __float2half(s_weight_grads[0][c]);
                }
            }
        }

        extern "C" __global__ void fc1_backward(
            const __half* __restrict__ d_fc1_out_grad,
            const __half* __restrict__ d_fc1_outputs,
            const __half* __restrict__ d_fc2_inputs,
            const __half* __restrict__ d_fc1_weights,
            __half* __restrict__ d_fc1_biases_grad,
            __half* __restrict__ d_conv2_out_grad)
        {
            // CELM: Frozen layer, no backward pass required.
        }

        extern "C" __global__ void fc1_backward_weights(
            const __half* __restrict__ d_fc1_out_grad,
            const __half* __restrict__ d_fc1_outputs,
            const __half* __restrict__ d_fc1_inputs,
            __half* __restrict__ d_fc1_weights_grad)
        {
            // CELM: Frozen layer, no backward pass required.
        }

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
            // CELM: Frozen layer, no backward pass required.
        }

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
            // CELM: Frozen layer, no backward pass required.
        }

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
            #define MAX_LR 0.014f
            #endif
            float max_lr = MAX_LR; 
            float beta1 = 0.7f;
            float beta2 = 0.9f;
            float epsilon = 1e-4f;
            
            int total_steps = TOTAL_STEPS;
            
            __shared__ float s_lr;
            __shared__ float s_beta1_t;
            __shared__ float s_beta2_t;
            
            if (threadIdx.x == 0)
            {
                s_beta1_t = powf(beta1, (float)step_val);
                s_beta2_t = powf(beta2, (float)step_val);
                
                float pct = (float)step_val / total_steps;
                float warmup_pct = 0.20f;
                float local_lr = 0.0f;
                if (pct < warmup_pct)
                {
                    float alpha = pct / warmup_pct;
                    local_lr = max_lr * (0.1f + 0.9f * alpha);
                }
                else
                {
                    float alpha = (pct - warmup_pct) / (1.0f - warmup_pct);
                    float cos_val = cosf(3.14159265f * alpha);
                    local_lr = max_lr * 0.5f * (1.0f + cos_val);
                }
                s_lr = local_lr;
            }
            __syncthreads();

            float lr = s_lr;
            float beta1_t = s_beta1_t;
            float beta2_t = s_beta2_t;

            for (int i = tid; i < num_elements; i += stride)
            {
                float g = __half2float(d_grad[i]) / BATCH_SIZE;
                float m = beta1 * __half2float(d_m[i]) + (1.0f - beta1) * g;
                float v = beta2 * __half2float(d_v[i]) + (1.0f - beta2) * g * g;

                d_m[i] = __float2half(m);
                d_v[i] = __float2half(v);

                float m_hat = m / (1.0f - beta1_t);
                float v_hat = v / (1.0f - beta2_t);

                float param_val = __half2float(d_param[i]);
                param_val -= lr * m_hat / (sqrtf(v_hat) + epsilon);
                d_param[i] = __float2half(param_val);
                d_grad[i] = __float2half(0.0f);
            }

            if (threadIdx.x == 0 && blockIdx.x == 0)
            {
                *d_step = step_val;
            }
        }
        """;
}
