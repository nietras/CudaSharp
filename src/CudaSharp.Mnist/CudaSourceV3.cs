namespace CudaSharp.Mnist;

public static partial class Program
{
    public static string CudaSourceV3 =>
        """
        #include <cuda_fp16.h>

        typedef unsigned int uint32_t;
        
        #ifndef BATCH_SIZE
        #define BATCH_SIZE 128
        #endif
        #define FILTER1_SIZE 5
        #define FILTER2_SIZE 5
        #define INPUT_SIZE 28
        #define POOL1_SIZE 12
        #define POOL2_SIZE 4
        #define FC2_INPUTS 256
        #define FC2_OUTPUTS 10

        #ifndef BATCHES_PER_EPOCH
        #define BATCHES_PER_EPOCH 300
        #endif
        #ifndef TOTAL_STEPS
        #define TOTAL_STEPS 600
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
                s_filter[tid_flat / 5][tid_flat % 5] = 
                    d_filters[filter_idx * 25 + tid_flat];
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
            __half zero = __float2half(0.0f);

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

                    int unpooled_idx = (batch_idx * 576 + cy * 24 + cx) * 8 + filter_idx;
                    d_unpooled_vals[unpooled_idx] = sum;

                    __half activated = __hgt(sum, zero) ? sum : zero;
                    if (__hgt(activated, max_val))
                    {
                        max_val = activated;
                    }
                }
            }

            const int out_idx = batch_idx * (12 * 12 * 8) 
                                + (out_y * 12 + out_x) * 8 
                                + filter_idx;
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

            __shared__ __half s_input[1152];
            __shared__ __half s_filters[3200];

            for (int i = tid; i < 1152; i += 256)
            {
                s_input[i] = d_inputs[batch_idx * 1152 + i];
            }

            for (int i = tid; i < 3200; i += 256)
            {
                s_filters[i] = d_filters[i];
            }
            __syncthreads();

            if (tid < 256)
            {
                int filter_idx = tid / 16;
                int spatial_idx = tid % 16;
                int out_x = spatial_idx % 4;
                int out_y = spatial_idx / 4;

                const int conv_x_base = out_x * 2;
                const int conv_y_base = out_y * 2;

                __half max_val = __float2half(-1e9f);
                __half zero = __float2half(0.0f);

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
                        for (int c = 0; c < 8; c++)
                        {
                            #pragma unroll
                            for (int fy = 0; fy < 5; fy++)
                            {
                                #pragma unroll
                                for (int fx = 0; fx < 5; fx++)
                                {
                                    int in_x = cx + fx;
                                    int in_y = cy + fy;
                                    sum += s_input[(in_y * 12 + in_x) * 8 + c] * s_filters[filter_idx * 200 + (fy * 5 + fx) * 8 + c];
                                }
                            }
                        }

                        int unpooled_idx = (batch_idx * 64 + cy * 8 + cx) * 16 + filter_idx;
                        d_unpooled_vals[unpooled_idx] = sum;

                        __half activated = __hgt(sum, zero) ? sum : zero;
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

        extern "C" __global__ void fc2_forward(
            const __half* __restrict__ d_inputs,
            const __half* __restrict__ d_weights,
            const __half* __restrict__ d_biases,
            __half* __restrict__ d_outputs)
        {
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x;

            __shared__ __half s_input[256];
            s_input[tid] = d_inputs[batch_idx * 256 + tid];
            __syncthreads();

            if (tid < 10)
            {
                __half sum = d_biases[tid];
                #pragma unroll 4
                for (int i = 0; i < 256; i++)
                {
                    sum += s_input[i] * d_weights[i * 10 + tid];
                }
                d_outputs[batch_idx * 10 + tid] = sum;
            }
        }

        extern "C" __global__ void fc2_backward(
            const __half* __restrict__ d_fc2_outputs,
            const int* __restrict__ d_labels,
            const __half* __restrict__ d_fc2_inputs,
            const __half* __restrict__ d_fc2_weights,
            __half* __restrict__ d_fc2_weights_grad,
            __half* __restrict__ d_fc2_biases_grad,
            __half* __restrict__ d_fc2_in_grad,
            const int* __restrict__ d_step)
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

            __half x_val = d_fc2_inputs[batch_idx * 256 + tid];

            __half sum_input_grad = __float2half(0.0f);
            #pragma unroll
            for (int c = 0; c < 10; c++)
            {
                sum_input_grad += s_grad[c] * d_fc2_weights[tid * 10 + c];
            }
            d_fc2_in_grad[batch_idx * 256 + tid] = sum_input_grad;

            if (__half2float(x_val) != 0.0f)
            {
                #pragma unroll
                for (int c = 0; c < 10; c++)
                {
                    __half g_val = s_grad[c];
                    if (__half2float(g_val) != 0.0f)
                    {
                        atomicAdd(&d_fc2_weights_grad[tid * 10 + c], g_val * x_val);
                    }
                }
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
            #ifndef CONV2_CHUNKS
            #define CONV2_CHUNKS 16
            #endif
            #define CONV2_BATCH_PER_CHUNK (BATCH_SIZE / CONV2_CHUNKS)

            const int filter_idx = blockIdx.x / CONV2_CHUNKS;
            const int chunk_idx = blockIdx.x % CONV2_CHUNKS;
            const int tid = threadIdx.x;

            __shared__ __half s_conv1_out[1152];
            __shared__ __half s_filter_grad[200];
            __shared__ __half s_bias_grad;
            __shared__ __half s_grad[8][8];

            __half zero = __float2half(0.0f);

            for (int i = tid; i < 200; i += 128)
            {
                s_filter_grad[i] = zero;
            }
            if (tid == 0)
            {
                s_bias_grad = zero;
            }
            __syncthreads();

            const int cx = (tid < 64) ? (tid % 8) : 0;
            const int cy = (tid < 64) ? (tid / 8) : 0;
            const int px = cx / 2;
            const int py = cy / 2;
            const int pool_idx = (py * 4 + px) * 16 + filter_idx;

            const int start_b = chunk_idx * CONV2_BATCH_PER_CHUNK;
            const int end_b = start_b + CONV2_BATCH_PER_CHUNK;

            for (int b = start_b; b < end_b; b++)
            {
                for (int i = tid; i < 1152; i += 128)
                {
                    s_conv1_out[i] = d_conv1_out[b * 1152 + i];
                }

                __half out_grad = d_conv2_out_grad[b * 256 + pool_idx];
                __half out_val = d_conv2_out_val[b * 256 + pool_idx];

                __half my_val = zero;
                if (tid < 64)
                {
                    my_val = d_conv2_unpooled_vals[(b * 64 + tid) * 16 + filter_idx];
                }

                __half grad = zero;
                if (tid < 64 && __heq(my_val, out_val) && __hgt(out_val, zero))
                {
                    grad = out_grad;
                }
                if (tid < 64)
                {
                    s_grad[cy][cx] = grad;
                }
                __syncthreads();

                for (int i = tid; i < 200; i += 128)
                {
                    int c = i % 8;
                    int fx = (i / 8) % 5;
                    int fy = i / 40;

                    __half w_grad = zero;
                    #pragma unroll
                    for (int y = 0; y < 8; y++)
                    {
                        #pragma unroll
                        for (int x = 0; x < 8; x++)
                        {
                            __half g = s_grad[y][x];
                            if (__half2float(g) != 0.0f)
                            {
                                int in_x = x + fx;
                                int in_y = y + fy;
                                w_grad += g * s_conv1_out[(in_y * 12 + in_x) * 8 + c];
                            }
                        }
                    }
                    s_filter_grad[i] += w_grad;
                }

                if (tid == 0)
                {
                    __half b_grad = zero;
                    for (int y = 0; y < 8; y++)
                    {
                        for (int x = 0; x < 8; x++)
                        {
                            b_grad += s_grad[y][x];
                        }
                    }
                    s_bias_grad += b_grad;
                }

                for (int i = tid; i < 1152; i += 128)
                {
                    int c = i % 8;
                    int ix = (i / 8) % 12;
                    int iy = i / 96;

                    __half sum_grad = zero;
                    #pragma unroll
                    for (int fy = 0; fy < 5; fy++)
                    {
                        #pragma unroll
                        for (int fx = 0; fx < 5; fx++)
                        {
                            int x = ix - fx;
                            int y = iy - fy;
                            if (x >= 0 && x < 8 && y >= 0 && y < 8)
                            {
                                int f_idx = filter_idx * 200 + (fy * 5 + fx) * 8 + c;
                                sum_grad += s_grad[y][x] * d_conv2_filters[f_idx];
                            }
                        }
                    }
                    if (__half2float(sum_grad) != 0.0f)
                    {
                        atomicAdd(&d_conv1_out_grad[b * 1152 + i], sum_grad);
                    }
                }

                __syncthreads();
            }

            for (int i = tid; i < 200; i += 128)
            {
                atomicAdd(&d_conv2_filters_grad[filter_idx * 200 + i], s_filter_grad[i]);
            }
            if (tid == 0)
            {
                atomicAdd(&d_conv2_biases_grad[filter_idx], s_bias_grad);
            }
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
            #ifndef CONV1_CHUNKS
            #define CONV1_CHUNKS 16
            #endif
            #define CONV1_BATCH_PER_CHUNK (BATCH_SIZE / CONV1_CHUNKS)

            const int filter_idx = blockIdx.x / CONV1_CHUNKS;
            const int chunk_idx = blockIdx.x % CONV1_CHUNKS;
            const int cx = threadIdx.x;
            const int cy = threadIdx.y;
            const int tid = cy * 24 + cx;

            __shared__ __half s_filter_grad[25];
            __shared__ __half s_bias_grad;
            __shared__ __half s_grad[24][24];
            __shared__ uint32_t s_image[28];

            __half zero = __float2half(0.0f);

            if (tid < 25)
            {
                s_filter_grad[tid] = zero;
            }
            if (tid == 0)
            {
                s_bias_grad = zero;
            }
            __syncthreads();

            const int batchOffset = ((*d_step) % BATCHES_PER_EPOCH) * BATCH_SIZE;

            const int px = cx / 2;
            const int py = cy / 2;
            const int pool_idx = (py * 12 + px) * 8 + filter_idx;

            const int start_b = chunk_idx * CONV1_BATCH_PER_CHUNK;
            const int end_b = start_b + CONV1_BATCH_PER_CHUNK;

            for (int b = start_b; b < end_b; b++)
            {
                if (tid < 28)
                {
                    s_image[tid] = d_inputs[(batchOffset + b) * 28 + tid];
                }

                __half out_grad = d_conv1_out_grad[b * 1152 + pool_idx];
                __half out_val = d_conv1_out_val[b * 1152 + pool_idx];
                __half my_val = d_conv1_unpooled_vals[(b * 576 + cy * 24 + cx) * 8 + filter_idx];

                __half grad = zero;
                if (__heq(my_val, out_val) && __hgt(out_val, zero))
                {
                    grad = out_grad;
                }
                s_grad[cy][cx] = grad;
                __syncthreads();

                int seed = b + *d_step;
                int dx = (is_training == 1) ? ((seed * 1103515245 + 12345) % 3 - 1) : 0;
                int dy = (is_training == 1) ? (((seed * 1103515245 + 12345) / 3) % 3 - 1) : 0;

                if (tid < 25)
                {
                    int fx = tid % 5;
                    int fy = tid / 5;

                    __half w_grad = zero;
                    #pragma unroll
                    for (int y = 0; y < 24; y++)
                    {
                        int shift_y = y + fy + dy;
                        uint32_t row_bits = 0;
                        if (shift_y >= 0 && shift_y < 28)
                        {
                            row_bits = s_image[shift_y];
                        }
                        #pragma unroll
                        for (int x = 0; x < 24; x++)
                        {
                            int img_x = x + fx + dx;
                            uint32_t pixel = 0;
                            if (img_x >= 0 && img_x < 28)
                            {
                                pixel = (row_bits >> img_x) & 1u;
                            }
                            w_grad += __half((float)pixel) * s_grad[y][x];
                        }
                    }
                    s_filter_grad[tid] += w_grad;
                }

                if (tid == 0)
                {
                    __half b_grad = zero;
                    for (int y = 0; y < 24; y++)
                    {
                        for (int x = 0; x < 24; x++)
                        {
                            b_grad += s_grad[y][x];
                        }
                    }
                    s_bias_grad += b_grad;
                }
                __syncthreads();
            }

            if (tid < 25)
            {
                atomicAdd(&d_conv1_filters_grad[filter_idx * 25 + tid], s_filter_grad[tid]);
            }
            if (tid == 0)
            {
                atomicAdd(&d_conv1_biases_grad[filter_idx], s_bias_grad);
            }
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
            #define MAX_LR 0.06f
            #endif
            float max_lr = MAX_LR; 
            float beta1 = 0.7f;
            float beta2 = 0.9f;
            float epsilon = 1e-8f;
            
            int total_steps = TOTAL_STEPS;
            float pct = (float)step_val / total_steps;
            float start_lr = max_lr / 25.0f;
            float end_lr = max_lr / 1000.0f;
            float peak_pct = 0.3f;
            
            float lr = 0.0f;
            if (pct < peak_pct)
            {
                float phase_pct = pct / peak_pct;
                float cos_val = cosf(3.14159265f * phase_pct);
                lr = start_lr + 0.5f * (max_lr - start_lr) * (1.0f - cos_val);
            }
            else
            {
                float phase_pct = (pct - peak_pct) / (1.0f - peak_pct);
                float cos_val = cosf(3.14159265f * phase_pct);
                lr = end_lr + 0.5f * (max_lr - end_lr) * (1.0f + cos_val);
            }

            float beta1_t = powf(beta1, step_val);
            float beta2_t = powf(beta2, step_val);

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
