namespace CudaSharp.Mnist;

public static partial class Program
{
    public static string CudaSourceV2 =>
        """
        typedef unsigned int uint32_t;
        
        #define BATCH_SIZE 128
        #define FILTER1_SIZE 5
        #define FILTER2_SIZE 5
        #define INPUT_SIZE 28
        #define POOL1_SIZE 12
        #define POOL2_SIZE 4
        #define FC2_INPUTS 256
        #define FC2_OUTPUTS 10

        #define BATCHES_PER_EPOCH 300
        #define TOTAL_STEPS 600

        // Helper to clear a gradient buffer asynchronously
        extern "C" __global__ void clear_gradient(float* __restrict__ d_grad, int num_elements)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            for (int i = tid; i < num_elements; i += stride)
            {
                d_grad[i] = 0.0f;
            }
        }

        // Fused Layer 1: 1-Bit Packed Conv1 + MaxPool1 + ReLU (with random shift augmentation)
        extern "C" __global__ void conv1_forward(
            const uint32_t* __restrict__ d_inputs,   // [TotalImages x 28] packed uints
            const float* __restrict__ d_filters,     // [8 x 5 x 5]
            const float* __restrict__ d_biases,      // [8]
            float* __restrict__ d_outputs,           // [BatchSize x 12 x 12 x 8]
            float* __restrict__ d_unpooled_vals,     // [BatchSize x 8 x 24 x 24]
            const int* __restrict__ d_step,
            int is_training)
        {
            const int batch_idx = blockIdx.x;
            const int filter_idx = blockIdx.y; // 0..7
            const int out_x = threadIdx.x;      // 0..11
            const int out_y = threadIdx.y;      // 0..11

            if (out_x >= 12 || out_y >= 12) return;

            int batchOffset = ((*d_step) % BATCHES_PER_EPOCH) * BATCH_SIZE;

            // Load filter weights to shared memory
            __shared__ float s_filter[5][5];
            int tid_flat = threadIdx.y * 12 + threadIdx.x; // 0..143
            if (tid_flat < 25)
            {
                s_filter[tid_flat / 5][tid_flat % 5] = 
                    d_filters[filter_idx * 25 + tid_flat];
            }
            
            // Load packed 1-bit input image rows to shared memory
            __shared__ uint32_t s_image[28];
            if (tid_flat < 28)
            {
                s_image[tid_flat] = d_inputs[(batchOffset + batch_idx) * 28 + tid_flat];
            }
            __syncthreads();

            const int conv_x_base = out_x * 2;
            const int conv_y_base = out_y * 2;

            // Simple pseudo-random shift generator
            int seed = batch_idx + *d_step;
            int dx = (is_training == 1) ? ((seed * 1103515245 + 12345) % 3 - 1) : 0; // -1, 0, or 1
            int dy = (is_training == 1) ? (((seed * 1103515245 + 12345) / 3) % 3 - 1) : 0; // -1, 0, or 1

            float max_val = -1e9f;

            #pragma unroll
            for (int py = 0; py < 2; py++)
            {
                #pragma unroll
                for (int px = 0; px < 2; px++)
                {
                    const int cx = conv_x_base + px;
                    const int cy = conv_y_base + py;

                    float sum = d_biases[filter_idx];
                    
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
                            if (pixel == 1u)
                            {
                                sum += s_filter[fy][fx];
                            }
                        }
                    }

                    int unpooled_idx = (batch_idx * 576 + cy * 24 + cx) * 8 + filter_idx;
                    d_unpooled_vals[unpooled_idx] = sum;

                    float activated = sum > 0.0f ? sum : 0.0f;
                    if (activated > max_val)
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

        // Optimized Fused Layer 2: Conv2 (8 channels -> 16 channels, 5x5 filter) + MaxPool2 + ReLU
        extern "C" __global__ void conv2_forward(
            const float* __restrict__ d_inputs,      // [BatchSize x 12 x 12 x 8]
            const float* __restrict__ d_filters,     // [16 x 5 x 5 x 8]
            const float* __restrict__ d_biases,      // [16]
            float* __restrict__ d_outputs,           // [BatchSize x 4 x 4 x 16]
            float* __restrict__ d_unpooled_vals)     // [BatchSize x 8 x 8 x 16]
        {
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x;

            // Load input activations and filter weights to shared memory
            __shared__ float s_input[1152]; // Flat 1D layout: [12][12][8]
            __shared__ float s_filters[3200]; // 16 filters x 5 x 5 x 8 channels = 3200 elements

            // Vectorized load of 1,152 input elements (288 float4s) and 3,200 filter weights (800 float4s) using float4
            for (int i = tid; i < 288; i += 256)
            {
                ((float4*)s_input)[i] = ((const float4*)d_inputs)[batch_idx * 288 + i];
            }

            for (int i = tid; i < 800; i += 256)
            {
                ((float4*)s_filters)[i] = ((const float4*)d_filters)[i];
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

                float max_val = -1e9f;

                #pragma unroll
                for (int py = 0; py < 2; py++)
                {
                    #pragma unroll
                    for (int px = 0; px < 2; px++)
                    {
                        const int cx = conv_x_base + px;
                        const int cy = conv_y_base + py;

                        float sum = d_biases[filter_idx];

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

                        float activated = sum > 0.0f ? sum : 0.0f;
                        if (activated > max_val)
                        {
                            max_val = activated;
                        }
                    }
                }

                const int out_idx_global = batch_idx * 256 + (out_y * 4 + out_x) * 16 + filter_idx;
                d_outputs[out_idx_global] = max_val;
            }
        }

        // FC2 Forward Pass (256 -> 10, Linear)
        extern "C" __global__ void fc2_forward(
            const float* __restrict__ d_inputs,      // [BatchSize x 256]
            const float* __restrict__ d_weights,     // [256 x 10]
            const float* __restrict__ d_biases,      // [10]
            float* __restrict__ d_outputs)           // [BatchSize x 10]
        {
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x; // 0..255

            __shared__ float s_input[256];
            // Vectorized load of 256 input elements (64 float4s)
            if (tid < 64)
            {
                ((float4*)s_input)[tid] = ((const float4*)d_inputs)[batch_idx * 64 + tid];
            }
            __syncthreads();

            if (tid < 10)
            {
                float sum = d_biases[tid];
                #pragma unroll 4
                for (int i = 0; i < 256; i++)
                {
                    sum += s_input[i] * d_weights[i * 10 + tid];
                }
                d_outputs[batch_idx * 10 + tid] = sum;
            }
        }

        // FC2 Layer Backward Pass
        extern "C" __global__ void fc2_backward(
            const float* __restrict__ d_fc2_outputs,  // [BatchSize x 10]
            const int* __restrict__ d_labels,        // [TotalImages]
            const float* __restrict__ d_fc2_inputs,   // [BatchSize x 256]
            const float* __restrict__ d_fc2_weights,   // [256 x 10]
            float* __restrict__ d_fc2_weights_grad,    // [256 x 10]
            float* __restrict__ d_fc2_biases_grad,     // [10]
            float* __restrict__ d_fc2_in_grad,        // [BatchSize x 256]
            const int* __restrict__ d_step)
        {
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x; // 0..255

            __shared__ float s_grad[10];

            int batchOffset = ((*d_step) % BATCHES_PER_EPOCH) * BATCH_SIZE;

            if (tid < 10)
            {
                float max_logit = -1e9f;
                for (int i = 0; i < 10; i++)
                {
                    float logit = d_fc2_outputs[batch_idx * 10 + i];
                    if (logit > max_logit) max_logit = logit;
                }

                float sum_exp = 0.0f;
                for (int i = 0; i < 10; i++)
                {
                    sum_exp += expf(d_fc2_outputs[batch_idx * 10 + i] - max_logit);
                }

                float prob = expf(d_fc2_outputs[batch_idx * 10 + tid] - max_logit) / sum_exp;
                int correct_label = d_labels[batchOffset + batch_idx];
                s_grad[tid] = prob - (tid == correct_label ? 1.0f : 0.0f);
            }
            __syncthreads();

            float x_val = d_fc2_inputs[batch_idx * 256 + tid];

            float sum_input_grad = 0.0f;
            #pragma unroll
            for (int c = 0; c < 10; c++)
            {
                sum_input_grad += s_grad[c] * d_fc2_weights[tid * 10 + c];
            }
            d_fc2_in_grad[batch_idx * 256 + tid] = sum_input_grad;

            if (x_val != 0.0f)
            {
                #pragma unroll
                for (int c = 0; c < 10; c++)
                {
                    float g_val = s_grad[c];
                    if (g_val != 0.0f)
                    {
                        atomicAdd(&d_fc2_weights_grad[tid * 10 + c], g_val * x_val);
                    }
                }
            }

            if (tid < 10)
            {
                float b_grad = s_grad[tid];
                if (b_grad != 0.0f)
                {
                    atomicAdd(&d_fc2_biases_grad[tid], b_grad);
                }
            }
        }

        // Conv2 Layer Backward Pass (Optimized chunked layout)
        extern "C" __global__ void conv2_backward(
            const float* __restrict__ d_conv2_out_grad, // [BatchSize x 256]
            const float* __restrict__ d_conv2_out_val,  // [BatchSize x 256]
            const float* __restrict__ d_conv2_unpooled_vals, // [BatchSize x 8 x 8 x 16]
            const float* __restrict__ d_conv1_out,      // [BatchSize x 12 x 12 x 8]
            const float* __restrict__ d_conv2_filters,  // [16 x 5 x 5 x 8]
            float* __restrict__ d_conv2_filters_grad,   // [16 x 5 x 5 x 8]
            float* __restrict__ d_conv2_biases_grad,    // [16]
            float* __restrict__ d_conv1_out_grad)       // [BatchSize x 12 x 12 x 8]
        {
            #define CONV2_CHUNKS 16
            #define CONV2_BATCH_PER_CHUNK (BATCH_SIZE / CONV2_CHUNKS)

            const int filter_idx = blockIdx.x / CONV2_CHUNKS;
            const int chunk_idx = blockIdx.x % CONV2_CHUNKS;
            const int tid = threadIdx.x;     // 0..127

            __shared__ float s_conv1_out[1152]; // Flat 1D layout: [12][12][8]
            __shared__ float s_filter_grad[200];
            __shared__ float s_bias_grad;
            __shared__ float s_grad[8][8];

            // Initialize shared accumulations
            for (int i = tid; i < 200; i += 128)
            {
                s_filter_grad[i] = 0.0f;
            }
            if (tid == 0)
            {
                s_bias_grad = 0.0f;
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
                // Vectorized load of 1,152 input elements (288 float4s) using float4
                for (int i = tid; i < 288; i += 128)
                {
                    ((float4*)s_conv1_out)[i] = ((const float4*)d_conv1_out)[b * 288 + i];
                }

                float out_grad = d_conv2_out_grad[b * 256 + pool_idx];
                float out_val = d_conv2_out_val[b * 256 + pool_idx];

                float my_val = 0.0f;
                if (tid < 64)
                {
                    my_val = d_conv2_unpooled_vals[(b * 64 + tid) * 16 + filter_idx];
                }

                float grad = 0.0f;
                if (tid < 64 && my_val == out_val && out_val > 0.0f)
                {
                    grad = out_grad;
                }
                if (tid < 64)
                {
                    s_grad[cy][cx] = grad;
                }
                __syncthreads();

                // 1. Accumulate weight gradients in shared memory
                for (int i = tid; i < 200; i += 128)
                {
                    int c = i % 8;
                    int fx = (i / 8) % 5;
                    int fy = i / 40;

                    float w_grad = 0.0f;
                    #pragma unroll
                    for (int y = 0; y < 8; y++)
                    {
                        #pragma unroll
                        for (int x = 0; x < 8; x++)
                        {
                            float g = s_grad[y][x];
                            if (g != 0.0f)
                            {
                                int in_x = x + fx;
                                int in_y = y + fy;
                                w_grad += g * s_conv1_out[(in_y * 12 + in_x) * 8 + c];
                            }
                        }
                    }
                    s_filter_grad[i] += w_grad;
                }

                // 2. Accumulate bias gradient
                if (tid == 0)
                {
                    float b_grad = 0.0f;
                    for (int y = 0; y < 8; y++)
                    {
                        for (int x = 0; x < 8; x++)
                        {
                            b_grad += s_grad[y][x];
                        }
                    }
                    s_bias_grad += b_grad;
                }

                // 3. Compute and add to d_conv1_out_grad in global memory
                for (int i = tid; i < 1152; i += 128)
                {
                    int c = i % 8;
                    int ix = (i / 8) % 12;
                    int iy = i / 96;

                    float sum_grad = 0.0f;
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
                    if (sum_grad != 0.0f)
                    {
                        atomicAdd(&d_conv1_out_grad[b * 1152 + i], sum_grad);
                    }
                }

                __syncthreads();
            }

            // Write final accumulated results to global memory
            for (int i = tid; i < 200; i += 128)
            {
                atomicAdd(&d_conv2_filters_grad[filter_idx * 200 + i], s_filter_grad[i]);
            }
            if (tid == 0)
            {
                atomicAdd(&d_conv2_biases_grad[filter_idx], s_bias_grad);
            }
        }

        // Conv1 Layer Backward Pass
        extern "C" __global__ void conv1_backward(
            const float* __restrict__ d_conv1_out_grad, // [BatchSize x 1152]
            const float* __restrict__ d_conv1_out_val,  // [BatchSize x 1152]
            const float* __restrict__ d_conv1_unpooled_vals, // [BatchSize x 24 x 24 x 8]
            const uint32_t* __restrict__ d_inputs,      // [TotalImages x 28] packed uints
            float* __restrict__ d_conv1_filters_grad,   // [8 x 5 x 5]
            float* __restrict__ d_conv1_biases_grad,    // [8]
            const int* __restrict__ d_step,
            int is_training)
        {
            #define CONV1_CHUNKS 16
            #define CONV1_BATCH_PER_CHUNK (BATCH_SIZE / CONV1_CHUNKS)

            const int filter_idx = blockIdx.x / CONV1_CHUNKS;
            const int chunk_idx = blockIdx.x % CONV1_CHUNKS;
            const int cx = threadIdx.x;       // 0..23
            const int cy = threadIdx.y;       // 0..23
            const int tid = cy * 24 + cx;     // 0..575

            __shared__ float s_filter_grad[25];
            __shared__ float s_bias_grad;
            __shared__ float s_grad[24][24];
            __shared__ uint32_t s_image[28];

            if (tid < 25)
            {
                s_filter_grad[tid] = 0.0f;
            }
            if (tid == 0)
            {
                s_bias_grad = 0.0f;
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

                float out_grad = d_conv1_out_grad[b * 1152 + pool_idx];
                float out_val = d_conv1_out_val[b * 1152 + pool_idx];
                float my_val = d_conv1_unpooled_vals[(b * 576 + cy * 24 + cx) * 8 + filter_idx];

                float grad = 0.0f;
                if (my_val == out_val && out_val > 0.0f)
                {
                    grad = out_grad;
                }
                s_grad[cy][cx] = grad;
                __syncthreads();

                // Simple pseudo-random shift generator
                int seed = b + *d_step;
                int dx = (is_training == 1) ? ((seed * 1103515245 + 12345) % 3 - 1) : 0;
                int dy = (is_training == 1) ? (((seed * 1103515245 + 12345) / 3) % 3 - 1) : 0;

                if (tid < 25)
                {
                    int fx = tid % 5;
                    int fy = tid / 5;

                    float w_grad = 0.0f;
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
                            if (pixel == 1u)
                            {
                                w_grad += s_grad[y][x];
                            }
                        }
                    }
                    s_filter_grad[tid] += w_grad;
                }

                if (tid == 0)
                {
                    float b_grad = 0.0f;
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

            // Write final accumulated results to global memory
            if (tid < 25)
            {
                atomicAdd(&d_conv1_filters_grad[filter_idx * 25 + tid], s_filter_grad[tid]);
            }
            if (tid == 0)
            {
                atomicAdd(&d_conv1_biases_grad[filter_idx], s_bias_grad);
            }
        }

        // Unified Adam Update Kernel
        extern "C" __global__ void adam_update(
            float* __restrict__ d_param,
            float* __restrict__ d_grad,
            float* __restrict__ d_m,
            float* __restrict__ d_v,
            int num_elements,
            int* __restrict__ d_step)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;

            int step_val = *d_step + 1;
            
            float max_lr = 0.06f; 
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
                float g = d_grad[i] / BATCH_SIZE;
                float m = beta1 * d_m[i] + (1.0f - beta1) * g;
                float v = beta2 * d_v[i] + (1.0f - beta2) * g * g;

                d_m[i] = m;
                d_v[i] = v;

                float m_hat = m / (1.0f - beta1_t);
                float v_hat = v / (1.0f - beta2_t);

                d_param[i] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
                d_grad[i] = 0.0f;
            }

            if (threadIdx.x == 0 && blockIdx.x == 0)
            {
                *d_step = step_val;
            }
        }
        """;
}
