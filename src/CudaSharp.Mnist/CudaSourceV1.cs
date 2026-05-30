namespace CudaSharp.Mnist;

public static partial class Program
{
    public static string CudaSourceV1 =>
        """
        typedef unsigned int uint32_t;
        
        #define BATCH_SIZE 128
        #define FILTER1_SIZE 5
        #define FILTER2_SIZE 3
        #define INPUT_SIZE 28
        #define POOL1_SIZE 12
        #define POOL2_SIZE 5
        #define FC1_INPUTS 400
        #define FC1_OUTPUTS 256
        #define FC2_OUTPUTS 10

        #define BATCHES_PER_EPOCH 200
        #define TOTAL_STEPS 400

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
            const float* __restrict__ d_filters,     // [16 x 5 x 5]
            const float* __restrict__ d_biases,      // [16]
            float* __restrict__ d_outputs,           // [BatchSize x 12 x 12 x 16]
            float* __restrict__ d_unpooled_vals,     // [BatchSize x 16 x 24 x 24]
            const int* __restrict__ d_step,
            int is_training)
        {
            const int batch_idx = blockIdx.x;
            const int filter_idx = blockIdx.y; // 0..15
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

                    int unpooled_idx = (batch_idx * 576 + cy * 24 + cx) * 16 + filter_idx;
                    d_unpooled_vals[unpooled_idx] = sum;

                    float activated = sum > 0.0f ? sum : 0.0f;
                    if (activated > max_val)
                    {
                        max_val = activated;
                    }
                }
            }

            const int out_idx = batch_idx * (12 * 12 * 16) 
                                + (out_y * 12 + out_x) * 16 
                                + filter_idx;
            d_outputs[out_idx] = max_val;
        }

        // Optimized Fused Layer 2: Conv2 (16 channels -> 16 channels, 3x3 filter) + MaxPool2 + ReLU
        extern "C" __global__ void conv2_forward(
            const float* __restrict__ d_inputs,      // [BatchSize x 12 x 12 x 16]
            const float* __restrict__ d_filters,     // [16 x 3 x 3 x 16]
            const float* __restrict__ d_biases,      // [16]
            float* __restrict__ d_outputs,           // [BatchSize x 5 x 5 x 16]
            float* __restrict__ d_unpooled_vals)     // [BatchSize x 10 x 10 x 16]
        {
            // Grid: BatchSize. 1 block per batch element!
            // Block size: 256 threads.
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x;

            // Load input activations and filter weights to shared memory
            __shared__ float s_input[2304]; // Flat 1D layout: [12][12][16]
            __shared__ float s_filters[2304]; // 16 filters x 3 x 3 x 16 channels = 2304 elements

            // Vectorized load of 2,304 input elements (576 float4s) and weights using float4
            for (int i = tid; i < 576; i += 256)
            {
                ((float4*)s_input)[i] = ((const float4*)d_inputs)[batch_idx * 576 + i];
            }

            for (int i = tid; i < 576; i += 256)
            {
                ((float4*)s_filters)[i] = ((const float4*)d_filters)[i];
            }
            __syncthreads();

            // We have 16 filters, each filter has 25 spatial output locations (5x5).
            // Total outputs to compute per block = 400.
            // With 256 threads, each thread computes 1 or 2 outputs.
            for (int out_idx = tid; out_idx < 400; out_idx += 256)
            {
                int filter_idx = out_idx / 25;
                int spatial_idx = out_idx % 25;
                int out_x = spatial_idx % 5;
                int out_y = spatial_idx / 5;

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
                        for (int c = 0; c < 16; c++)
                        {
                            #pragma unroll
                            for (int fy = 0; fy < 3; fy++)
                            {
                                #pragma unroll
                                for (int fx = 0; fx < 3; fx++)
                                {
                                    int in_x = cx + fx;
                                    int in_y = cy + fy;
                                    sum += s_input[(in_y * 12 + in_x) * 16 + c] * s_filters[filter_idx * 144 + (fy * 3 + fx) * 16 + c];
                                }
                            }
                        }

                        int unpooled_idx = (batch_idx * 100 + cy * 10 + cx) * 16 + filter_idx;
                        d_unpooled_vals[unpooled_idx] = sum;

                        float activated = sum > 0.0f ? sum : 0.0f;
                        if (activated > max_val)
                        {
                            max_val = activated;
                        }
                    }
                }

                const int out_idx_global = batch_idx * 400 + (out_y * 5 + out_x) * 16 + filter_idx;
                d_outputs[out_idx_global] = max_val;
            }
        }

        // FC1 Forward Pass (400 -> 256, ReLU)
        extern "C" __global__ void fc1_forward(
            const float* __restrict__ d_inputs,      // [BatchSize x 400]
            const float* __restrict__ d_weights,     // [400 x 256]
            const float* __restrict__ d_biases,      // [256]
            float* __restrict__ d_outputs)           // [BatchSize x 256]
        {
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x; // 0..255

            __shared__ float s_input[400];
            // Vectorized load of 400 input elements (100 float4s)
            if (tid < 100)
            {
                float4 val = ((const float4*)d_inputs)[batch_idx * 100 + tid];
                ((float4*)s_input)[tid] = val;
            }
            __syncthreads();

            float sum = d_biases[tid];
            #pragma unroll 4
            for (int i = 0; i < 400; i++)
            {
                sum += s_input[i] * d_weights[i * 256 + tid];
            }
            d_outputs[batch_idx * 256 + tid] = sum > 0.0f ? sum : 0.0f;
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
            const float* __restrict__ d_fc1_outputs,   // [BatchSize x 256]
            const float* __restrict__ d_fc2_weights,   // [256 x 10]
            float* __restrict__ d_fc2_weights_grad,    // [256 x 10]
            float* __restrict__ d_fc2_biases_grad,     // [10]
            float* __restrict__ d_fc1_out_grad,       // [BatchSize x 256]
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

            float x_val = d_fc1_outputs[batch_idx * 256 + tid];

            float sum_input_grad = 0.0f;
            #pragma unroll
            for (int c = 0; c < 10; c++)
            {
                sum_input_grad += s_grad[c] * d_fc2_weights[tid * 10 + c];
            }
            d_fc1_out_grad[batch_idx * 256 + tid] = sum_input_grad;

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

        // FC1 Layer Backward Pass (simplified: only inputs and biases)
        extern "C" __global__ void fc1_backward(
            const float* __restrict__ d_fc1_out_grad, // [BatchSize x 256]
            const float* __restrict__ d_fc1_outputs,  // [BatchSize x 256]
            const float* __restrict__ d_conv2_outputs, // [BatchSize x 400]
            const float* __restrict__ d_fc1_weights,  // [400 x 256]
            float* __restrict__ d_fc1_biases_grad,    // [256]
            float* __restrict__ d_conv2_out_grad)     // [BatchSize x 400]
        {
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x; // 0..255

            __shared__ float s_out_grad[256];

            float out_val = d_fc1_outputs[batch_idx * 256 + tid];
            s_out_grad[tid] = out_val > 0.0f ? d_fc1_out_grad[batch_idx * 256 + tid] : 0.0f;
            __syncthreads();

            const int warp_id = tid / 32;
            const int lane = tid % 32;

            for (int i = warp_id; i < 400; i += 8)
            {
                float sum = 0.0f;
                #pragma unroll
                for (int k = 0; k < 8; k++)
                {
                    int c = k * 32 + lane;
                    sum += s_out_grad[c] * d_fc1_weights[i * 256 + c];
                }

                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2)
                {
                    sum += __shfl_down_sync(0xffffffff, sum, offset);
                }

                if (lane == 0)
                {
                    d_conv2_out_grad[batch_idx * 400 + i] = sum;
                }
            }

            float my_grad = s_out_grad[tid];
            if (my_grad != 0.0f)
            {
                atomicAdd(&d_fc1_biases_grad[tid], my_grad);
            }
        }

        // FC1 Weights Gradient Kernel (Zero atomics, block-level parallel reductions!)
        extern "C" __global__ void fc1_backward_weights(
            const float* __restrict__ d_fc1_out_grad, // [BatchSize x 256]
            const float* __restrict__ d_fc1_outputs,  // [BatchSize x 256]
            const float* __restrict__ d_conv2_outputs, // [BatchSize x 400]
            float* __restrict__ d_fc1_weights_grad)   // [400 x 256]
        {
            const int input_idx = blockIdx.x; // 0..399
            const int tid = threadIdx.x;      // 0..31

            __shared__ float s_accum[32][256];

            #pragma unroll 4
            for (int c = 0; c < 256; c++)
            {
                s_accum[tid][c] = 0.0f;
            }
            __syncthreads();

            #pragma unroll
            for (int i = 0; i < (BATCH_SIZE / 32); i++)
            {
                int b = i * 32 + tid;
                float x_val = d_conv2_outputs[b * 400 + input_idx];
                #pragma unroll 4
                for (int c = 0; c < 256; c++)
                {
                    float out_val = d_fc1_outputs[b * 256 + c];
                    float g_val = out_val > 0.0f ? d_fc1_out_grad[b * 256 + c] : 0.0f;
                    s_accum[tid][c] += x_val * g_val;
                }
            }
            __syncthreads();

            for (int stride = 16; stride > 0; stride >>= 1)
            {
                if (tid < stride)
                {
                    #pragma unroll 4
                    for (int c = 0; c < 256; c++)
                    {
                        s_accum[tid][c] += s_accum[tid + stride][c];
                    }
                }
                __syncthreads();
            }

            if (tid == 0)
            {
                #pragma unroll 4
                for (int c = 0; c < 256; c++)
                {
                    d_fc1_weights_grad[input_idx * 256 + c] = s_accum[0][c];
                }
            }
        }

        // Conv2 Layer Backward Pass (Optimized chunked layout)
        extern "C" __global__ void conv2_backward(
            const float* __restrict__ d_conv2_out_grad, // [BatchSize x 400]
            const float* __restrict__ d_conv2_out_val,  // [BatchSize x 400]
            const float* __restrict__ d_conv2_unpooled_vals, // [BatchSize x 10 x 10 x 16]
            const float* __restrict__ d_conv1_out,      // [BatchSize x 12 x 12 x 16]
            const float* __restrict__ d_conv2_filters,  // [16 x 3 x 3 x 16]
            float* __restrict__ d_conv2_filters_grad,   // [16 x 3 x 3 x 16]
            float* __restrict__ d_conv2_biases_grad,    // [16]
            float* __restrict__ d_conv1_out_grad)       // [BatchSize x 12 x 12 x 16]
        {
            #define CONV2_CHUNKS 16
            #define CONV2_BATCH_PER_CHUNK (BATCH_SIZE / CONV2_CHUNKS)

            const int filter_idx = blockIdx.x / CONV2_CHUNKS;
            const int chunk_idx = blockIdx.x % CONV2_CHUNKS;
            const int tid = threadIdx.x;     // 0..127

            __shared__ float s_conv1_out[2304]; // Flat 1D layout: [12][12][16]
            __shared__ float s_filter_grad[144];
            __shared__ float s_grad[10][10];

            // Initialize shared accumulations
            for (int i = tid; i < 144; i += 128)
            {
                s_filter_grad[i] = 0.0f;
            }
            __syncthreads();

            const int cx = (tid < 100) ? (tid % 10) : 0;
            const int cy = (tid < 100) ? (tid / 10) : 0;
            const int px = cx / 2;
            const int py = cy / 2;
            const int pool_idx = (py * 5 + px) * 16 + filter_idx;

            const int start_b = chunk_idx * CONV2_BATCH_PER_CHUNK;
            const int end_b = start_b + CONV2_BATCH_PER_CHUNK;

            float local_bias_grad = 0.0f;

            for (int b = start_b; b < end_b; b++)
            {
                // Vectorized load of 2,304 elements (576 float4s) from d_conv1_out to shared memory
                for (int i = tid; i < 576; i += 128)
                {
                    ((float4*)s_conv1_out)[i] = ((const float4*)d_conv1_out)[b * 576 + i];
                }

                float out_grad = d_conv2_out_grad[b * 400 + pool_idx];
                float out_val = d_conv2_out_val[b * 400 + pool_idx];

                float my_val = 0.0f;
                if (tid < 100)
                {
                    my_val = d_conv2_unpooled_vals[(b * 100 + tid) * 16 + filter_idx];
                }

                float grad = 0.0f;
                if (tid < 100 && my_val == out_val && out_val > 0.0f)
                {
                    grad = out_grad;
                }
                if (tid < 100)
                {
                    s_grad[cy][cx] = grad;
                    local_bias_grad += grad;
                }
                __syncthreads();

                // 1. Accumulate weight gradients in shared memory
                for (int i = tid; i < 144; i += 128)
                {
                    int c = i % 16;
                    int fx = (i / 16) % 3;
                    int fy = i / 48;

                    float w_grad = 0.0f;
                    #pragma unroll
                    for (int y = 0; y < 10; y++)
                    {
                        #pragma unroll
                        for (int x = 0; x < 10; x++)
                        {
                            float g = s_grad[y][x];
                            if (g != 0.0f)
                            {
                                int in_x = x + fx;
                                int in_y = y + fy;
                                w_grad += g * s_conv1_out[(in_y * 12 + in_x) * 16 + c];
                            }
                        }
                    }
                    s_filter_grad[i] += w_grad;
                }

                // 3. Compute and add to d_conv1_out_grad in global memory
                for (int i = tid; i < 2304; i += 128)
                {
                    int c = i % 16;
                    int ix = (i / 16) % 12;
                    int iy = i / 192;

                    float sum_grad = 0.0f;
                    #pragma unroll
                    for (int fy = 0; fy < 3; fy++)
                    {
                        #pragma unroll
                        for (int fx = 0; fx < 3; fx++)
                        {
                            int x = ix - fx;
                            int y = iy - fy;
                            if (x >= 0 && x < 10 && y >= 0 && y < 10)
                            {
                                int f_idx = filter_idx * 144 + (fy * 3 + fx) * 16 + c;
                                sum_grad += s_grad[y][x] * d_conv2_filters[f_idx];
                            }
                        }
                    }
                    if (sum_grad != 0.0f)
                    {
                        atomicAdd(&d_conv1_out_grad[b * 2304 + i], sum_grad);
                    }
                }

                __syncthreads();
            }

            // Write final accumulated results to global memory
            for (int i = tid; i < 144; i += 128)
            {
                atomicAdd(&d_conv2_filters_grad[filter_idx * 144 + i], s_filter_grad[i]);
            }

            // Parallel reduction for bias gradient
            __shared__ float s_bias_reduce[128];
            s_bias_reduce[tid] = (tid < 100) ? local_bias_grad : 0.0f;
            __syncthreads();

            for (int stride = 64; stride > 0; stride /= 2)
            {
                if (tid < stride)
                {
                    s_bias_reduce[tid] += s_bias_reduce[tid + stride];
                }
                __syncthreads();
            }

            if (tid == 0)
            {
                atomicAdd(&d_conv2_biases_grad[filter_idx], s_bias_reduce[0]);
            }
        }

        // Conv1 Layer Backward Pass (Optimized chunked layout with random shift)
        extern "C" __global__ void conv1_backward(
            const float* __restrict__ d_conv1_out_grad, // [BatchSize x 2304]
            const float* __restrict__ d_conv1_out_val,  // [BatchSize x 2304]
            const float* __restrict__ d_conv1_unpooled_vals, // [BatchSize x 24 x 24 x 16]
            const uint32_t* __restrict__ d_inputs,      // [TotalImages x 28] packed uints
            float* __restrict__ d_conv1_filters_grad,   // [16 x 5 x 5]
            float* __restrict__ d_conv1_biases_grad,    // [16]
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
            __shared__ float s_grad[24][24];
            __shared__ uint32_t s_image[28];

            if (tid < 25)
            {
                s_filter_grad[tid] = 0.0f;
            }
            __syncthreads();

            const int batchOffset = ((*d_step) % BATCHES_PER_EPOCH) * BATCH_SIZE;

            const int px = cx / 2;
            const int py = cy / 2;
            const int pool_idx = (py * 12 + px) * 16 + filter_idx;

            const int start_b = chunk_idx * CONV1_BATCH_PER_CHUNK;
            const int end_b = start_b + CONV1_BATCH_PER_CHUNK;

            float local_bias_grad = 0.0f;

            for (int b = start_b; b < end_b; b++)
            {
                if (tid < 28)
                {
                    s_image[tid] = d_inputs[(batchOffset + b) * 28 + tid];
                }

                float out_grad = d_conv1_out_grad[b * 2304 + pool_idx];
                float out_val = d_conv1_out_val[b * 2304 + pool_idx];
                float my_val = d_conv1_unpooled_vals[(b * 576 + cy * 24 + cx) * 16 + filter_idx];

                float grad = 0.0f;
                if (my_val == out_val && out_val > 0.0f)
                {
                    grad = out_grad;
                }
                s_grad[cy][cx] = grad;
                local_bias_grad += grad;
                __syncthreads();

                // Simple pseudo-random shift generator (must match forward shift!)
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
                __syncthreads();
            }

            // Write final accumulated results to global memory
            if (tid < 25)
            {
                atomicAdd(&d_conv1_filters_grad[filter_idx * 25 + tid], s_filter_grad[tid]);
            }

            // Parallel reduction for bias gradient
            __shared__ float s_bias_reduce[576];
            s_bias_reduce[tid] = local_bias_grad;
            __syncthreads();

            for (int stride = 256; stride > 0; stride /= 2)
            {
                if (tid < stride && tid + stride < 576)
                {
                    s_bias_reduce[tid] += s_bias_reduce[tid + stride];
                }
                __syncthreads();
            }

            if (tid == 0)
            {
                atomicAdd(&d_conv1_biases_grad[filter_idx], s_bias_reduce[0]);
            }
        }

        // Unified Adam Update Kernel (Now increments step counter on-device)
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

            int step_val = *d_step + 1; // 1-indexed for beta power
            
            // Get learning rate for this step using OneCycleLR formula
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
                float g = d_grad[i] / BATCH_SIZE;
                float m = beta1 * d_m[i] + (1.0f - beta1) * g;
                float v = beta2 * d_v[i] + (1.0f - beta2) * g * g;

                d_m[i] = m;
                d_v[i] = v;

                float m_hat = m / (1.0f - beta1_t);
                float v_hat = v / (1.0f - beta2_t);

                d_param[i] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
                d_grad[i] = 0.0f; // Reset gradient on-device!
            }

            if (threadIdx.x == 0 && blockIdx.x == 0)
            {
                *d_step = step_val;
            }
        }
        """;
}
