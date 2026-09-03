namespace CudaSharp.Mnist;

public static partial class Program
{
    public static string CudaSourceV7 =>
        """
        #include <cuda_fp16.h>

        typedef unsigned int uint32_t;

        #ifndef BATCH_SIZE
        #define BATCH_SIZE 256
        #endif
        #define FILTER1_SIZE 3
        #define FILTER2_SIZE 3
        #define INPUT_SIZE 28
        #define POOL1_SIZE 13
        #define POOL2_SIZE 5
        #define FC2_INPUTS 400
        #define FC2_OUTPUTS 10

        #ifndef BATCHES_PER_EPOCH
        #define BATCHES_PER_EPOCH 200
        #endif
        #ifndef TOTAL_STEPS
        #define TOTAL_STEPS 100
        #endif

        // Helper to clear a gradient buffer asynchronously
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

        // Dummy empty kernels to satisfy C# Module loading structure
        extern "C" __global__ void fused_forward(
            const uint32_t* d_inputs, const __half* d_conv1_filters, const __half* d_conv1_biases,
            const __half* d_conv2_filters, const __half* d_conv2_biases, const __half* d_fc2_weights,
            const __half* d_fc2_biases, __half* d_conv1_out, __half* d_conv1_unpooled,
            __half* d_conv2_out, __half* d_conv2_unpooled, __half* d_fc2_out,
            const int* d_step, int is_training) {}

        extern "C" __global__ void fused_backward(
            const __half* d_fc2_out, const unsigned char* d_labels, const __half* d_conv2_out,
            const __half* d_fc2_weights, __half* d_fc2_weights_grad, __half* d_fc2_biases_grad,
            const __half* d_conv2_unpooled, const __half* d_conv1_out, const __half* d_conv2_filters,
            __half* d_conv2_filters_grad, __half* d_conv2_biases_grad, const __half* d_conv1_unpooled,
            const uint32_t* d_inputs, __half* d_conv1_filters_grad, __half* d_conv1_biases_grad,
            const int* d_step) {}

        // Conv1 Forward Pass (1-bit input -> 8 filters, 13x13 output after MaxPool + ReLU)
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

            if (out_x >= 13 || out_y >= 13) return;

            int batchOffset = ((*d_step) % BATCHES_PER_EPOCH) * BATCH_SIZE;

            __shared__ __half s_filter[3][3];
            int tid_flat = threadIdx.y * 13 + threadIdx.x;
            if (tid_flat < 9)
            {
                s_filter[tid_flat / 3][tid_flat % 3] = d_filters[filter_idx * 9 + tid_flat];
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
                    for (int fy = 0; fy < 3; fy++)
                    {
                        int shift_y = cy + fy + dy;
                        uint32_t row_bits = 0;
                        if (shift_y >= 0 && shift_y < 28)
                        {
                            row_bits = s_image[shift_y];
                        }
                        #pragma unroll
                        for (int fx = 0; fx < 3; fx++)
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

                    int unpooled_idx = (batch_idx * 676 + cy * 26 + cx) * 8 + filter_idx;
                    d_unpooled_vals[unpooled_idx] = sum;

                    __half activated = __hgt(sum, zero) ? sum : zero;
                    if (__hgt(activated, max_val))
                    {
                        max_val = activated;
                    }
                }
            }

            const int out_idx = batch_idx * 1352 + (out_y * 13 + out_x) * 8 + filter_idx;
            d_outputs[out_idx] = max_val;
        }

        // Conv2 Forward Pass (8 channels -> 16 filters, 5x5 output after MaxPool + ReLU)
        extern "C" __global__ void conv2_forward(
            const __half* __restrict__ d_inputs,
            const __half* __restrict__ d_filters,
            const __half* __restrict__ d_biases,
            __half* __restrict__ d_outputs,
            __half* __restrict__ d_unpooled_vals)
        {
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x;

            __shared__ __half s_input[1352]; // 13 * 13 * 8 = 1352
            __shared__ __half s_filters[1152]; // 16 * 8 * 9 = 1152

            for (int i = tid; i < 1352; i += 256)
            {
                s_input[i] = d_inputs[batch_idx * 1352 + i];
            }

            for (int i = tid; i < 1152; i += 256)
            {
                s_filters[i] = d_filters[i];
            }
            __syncthreads();

            #pragma unroll
            for (int out_idx = tid; out_idx < 400; out_idx += 256)
            {
                int filter_idx = out_idx / 25;
                int spatial_idx = out_idx % 25;
                int out_x = spatial_idx % 5;
                int out_y = spatial_idx / 5;

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
                            for (int fy = 0; fy < 3; fy++)
                            {
                                #pragma unroll
                                for (int fx = 0; fx < 3; fx++)
                                {
                                    int in_x = cx + fx;
                                    int in_y = cy + fy;
                                    sum += s_input[(in_y * 13 + in_x) * 8 + c] * s_filters[filter_idx * 72 + (fy * 3 + fx) * 8 + c];
                                }
                            }
                        }

                        int unpooled_idx = (batch_idx * 100 + cy * 10 + cx) * 16 + filter_idx;
                        d_unpooled_vals[unpooled_idx] = sum;

                        __half activated = __hgt(sum, zero) ? sum : zero;
                        if (__hgt(activated, max_val))
                        {
                            max_val = activated;
                        }
                    }
                }

                const int out_idx_global = batch_idx * 400 + (out_y * 5 + out_x) * 16 + filter_idx;
                d_outputs[out_idx_global] = max_val;
            }
        }

        // FC2 Linear Forward Pass (400 features -> 10 classes)
        extern "C" __global__ void fc2_forward(
            const __half* __restrict__ d_inputs,
            const __half* __restrict__ d_weights,
            const __half* __restrict__ d_biases,
            __half* __restrict__ d_outputs)
        {
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x;

            __shared__ __half s_input[400];
            s_input[tid] = d_inputs[batch_idx * 400 + tid];
            if (tid + 256 < 400)
            {
                s_input[tid + 256] = d_inputs[batch_idx * 400 + tid + 256];
            }
            __syncthreads();

            if (tid < 10)
            {
                __half sum = d_biases[tid];
                #pragma unroll 4
                for (int i = 0; i < 400; i++)
                {
                    sum += s_input[i] * d_weights[i * 10 + tid];
                }
                d_outputs[batch_idx * 10 + tid] = sum;
            }
        }

        // FC2 Backward Pass (Activation gradients and bias gradients only, ZERO weights computation!)
        extern "C" __global__ void fc2_backward(
            const __half* __restrict__ d_fc2_outputs,
            const unsigned char* __restrict__ d_labels,
            const __half* __restrict__ d_fc2_inputs,
            const __half* __restrict__ d_fc2_weights,
            __half* __restrict__ d_fc2_weights_grad, // unused
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

            __half sum_input_grad = __float2half(0.0f);
            __half sum_input_grad2 = __float2half(0.0f);
            #pragma unroll
            for (int c = 0; c < 10; c++)
            {
                sum_input_grad += s_grad[c] * d_fc2_weights[tid * 10 + c];
                if (tid + 256 < 400)
                {
                    sum_input_grad2 += s_grad[c] * d_fc2_weights[(tid + 256) * 10 + c];
                }
            }
            d_fc2_in_grad[batch_idx * 400 + tid] = sum_input_grad;
            if (tid + 256 < 400)
            {
                d_fc2_in_grad[batch_idx * 400 + tid + 256] = sum_input_grad2;
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

        // FC2 Backward Weights Pass (Zero atomics, block-level parallel reductions!)
        extern "C" __global__ void fc2_backward_weights(
            const __half* __restrict__ d_fc2_outputs,  // [BatchSize x 10]
            const unsigned char* __restrict__ d_labels,         // [TotalImages]
            const __half* __restrict__ d_fc2_inputs,   // [BatchSize x 400]
            __half* __restrict__ d_fc2_weights_grad,   // [400 x 10]
            const int* __restrict__ d_step)
        {
            const int input_idx = blockIdx.x; // 0..399
            const int tid = threadIdx.x;      // 0..127

            int batchOffset = ((*d_step) % BATCHES_PER_EPOCH) * BATCH_SIZE;

            __shared__ float s_weight_grads[128][10];

            #pragma unroll
            for (int c = 0; c < 10; c++)
            {
                s_weight_grads[tid][c] = 0.0f;
            }
            __syncthreads();

            #pragma unroll
            for (int i = 0; i < 2; i++)
            {
                int b = i * 128 + tid;

                float max_logit = -1e9f;
                for (int c = 0; c < 10; c++)
                {
                    float logit = __half2float(d_fc2_outputs[b * 10 + c]);
                    if (logit > max_logit) max_logit = logit;
                }

                float sum_exp = 0.0f;
                for (int c = 0; c < 10; c++)
                {
                    sum_exp += expf(__half2float(d_fc2_outputs[b * 10 + c]) - max_logit);
                }

                int correct_label = d_labels[batchOffset + b];
                float x_val = __half2float(d_fc2_inputs[b * 400 + input_idx]);

                #pragma unroll
                for (int c = 0; c < 10; c++)
                {
                    float prob = expf(__half2float(d_fc2_outputs[b * 10 + c]) - max_logit) / sum_exp;
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

        // Conv2 Backward Pass
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

            __shared__ __half s_conv1_out[1352]; // 13 * 13 * 8 = 1352
            __shared__ __half s_filter_grad[72]; // 8 * 9 = 72
            __shared__ __half s_bias_grad;
            __shared__ __half s_grad[10][10];

            __half zero = __float2half(0.0f);

            for (int i = tid; i < 72; i += 128)
            {
                s_filter_grad[i] = zero;
            }
            if (tid == 0)
            {
                s_bias_grad = zero;
            }
            __syncthreads();

            const int cx = (tid < 100) ? (tid % 10) : 0;
            const int cy = (tid < 100) ? (tid / 10) : 0;
            const int px = cx / 2;
            const int py = cy / 2;

            const int start_b = chunk_idx * CONV2_BATCH_PER_CHUNK;
            const int end_b = start_b + CONV2_BATCH_PER_CHUNK;

            int c_arr[1], fx_arr[1], fy_arr[1];
            int i_count = 0;
            for (int i = tid; i < 72; i += 128)
            {
                c_arr[i_count] = i % 8;
                fx_arr[i_count] = (i / 8) % 3;
                fy_arr[i_count] = i / 24;
                i_count++;
            }

            int c_grad_arr[11], ix_grad_arr[11], iy_grad_arr[11];
            int i_grad_count = 0;
            for (int i = tid; i < 1352; i += 128)
            {
                c_grad_arr[i_grad_count] = i % 8;
                ix_grad_arr[i_grad_count] = (i / 8) % 13;
                iy_grad_arr[i_grad_count] = i / 104;
                i_grad_count++;
            }

            for (int b = start_b; b < end_b; b++)
            {
                for (int i = tid; i < 1352; i += 128)
                {
                    s_conv1_out[i] = d_conv1_out[b * 1352 + i];
                }

                __half out_grad = zero;
                __half out_val = zero;
                if (tid < 100 && px < 5 && py < 5)
                {
                    int pool_idx = (py * 5 + px) * 16 + filter_idx;
                    out_grad = d_conv2_out_grad[b * 400 + pool_idx];
                    out_val = d_conv2_out_val[b * 400 + pool_idx];
                }

                __half my_val = zero;
                if (tid < 100)
                {
                    my_val = d_conv2_unpooled_vals[(b * 100 + tid) * 16 + filter_idx];
                }

                __half grad = zero;
                if (tid < 100 && __heq(my_val, out_val) && __hgt(out_val, zero))
                {
                    grad = out_grad;
                }
                if (tid < 100)
                {
                    s_grad[cy][cx] = grad;
                }
                __syncthreads();

                int i_idx = 0;
                for (int i = tid; i < 72; i += 128)
                {
                    int c = c_arr[i_idx];
                    int fx = fx_arr[i_idx];
                    int fy = fy_arr[i_idx];
                    i_idx++;

                    __half w_grad = zero;
                    #pragma unroll
                    for (int y = 0; y < 10; y++)
                    {
                        #pragma unroll
                        for (int x = 0; x < 10; x++)
                        {
                            __half g = s_grad[y][x];
                            if (__half2float(g) != 0.0f)
                            {
                                int in_x = x + fx;
                                int in_y = y + fy;
                                w_grad += g * s_conv1_out[(in_y * 13 + in_x) * 8 + c];
                            }
                        }
                    }
                    s_filter_grad[i] += w_grad;
                }

                if (tid == 0)
                {
                    __half b_grad = zero;
                    for (int y = 0; y < 10; y++)
                    {
                        for (int x = 0; x < 10; x++)
                        {
                            b_grad += s_grad[y][x];
                        }
                    }
                    s_bias_grad += b_grad;
                }

                int i_grad_idx = 0;
                for (int i = tid; i < 1352; i += 128)
                {
                    int c = c_grad_arr[i_grad_idx];
                    int ix = ix_grad_arr[i_grad_idx];
                    int iy = iy_grad_arr[i_grad_idx];
                    i_grad_idx++;

                    __half sum_grad = zero;
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
                                int f_idx = filter_idx * 72 + (fy * 3 + fx) * 8 + c;
                                sum_grad += s_grad[y][x] * d_conv2_filters[f_idx];
                            }
                        }
                    }
                    if (__half2float(sum_grad) != 0.0f)
                    {
                        atomicAdd(&d_conv1_out_grad[b * 1352 + i], sum_grad);
                    }
                }

                __syncthreads();
            }

            for (int i = tid; i < 72; i += 128)
            {
                atomicAdd(&d_conv2_filters_grad[filter_idx * 72 + i], s_filter_grad[i]);
            }
            if (tid == 0)
            {
                atomicAdd(&d_conv2_biases_grad[filter_idx], s_bias_grad);
            }
        }

        // Conv1 Backward Pass
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
            const int tid = threadIdx.y * 16 + threadIdx.x; // Block is 16x16 = 256 threads

            __shared__ __half s_filter_grad[9];
            __shared__ __half s_bias_grad;
            __shared__ __half s_grad[26][26];
            __shared__ uint32_t s_image[28];

            __half zero = __float2half(0.0f);

            if (tid < 9)
            {
                s_filter_grad[tid] = zero;
            }
            if (tid == 0)
            {
                s_bias_grad = zero;
            }
            __syncthreads();

            const int batchOffset = ((*d_step) % BATCHES_PER_EPOCH) * BATCH_SIZE;

            const int start_b = chunk_idx * CONV1_BATCH_PER_CHUNK;
            const int end_b = start_b + CONV1_BATCH_PER_CHUNK;

            int fx = tid % 3;
            int fy = tid / 3;

            for (int b = start_b; b < end_b; b++)
            {
                if (tid < 28)
                {
                    s_image[tid] = d_inputs[(batchOffset + b) * 28 + tid];
                }

                // Parallel populate s_grad across 256 threads
                for (int i = tid; i < 676; i += 256)
                {
                    int gy = i / 26;
                    int gx = i % 26;
                    int px = gx / 2;
                    int py = gy / 2;
                    int pool_idx = (py * 13 + px) * 8 + filter_idx;

                    __half out_grad = d_conv1_out_grad[b * 1352 + pool_idx];
                    __half out_val = d_conv1_out_val[b * 1352 + pool_idx];
                    __half my_val = d_conv1_unpooled_vals[(b * 676 + gy * 26 + gx) * 8 + filter_idx];

                    __half grad = zero;
                    if (__heq(my_val, out_val) && __hgt(out_val, zero))
                    {
                        grad = out_grad;
                    }
                    s_grad[gy][gx] = grad;
                }
                __syncthreads();

                int seed = b + *d_step;
                int dx = (is_training == 1) ? ((seed * 1103515245 + 12345) % 3 - 1) : 0;
                int dy = (is_training == 1) ? (((seed * 1103515245 + 12345) / 3) % 3 - 1) : 0;

                if (tid < 9)
                {
                    __half w_grad = zero;
                    #pragma unroll
                    for (int y = 0; y < 26; y++)
                    {
                        int shift_y = y + fy + dy;
                        uint32_t row_bits = 0;
                        if (shift_y >= 0 && shift_y < 28)
                        {
                            row_bits = s_image[shift_y];
                        }
                        #pragma unroll
                        for (int x = 0; x < 26; x++)
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
                    for (int y = 0; y < 26; y++)
                    {
                        for (int x = 0; x < 26; x++)
                        {
                            b_grad += s_grad[y][x];
                        }
                    }
                    s_bias_grad += b_grad;
                }
                __syncthreads();
            }

            if (tid < 9)
            {
                atomicAdd(&d_conv1_filters_grad[filter_idx * 9 + tid], s_filter_grad[tid]);
            }
            if (tid == 0)
            {
                atomicAdd(&d_conv1_biases_grad[filter_idx], s_bias_grad);
            }
        }

        // Adam parameter updates (custom FP16)
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
            float max_lr = MAX_LR; 
            float beta1 = 0.7f;
            float beta2 = 0.9f;
            float epsilon = 1e-8f;
            
            int total_steps = TOTAL_STEPS;
            float lr = 0.0f;
            int decay_start = (int)(total_steps * 0.75f);
            if (step_val < decay_start)
            {
                lr = max_lr;
            }
            else
            {
                float phase_pct = (float)(step_val - decay_start) / (total_steps - decay_start);
                float cos_val = cosf(3.14159265f * phase_pct);
                lr = max_lr * 0.5f * (1.0f + cos_val);
            }

            float beta1_t = powf(beta1, step_val);
            float beta2_t = powf(beta2, step_val);

            for (int i = tid; i < num_elements; i += stride)
            {
                float g = __half2float(d_grad[i]) / BATCH_SIZE;
                if (!isfinite(g))
                {
                    d_grad[i] = __float2half(0.0f);
                    continue;
                }

                float m = beta1 * __half2float(d_m[i]) + (1.0f - beta1) * g;
                float v = beta2 * __half2float(d_v[i]) + (1.0f - beta2) * g * g;

                if (!isfinite(m) || !isfinite(v))
                {
                    d_m[i] = __float2half(0.0f);
                    d_v[i] = __float2half(0.0f);
                    d_grad[i] = __float2half(0.0f);
                    continue;
                }

                d_m[i] = __float2half(m);
                d_v[i] = __float2half(v);

                float m_hat = m / (1.0f - beta1_t);
                float v_hat = v / (1.0f - beta2_t);
                if (!isfinite(m_hat) || !isfinite(v_hat) || v_hat < 0.0f)
                {
                    d_grad[i] = __float2half(0.0f);
                    continue;
                }

                float param_val = __half2float(d_param[i]);
                param_val -= lr * m_hat / (sqrtf(v_hat) + epsilon);
                if (!isfinite(param_val))
                {
                    d_m[i] = __float2half(0.0f);
                    d_v[i] = __float2half(0.0f);
                    d_grad[i] = __float2half(0.0f);
                    continue;
                }

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
