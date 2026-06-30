namespace CudaSharp.Mnist;

public static partial class Program
{
    public static string CudaSourceV8 =>
        """
        #include <cuda_fp16.h>

        typedef unsigned int uint32_t;

        #ifndef BATCH_SIZE
        #define BATCH_SIZE 256
        #endif
        #define FILTER_SIZE 3
        #define INPUT_SIZE 28
        #define CONV1_OUT_SIZE 14
        #define RESBLOCK_OUT_SIZE 14
        #define POOL2_OUT_SIZE 7
        #define FC2_INPUTS 784
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

        // Conv1 Forward Pass (1-bit input -> 16 filters, 14x14 output with Stride 2)
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

            if (out_x >= 14 || out_y >= 14) return;

            int batchOffset = ((*d_step) % BATCHES_PER_EPOCH) * BATCH_SIZE;

            __shared__ __half s_filter[3][3];
            int tid_flat = threadIdx.y * 14 + threadIdx.x;
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

            __half sum = d_biases[filter_idx];
            __half zero = __float2half(0.0f);

            #pragma unroll
            for (int fy = 0; fy < 3; fy++)
            {
                int shift_y = conv_y_base + fy + dy;
                uint32_t row_bits = 0;
                if (shift_y >= 0 && shift_y < 28)
                {
                    row_bits = s_image[shift_y];
                }
                #pragma unroll
                for (int fx = 0; fx < 3; fx++)
                {
                    int img_x = conv_x_base + fx + dx;
                    uint32_t pixel = 0;
                    if (img_x >= 0 && img_x < 28)
                    {
                        pixel = (row_bits >> img_x) & 1u;
                    }
                    sum += __half((float)pixel) * s_filter[fy][fx];
                }
            }

            int out_idx_unpooled = (batch_idx * 196 + out_y * 14 + out_x) * 16 + filter_idx;
            d_unpooled_vals[out_idx_unpooled] = sum; // unpooled representation

            __half activated = __hgt(sum, zero) ? sum : zero;
            d_outputs[out_idx_unpooled] = activated;
        }

        // Fused ResBlock Forward Pass
        // Cooperative Thread Block loads 14x14x16 input, computes Conv2a (16 filters) -> ReLU,
        // then computes Conv2b (16 filters) -> Residual Add + ReLU -> output.
        extern "C" __global__ void conv2_forward(
            const __half* __restrict__ d_inputs,         // [Batch x 14x14x16]
            const __half* __restrict__ d_filters,        // [16 x 16 x 3 x 3] = 2304 elements (Conv2a)
            const __half* __restrict__ d_biases,         // [16] (Conv2a)
            __half* __restrict__ d_outputs,              // [Batch x 14x14x16]
            __half* __restrict__ d_unpooled_vals,        // [Batch x 14x14x16] (Used to store intermediate Conv2a unpooled)
            const __half* __restrict__ d_conv2b_filters, // [16 x 16 x 3 x 3] = 2304 elements (reused from d_fc1_weights)
            const __half* __restrict__ d_conv2b_biases)  // [16] (reused from d_fc1_biases)
        {
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x;

            __shared__ __half s_input[3136];          // 14 * 14 * 16 = 3136 elements = 6272 bytes
            __shared__ __half s_intermediate[3136];   // 14 * 14 * 16 = 3136 elements = 6272 bytes
            __shared__ __half s_filters_a[2304];      // Conv2a filters in shared memory
            __shared__ __half s_filters_b[2304];      // Conv2b filters in shared memory

            // Cooperative load input & filters
            for (int i = tid; i < 3136; i += 256)
            {
                s_input[i] = d_inputs[batch_idx * 3136 + i];
            }
            for (int i = tid; i < 2304; i += 256)
            {
                s_filters_a[i] = d_filters[i];
                s_filters_b[i] = d_conv2b_filters[i];
            }
            __syncthreads();

            __half zero = __float2half(0.0f);

            // Compute Conv2a (Thread mappings: 256 threads cooperative over 3136 outputs)
            for (int out_idx = tid; out_idx < 3136; out_idx += 256)
            {
                int filter_idx = out_idx % 16;
                int spatial_idx = out_idx / 16;
                int out_x = spatial_idx % 14;
                int out_y = spatial_idx / 14;

                __half sum = d_biases[filter_idx];

                #pragma unroll
                for (int c = 0; c < 16; c++)
                {
                    #pragma unroll
                    for (int fy = 0; fy < 3; fy++)
                    {
                        int in_y = out_y + fy - 1;
                        if (in_y < 0 || in_y >= 14) continue;
                        #pragma unroll
                        for (int fx = 0; fx < 3; fx++)
                        {
                            int in_x = out_x + fx - 1;
                            if (in_x < 0 || in_x >= 14) continue;

                            sum += s_input[(in_y * 14 + in_x) * 16 + c] * s_filters_a[filter_idx * 144 + (fy * 3 + fx) * 16 + c];
                        }
                    }
                }

                // Store intermediate Conv2a unpooled outputs
                d_unpooled_vals[batch_idx * 3136 + out_idx] = sum;

                __half activated = __hgt(sum, zero) ? sum : zero;
                s_intermediate[out_idx] = activated;
            }
            __syncthreads();

            // Compute Conv2b & Add Residual
            for (int out_idx = tid; out_idx < 3136; out_idx += 256)
            {
                int filter_idx = out_idx % 16;
                int spatial_idx = out_idx / 16;
                int out_x = spatial_idx % 14;
                int out_y = spatial_idx / 14;

                __half sum = d_conv2b_biases[filter_idx];

                #pragma unroll
                for (int c = 0; c < 16; c++)
                {
                    #pragma unroll
                    for (int fy = 0; fy < 3; fy++)
                    {
                        int in_y = out_y + fy - 1;
                        if (in_y < 0 || in_y >= 14) continue;
                        #pragma unroll
                        for (int fx = 0; fx < 3; fx++)
                        {
                            int in_x = out_x + fx - 1;
                            if (in_x < 0 || in_x >= 14) continue;

                            sum += s_intermediate[(in_y * 14 + in_x) * 16 + c] * s_filters_b[filter_idx * 144 + (fy * 3 + fx) * 16 + c];
                        }
                    }
                }

                // Fused Residual Addition!
                sum += s_input[out_idx];

                // Apply ReLU
                __half activated = __hgt(sum, zero) ? sum : zero;

                // Write final outputs to global VRAM
                d_outputs[batch_idx * 3136 + out_idx] = activated;
            }
        }

        // Fused MaxPool2 Pass (14x14x16 -> 7x7x16 = 784 features)
        extern "C" __global__ void pool2_forward(
            const __half* __restrict__ d_inputs,
            __half* __restrict__ d_outputs)
        {
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x;

            if (tid >= 784) return;

            int filter_idx = tid % 16;
            int spatial_idx = tid / 16;
            int out_x = spatial_idx % 7;
            int out_y = spatial_idx / 7;

            int in_x_base = out_x * 2;
            int in_y_base = out_y * 2;

            __half max_val = __float2half(-1e9f);

            #pragma unroll
            for (int py = 0; py < 2; py++)
            {
                #pragma unroll
                for (int px = 0; px < 2; px++)
                {
                    int in_x = in_x_base + px;
                    int in_y = in_y_base + py;
                    __half val = d_inputs[(batch_idx * 196 + in_y * 14 + in_x) * 16 + filter_idx];
                    if (__hgt(val, max_val))
                    {
                        max_val = val;
                    }
                }
            }

            d_outputs[batch_idx * 784 + tid] = max_val;
        }

        // FC2 Linear Forward Pass (784 features -> 10 classes)
        extern "C" __global__ void fc2_forward(
            const __half* __restrict__ d_inputs,
            const __half* __restrict__ d_weights,
            const __half* __restrict__ d_biases,
            __half* __restrict__ d_outputs)
        {
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x;

            __shared__ __half s_input[784];
            for (int i = tid; i < 784; i += 256)
            {
                s_input[i] = d_inputs[batch_idx * 784 + i];
            }
            __syncthreads();

            if (tid < 10)
            {
                __half sum = d_biases[tid];
                #pragma unroll 4
                for (int i = 0; i < 784; i++)
                {
                    sum += s_input[i] * d_weights[i * 10 + tid];
                }
                d_outputs[batch_idx * 10 + tid] = sum;
            }
        }

        // FC2 Backward Pass (Activation gradients and bias gradients only)
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

            __half zero = __float2half(0.0f);
            for (int i = tid; i < 784; i += 256)
            {
                __half sum_val = zero;
                #pragma unroll
                for (int c = 0; c < 10; c++)
                {
                    sum_val += s_grad[c] * d_fc2_weights[i * 10 + c];
                }
                d_fc2_in_grad[batch_idx * 784 + i] = sum_val;
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
            const __half* __restrict__ d_fc2_inputs,   // [BatchSize x 784]
            __half* __restrict__ d_fc2_weights_grad,   // [784 x 10]
            const int* __restrict__ d_step)
        {
            const int input_idx = blockIdx.x; // 0..783
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
                float x_val = __half2float(d_fc2_inputs[b * 784 + input_idx]);

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

        // Fused MaxPool2 Backward Pass
        extern "C" __global__ void pool2_backward(
            const __half* __restrict__ d_pool2_out_grad,   // [Batch x 784]
            const __half* __restrict__ d_pool2_out_val,    // [Batch x 784]
            const __half* __restrict__ d_conv2_out,        // [Batch x 14x14x16]
            __half* __restrict__ d_conv2_out_grad)         // [Batch x 14x14x16]
        {
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x;

            if (tid >= 784) return;

            int filter_idx = tid % 16;
            int spatial_idx = tid / 16;
            int out_x = spatial_idx % 7;
            int out_y = spatial_idx / 7;

            int in_x_base = out_x * 2;
            int in_y_base = out_y * 2;

            __half grad = d_pool2_out_grad[batch_idx * 784 + tid];
            __half val = d_pool2_out_val[batch_idx * 784 + tid];
            __half zero = __float2half(0.0f);

            #pragma unroll
            for (int py = 0; py < 2; py++)
            {
                #pragma unroll
                for (int px = 0; px < 2; px++)
                {
                    int in_x = in_x_base + px;
                    int in_y = in_y_base + py;
                    int in_idx = (batch_idx * 196 + in_y * 14 + in_x) * 16 + filter_idx;
                    __half in_val = d_conv2_out[in_idx];
                    if (__heq(in_val, val) && __hgt(val, zero))
                    {
                        d_conv2_out_grad[in_idx] = grad;
                    }
                    else
                    {
                        d_conv2_out_grad[in_idx] = zero;
                    }
                }
            }
        }

        // Conv2b Backward Pass
        extern "C" __global__ void conv2_backward(
            const __half* __restrict__ d_conv2_out_grad,         // [Batch x 14x14x16]
            const __half* __restrict__ d_conv2_out_val,          // [Batch x 14x14x16] (Conv2b outputs)
            const __half* __restrict__ d_intermediate_val,       // [Batch x 14x14x16] (Conv2a active outputs)
            const __half* __restrict__ d_conv1_out,              // [Batch x 14x14x16]
            const __half* __restrict__ d_conv2_filters,          // unused
            __half* __restrict__ d_conv2_filters_grad,           // unused
            __half* __restrict__ d_conv2_biases_grad,            // unused
            __half* __restrict__ d_conv1_out_grad,               // unused
            const __half* __restrict__ d_conv2b_filters,         // [16 x 16 x 3 x 3] = 2304 elements (Conv2b filters)
            __half* __restrict__ d_conv2b_filters_grad,          // [16 x 16 x 9] (Conv2b filters grad)
            __half* __restrict__ d_conv2b_biases_grad,           // [16] (Conv2b biases grad)
            __half* __restrict__ d_intermediate_grad)             // [Batch x 14x14x16] (Intermediate output grad)
        {
            #ifndef CONV2_CHUNKS
            #define CONV2_CHUNKS 16
            #endif
            #define CONV2_BATCH_PER_CHUNK (BATCH_SIZE / CONV2_CHUNKS)

            const int filter_idx = blockIdx.x / CONV2_CHUNKS; // 0..15 (Conv2b output channel)
            const int chunk_idx = blockIdx.x % CONV2_CHUNKS;
            const int tid = threadIdx.x;

            __shared__ __half s_intermediate_val[3136];   // 14 * 14 * 16 = 3136 elements
            __shared__ __half s_conv2b_grad[196];         // 14x14 = 196 elements
            __shared__ __half s_filter_grad[144];         // 16 * 9 = 144 elements
            __shared__ __half s_bias_grad;

            __half zero = __float2half(0.0f);

            for (int i = tid; i < 144; i += 128)
            {
                s_filter_grad[i] = zero;
            }
            if (tid == 0)
            {
                s_bias_grad = zero;
            }
            __syncthreads();

            const int start_b = chunk_idx * CONV2_BATCH_PER_CHUNK;
            const int end_b = start_b + CONV2_BATCH_PER_CHUNK;

            int c_arr[2], fx_arr[2], fy_arr[2];
            int i_count = 0;
            for (int i = tid; i < 144; i += 128)
            {
                c_arr[i_count] = i % 16;
                fx_arr[i_count] = (i / 16) % 3;
                fy_arr[i_count] = i / 48;
                i_count++;
            }

            for (int b = start_b; b < end_b; b++)
            {
                for (int i = tid; i < 3136; i += 128)
                {
                    __half val = d_intermediate_val[b * 3136 + i];
                    s_intermediate_val[i] = __hgt(val, zero) ? val : zero;
                }
                for (int i = tid; i < 196; i += 128)
                {
                    int out_idx = i * 16 + filter_idx;
                    __half out_grad = d_conv2_out_grad[b * 3136 + out_idx];
                    __half out_val = d_conv2_out_val[b * 3136 + out_idx];
                    // ReLU backward on Conv2b
                    s_conv2b_grad[i] = __hgt(out_val, zero) ? out_grad : zero;
                }
                __syncthreads();

                // Accumulate weights gradient
                int idx = 0;
                for (int i = tid; i < 144; i += 128)
                {
                    int c = c_arr[idx];
                    int fx = fx_arr[idx];
                    int fy = fy_arr[idx];
                    idx++;

                    __half w_grad = zero;
                    #pragma unroll
                    for (int y = 0; y < 14; y++)
                    {
                        int in_y = y + fy - 1;
                        if (in_y < 0 || in_y >= 14) continue;
                        #pragma unroll
                        for (int x = 0; x < 14; x++)
                        {
                            int in_x = x + fx - 1;
                            if (in_x < 0 || in_x >= 14) continue;

                            w_grad += s_conv2b_grad[y * 14 + x] * s_intermediate_val[(in_y * 14 + in_x) * 16 + c];
                        }
                    }
                    s_filter_grad[i] += w_grad;
                }

                if (tid == 0)
                {
                    __half b_grad = zero;
                    for (int i = 0; i < 196; i++)
                    {
                        b_grad += s_conv2b_grad[i];
                    }
                    s_bias_grad += b_grad;
                }

                // Backpropagate to intermediate active activations
                for (int i = tid; i < 3136; i += 128)
                {
                    int c = i % 16;
                    int spatial_idx = i / 16;
                    int ix = spatial_idx % 14;
                    int iy = spatial_idx / 14;

                    __half sum_grad = zero;
                    #pragma unroll
                    for (int fy = 0; fy < 3; fy++)
                    {
                        int y = iy - fy + 1;
                        if (y < 0 || y >= 14) continue;
                        #pragma unroll
                        for (int fx = 0; fx < 3; fx++)
                        {
                            int x = ix - fx + 1;
                            if (x < 0 || x >= 14) continue;

                            int f_idx = filter_idx * 144 + (fy * 3 + fx) * 16 + c;
                            sum_grad += s_conv2b_grad[y * 14 + x] * d_conv2b_filters[f_idx];
                        }
                    }
                    if (__half2float(sum_grad) != 0.0f)
                    {
                        atomicAdd(&d_intermediate_grad[b * 3136 + i], sum_grad);
                    }
                }
                __syncthreads();
            }

            for (int i = tid; i < 144; i += 128)
            {
                atomicAdd(&d_conv2b_filters_grad[filter_idx * 144 + i], s_filter_grad[i]);
            }
            if (tid == 0)
            {
                atomicAdd(&d_conv2b_biases_grad[filter_idx], s_bias_grad);
            }
        }

        // Conv2a Backward Pass (including Fused Residual Addition!)
        extern "C" __global__ void fc1_backward(
            const __half* __restrict__ d_intermediate_grad,      // [Batch x 14x14x16] (From Conv2b backward)
            const __half* __restrict__ d_intermediate_unpooled,  // [Batch x 14x14x16] (Conv2a unpooled outputs)
            const __half* __restrict__ d_conv1_out,              // [Batch x 14x14x16] (ResBlock inputs)
            const __half* __restrict__ d_conv2a_filters,         // [16 x 16 x 3 x 3] = 2304 elements (Conv2a filters)
            __half* __restrict__ d_conv2a_filters_grad,          // [16 x 16 x 9] (Conv2a filters grad)
            __half* __restrict__ d_conv2a_biases_grad,            // [16] (Conv2a biases grad)
            __half* __restrict__ d_conv1_out_grad,               // [Batch x 14x14x16] (Output input grads)
            const __half* __restrict__ d_conv2b_out_grad)        // [Batch x 14x14x16] (Residual shortcut grad)
        {
            #ifndef CONV2_CHUNKS
            #define CONV2_CHUNKS 16
            #endif
            #define CONV2_BATCH_PER_CHUNK (BATCH_SIZE / CONV2_CHUNKS)

            const int filter_idx = blockIdx.x / CONV2_CHUNKS; // 0..15 (Conv2a output channel)
            const int chunk_idx = blockIdx.x % CONV2_CHUNKS;
            const int tid = threadIdx.x;

            __shared__ __half s_conv1_out[3136];          // 14 * 14 * 16 = 3136 elements
            __shared__ __half s_conv2a_grad[196];         // 14x14 = 196 elements
            __shared__ __half s_filter_grad[144];         // 16 * 9 = 144 elements
            __shared__ __half s_bias_grad;

            __half zero = __float2half(0.0f);

            for (int i = tid; i < 144; i += 128)
            {
                s_filter_grad[i] = zero;
            }
            if (tid == 0)
            {
                s_bias_grad = zero;
            }
            __syncthreads();

            const int start_b = chunk_idx * CONV2_BATCH_PER_CHUNK;
            const int end_b = start_b + CONV2_BATCH_PER_CHUNK;

            int c_arr[2], fx_arr[2], fy_arr[2];
            int i_count = 0;
            for (int i = tid; i < 144; i += 128)
            {
                c_arr[i_count] = i % 16;
                fx_arr[i_count] = (i / 16) % 3;
                fy_arr[i_count] = i / 48;
                i_count++;
            }

            for (int b = start_b; b < end_b; b++)
            {
                for (int i = tid; i < 3136; i += 128)
                {
                    s_conv1_out[i] = d_conv1_out[b * 3136 + i];
                }
                for (int i = tid; i < 196; i += 128)
                {
                    int out_idx = i * 16 + filter_idx;
                    __half out_grad = d_intermediate_grad[b * 3136 + out_idx];
                    __half out_unpooled = d_intermediate_unpooled[b * 3136 + out_idx];
                    // ReLU backward on Conv2a
                    s_conv2a_grad[i] = __hgt(out_unpooled, zero) ? out_grad : zero;
                }
                __syncthreads();

                // Accumulate weights gradient
                int idx = 0;
                for (int i = tid; i < 144; i += 128)
                {
                    int c = c_arr[idx];
                    int fx = fx_arr[idx];
                    int fy = fy_arr[idx];
                    idx++;

                    __half w_grad = zero;
                    #pragma unroll
                    for (int y = 0; y < 14; y++)
                    {
                        int in_y = y + fy - 1;
                        if (in_y < 0 || in_y >= 14) continue;
                        #pragma unroll
                        for (int x = 0; x < 14; x++)
                        {
                            int in_x = x + fx - 1;
                            if (in_x < 0 || in_x >= 14) continue;

                            w_grad += s_conv2a_grad[y * 14 + x] * s_conv1_out[(in_y * 14 + in_x) * 16 + c];
                        }
                    }
                    s_filter_grad[i] += w_grad;
                }

                if (tid == 0)
                {
                    __half b_grad = zero;
                    for (int i = 0; i < 196; i++)
                    {
                        b_grad += s_conv2a_grad[i];
                    }
                    s_bias_grad += b_grad;
                }

                // Backpropagate to inputs & Fused Residual Addition!
                for (int i = tid; i < 3136; i += 128)
                {
                    int c = i % 16;
                    int spatial_idx = i / 16;
                    int ix = spatial_idx % 14;
                    int iy = spatial_idx / 14;

                    __half sum_grad = zero;
                    #pragma unroll
                    for (int fy = 0; fy < 3; fy++)
                    {
                        int y = iy - fy + 1;
                        if (y < 0 || y >= 14) continue;
                        #pragma unroll
                        for (int fx = 0; fx < 3; fx++)
                        {
                            int x = ix - fx + 1;
                            if (x < 0 || x >= 14) continue;

                            int f_idx = filter_idx * 144 + (fy * 3 + fx) * 16 + c;
                            sum_grad += s_conv2a_grad[y * 14 + x] * d_conv2a_filters[f_idx];
                        }
                    }

                    // Add shortcut gradient (exactly once) and accumulate conv2a gradients using atomicAdd
                    __half shortcut_grad = d_conv2b_out_grad[b * 3136 + i];
                    if (filter_idx == 0 && __half2float(shortcut_grad) != 0.0f)
                    {
                        atomicAdd(&d_conv1_out_grad[b * 3136 + i], shortcut_grad);
                    }
                    if (__half2float(sum_grad) != 0.0f)
                    {
                        atomicAdd(&d_conv1_out_grad[b * 3136 + i], sum_grad);
                    }
                }
                __syncthreads();
            }

            for (int i = tid; i < 144; i += 128)
            {
                atomicAdd(&d_conv2a_filters_grad[filter_idx * 144 + i], s_filter_grad[i]);
            }
            if (tid == 0)
            {
                atomicAdd(&d_conv2a_biases_grad[filter_idx], s_bias_grad);
            }
        }

        // Conv1 Backward Pass (14x14x16 -> 28x28 1-bit input gradients)
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
            const int tid = threadIdx.y * 16 + threadIdx.x; // 16x16 = 256 threads

            __shared__ __half s_filter_grad[9];
            __shared__ __half s_bias_grad;
            __shared__ __half s_grad[14][14];
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
                for (int i = tid; i < 196; i += 256)
                {
                    int gy = i / 14;
                    int gx = i % 14;

                    int out_idx = (b * 196 + i) * 16 + filter_idx;
                    __half out_grad = d_conv1_out_grad[out_idx];
                    __half out_val = d_conv1_out_val[out_idx];

                    __half grad = zero;
                    if (__hgt(out_val, zero))
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
                    for (int y = 0; y < 14; y++)
                    {
                        int shift_y = y * 2 + fy + dy;
                        uint32_t row_bits = 0;
                        if (shift_y >= 0 && shift_y < 28)
                        {
                            row_bits = s_image[shift_y];
                        }
                        #pragma unroll
                        for (int x = 0; x < 14; x++)
                        {
                            int img_x = x * 2 + fx + dx;
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
                    for (int y = 0; y < 14; y++)
                    {
                        for (int x = 0; x < 14; x++)
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
