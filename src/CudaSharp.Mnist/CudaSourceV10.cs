namespace CudaSharp.Mnist;

public static partial class Program
{
    public static string CudaSourceV10 =>
        """
        #include <cuda_fp16.h>

        typedef unsigned int uint32_t;

        #ifndef BATCH_SIZE
        #define BATCH_SIZE 256
        #endif
        #define FILTER1_SIZE 5
        #define FILTER2_SIZE 5
        #define INPUT_SIZE 28
        #define POOL1_SIZE 12
        #define POOL2_SIZE 4
        #define FC1_INPUTS 256
        #define FC1_OUTPUTS 120
        #define FC2_OUTPUTS 10

        #ifndef BATCHES_PER_EPOCH
        #define BATCHES_PER_EPOCH 200
        #endif
        #ifndef TOTAL_STEPS
        #define TOTAL_STEPS 300
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

        // Fast FP16 GELU approximation
        __device__ inline __half gelu(__half x)
        {
            float val = __half2float(x);
            float c = 0.79788456f; // sqrt(2/pi)
            float tanh_arg = c * (val + 0.044715f * val * val * val);
            float g = 0.5f * val * (1.0f + tanhf(tanh_arg));
            return __float2half(g);
        }

        // Fused GELU backpropagation gradient helper
        __device__ inline __half d_gelu(__half x, __half dy)
        {
            float val = __half2float(x);
            float g_dy = __half2float(dy);
            float c = 0.79788456f; // sqrt(2/pi)
            float tanh_arg = c * (val + 0.044715f * val * val * val);
            float t = tanhf(tanh_arg);
            float sech2 = 1.0f - t * t;
            float dtanh = c * (1.0f + 3.0f * 0.044715f * val * val) * sech2;
            float derivative = 0.5f * (1.0f + t) + 0.5f * val * dtanh;
            return __float2half(g_dy * derivative);
        }

        // Dummy empty kernels to satisfy C# Module loading structure
        extern "C" __global__ void fused_forward(
            const uint32_t* d_inputs, const __half* d_conv1_filters, const __half* d_conv1_biases,
            const __half* d_conv2_filters, const __half* d_conv2_biases, const __half* d_fc2_weights,
            const __half* d_fc2_biases, __half* d_conv1_out, __half* d_conv1_unpooled,
            __half* d_conv2_out, __half* d_conv2_unpooled, __half* d_fc2_out,
            const int* d_step, int is_training) {}

        extern "C" __global__ void fused_backward(
            const __half* d_fc2_out, const int* d_labels, const __half* d_conv2_out,
            const __half* d_fc2_weights, __half* d_fc2_weights_grad, __half* d_fc2_biases_grad,
            const __half* d_conv2_unpooled, const __half* d_conv1_out, const __half* d_conv2_filters,
            __half* d_conv2_filters_grad, __half* d_conv2_biases_grad, const __half* d_conv1_unpooled,
            const uint32_t* d_inputs, __half* d_conv1_filters_grad, __half* d_conv1_biases_grad,
            const int* d_step) {}

        // Conv1 Forward Pass (1-bit input -> 6 filters, 12x12 output after MaxPool + GELU)
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

                    int unpooled_idx = (batch_idx * 576 + cy * 24 + cx) * 6 + filter_idx;
                    d_unpooled_vals[unpooled_idx] = sum;

                    __half activated = gelu(sum);
                    if (__hgt(activated, max_val))
                    {
                        max_val = activated;
                    }
                }
            }

            const int out_idx = batch_idx * 864 + (out_y * 12 + out_x) * 6 + filter_idx;
            d_outputs[out_idx] = max_val;
        }

        // Conv2 Forward Pass (6 channels -> 16 filters, 4x4 output after MaxPool + GELU)
        extern "C" __global__ void conv2_forward(
            const __half* __restrict__ d_inputs,
            const __half* __restrict__ d_filters,
            const __half* __restrict__ d_biases,
            __half* __restrict__ d_outputs,
            __half* __restrict__ d_unpooled_vals)
        {
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x;

            __shared__ __half s_input[864]; // 12 * 12 * 6 = 864
            __shared__ __half s_filters[2400]; // 16 * 6 * 25 = 2400

            for (int i = tid; i < 864; i += 256)
            {
                s_input[i] = d_inputs[batch_idx * 864 + i];
            }

            for (int i = tid; i < 2400; i += 256)
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
                        for (int c = 0; c < 6; c++)
                        {
                            #pragma unroll
                            for (int fy = 0; fy < 5; fy++)
                            {
                                #pragma unroll
                                for (int fx = 0; fx < 5; fx++)
                                {
                                    int in_x = cx + fx;
                                    int in_y = cy + fy;
                                    sum += s_input[(in_y * 12 + in_x) * 6 + c] * s_filters[filter_idx * 150 + (fy * 5 + fx) * 6 + c];
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

        // FC1 Linear Forward Pass (256 features -> 120 classes, fused with GELU and pre-activation logging)
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
            s_input[tid] = d_inputs[batch_idx * 256 + tid];
            __syncthreads();

            if (tid < 120)
            {
                __half sum = d_biases[tid];
                #pragma unroll
                for (int i = 0; i < 256; i++)
                {
                    sum += s_input[i] * d_weights[i * 120 + tid];
                }
                d_unpooled_vals[batch_idx * 120 + tid] = sum;
                d_outputs[batch_idx * 120 + tid] = gelu(sum);
            }
        }

        // FC2 Linear Forward Pass (120 features -> 10 classes)
        extern "C" __global__ void fc2_forward(
            const __half* __restrict__ d_inputs,
            const __half* __restrict__ d_weights,
            const __half* __restrict__ d_biases,
            __half* __restrict__ d_outputs)
        {
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x;

            __shared__ __half s_input[120];
            if (tid < 120)
            {
                s_input[tid] = d_inputs[batch_idx * 120 + tid];
            }
            __syncthreads();

            if (tid < 10)
            {
                __half sum = d_biases[tid];
                #pragma unroll
                for (int i = 0; i < 120; i++)
                {
                    sum += s_input[i] * d_weights[i * 10 + tid];
                }
                d_outputs[batch_idx * 10 + tid] = sum;
            }
        }

        // FC2 Backward Pass (Activation gradients and bias gradients only, ZERO weights computation!)
        extern "C" __global__ void fc2_backward(
            const __half* __restrict__ d_fc2_outputs,
            const int* __restrict__ d_labels,
            const __half* __restrict__ d_fc1_outputs,
            const __half* __restrict__ d_fc2_weights,
            __half* __restrict__ d_fc2_weights_grad, // unused
            __half* __restrict__ d_fc2_biases_grad,
            __half* __restrict__ d_fc1_out_grad,
            const int* __restrict__ d_step,
            const __half* __restrict__ d_fc1_unpooled) // logged pre-activation values
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

            if (tid < 120)
            {
                __half sum_input_grad = __float2half(0.0f);
                #pragma unroll
                for (int c = 0; c < 10; c++)
                {
                    sum_input_grad += s_grad[c] * d_fc2_weights[tid * 10 + c];
                }
                // Backprop over GELU of FC1
                __half fc1_unpooled = d_fc1_unpooled[batch_idx * 120 + tid];
                d_fc1_out_grad[batch_idx * 120 + tid] = d_gelu(fc1_unpooled, sum_input_grad);
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
            const int* __restrict__ d_labels,         // [TotalImages]
            const __half* __restrict__ d_fc1_outputs,   // [BatchSize x 120]
            __half* __restrict__ d_fc2_weights_grad,   // [120 x 10]
            const int* __restrict__ d_step)
        {
            const int input_idx = blockIdx.x; // 0..119
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
                float x_val = __half2float(d_fc1_outputs[b * 120 + input_idx]);

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

        // FC1 Backward Pass (Activation gradients and bias gradients only, ZERO weights computation!)
        // Fused with SOTA Coalesced Weight Chunk Loading
        extern "C" __global__ void fc1_backward(
            const __half* __restrict__ d_fc1_out_grad, // [BatchSize x 120]
            const __half* __restrict__ d_fc1_outputs,  // unused
            const __half* __restrict__ d_fc2_inputs,   // [BatchSize x 256] (d_conv2Out)
            const __half* __restrict__ d_fc1_weights,   // [256 x 120]
            __half* __restrict__ d_fc1_biases_grad,
            __half* __restrict__ d_conv2_out_grad)     // [BatchSize x 256]
        {
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x; // 0..255

            __shared__ __half s_grad[120];
            __shared__ __half s_weights[256][120];

            // Load s_grad
            if (tid < 120)
            {
                s_grad[tid] = d_fc1_out_grad[batch_idx * 120 + tid];
            }

            int warp_id = tid / 32;
            int lane_id = tid % 32;
            int warp_row_start = warp_id * 32;
            
            // Cooperatively load 32x120 weights for this warp in a fully coalesced manner
            int total_warp_elements = 32 * 120;
            #pragma unroll
            for (int i = lane_id; i < total_warp_elements; i += 32)
            {
                int r = i / 120;
                int c = i % 120;
                s_weights[warp_row_start + r][c] = d_fc1_weights[(warp_row_start + r) * 120 + c];
            }
            __syncthreads();

            __half sum_grad = __float2half(0.0f);
            #pragma unroll
            for (int c = 0; c < 120; c++)
            {
                sum_grad += s_grad[c] * s_weights[tid][c];
            }

            d_conv2_out_grad[batch_idx * 256 + tid] = sum_grad;

            if (tid < 120)
            {
                __half b_grad = s_grad[tid];
                if (__half2float(b_grad) != 0.0f)
                {
                    atomicAdd(&d_fc1_biases_grad[tid], b_grad);
                }
            }
        }

        // FC1 Backward Weights Pass (Zero atomics, block-level parallel reductions!)
        extern "C" __global__ void fc1_backward_weights(
            const __half* __restrict__ d_fc1_out_grad, // [BatchSize x 120]
            const __half* __restrict__ d_fc1_outputs,  // unused
            const __half* __restrict__ d_fc1_inputs,   // [BatchSize x 256] (d_conv2Out)
            __half* __restrict__ d_fc1_weights_grad)   // [256 x 120]
        {
            const int input_idx = blockIdx.x; // 0..255
            const int tid = threadIdx.x;      // 0..63

            __shared__ float s_accum[64][120];

            #pragma unroll
            for (int c = 0; c < 120; c++)
            {
                s_accum[tid][c] = 0.0f;
            }
            __syncthreads();

            #pragma unroll
            for (int i = 0; i < (BATCH_SIZE / 64); i++)
            {
                int b = i * 64 + tid;
                float x_val = __half2float(d_fc1_inputs[b * 256 + input_idx]);
                #pragma unroll
                for (int c = 0; c < 120; c++)
                {
                    s_accum[tid][c] += x_val * __half2float(d_fc1_out_grad[b * 120 + c]);
                }
            }
            __syncthreads();

            for (int stride = 32; stride > 0; stride >>= 1)
            {
                if (tid < stride)
                {
                    #pragma unroll
                    for (int c = 0; c < 120; c++)
                    {
                        s_accum[tid][c] += s_accum[tid + stride][c];
                    }
                }
                __syncthreads();
            }

            if (tid == 0)
            {
                #pragma unroll
                for (int c = 0; c < 120; c++)
                {
                    d_fc1_weights_grad[input_idx * 120 + c] = __float2half(s_accum[0][c]);
                }
            }
        }

        // Conv2 Backward Pass (16 channels -> 6 filters, 8x8 input gradients)
        extern "C" __global__ void conv2_backward(
            const __half* __restrict__ d_conv2_out_grad,         // [Batch x 256]
            const __half* __restrict__ d_conv2_out_val,          // [Batch x 256]
            const __half* __restrict__ d_conv2_unpooled_vals,    // [Batch x 8x8x16]
            const __half* __restrict__ d_conv1_out,              // [Batch x 12x12x6]
            const __half* __restrict__ d_conv2_filters,          // [16 x 6 x 5 x 5] = 2400 elements
            __half* __restrict__ d_conv2_filters_grad,           // [16 x 6 x 25]
            __half* __restrict__ d_conv2_biases_grad,            // [16]
            __half* __restrict__ d_conv1_out_grad)               // [Batch x 12x12x6]
        {
            #ifndef CONV2_CHUNKS
            #define CONV2_CHUNKS 16
            #endif
            #define CONV2_BATCH_PER_CHUNK (BATCH_SIZE / CONV2_CHUNKS)

            const int filter_idx = blockIdx.x / CONV2_CHUNKS; // 0..15 (Conv2 output channel)
            const int chunk_idx = blockIdx.x % CONV2_CHUNKS;
            const int tid = threadIdx.x;

            __shared__ __half s_conv1_out[864];           // 12 * 12 * 6 = 864 elements
            __shared__ __half s_filter_grad[150];         // 6 * 25 = 150 elements
            __shared__ __half s_bias_grad;
            __shared__ __half s_grad[8][8];

            __half zero = __float2half(0.0f);

            for (int i = tid; i < 150; i += 128)
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

            const int start_b = chunk_idx * CONV2_BATCH_PER_CHUNK;
            const int end_b = start_b + CONV2_BATCH_PER_CHUNK;

            int c_arr[2], fx_arr[2], fy_arr[2];
            int i_count = 0;
            for (int i = tid; i < 150; i += 128)
            {
                c_arr[i_count] = i % 6;
                fx_arr[i_count] = (i / 6) % 5;
                fy_arr[i_count] = i / 30;
                i_count++;
            }

            for (int b = start_b; b < end_b; b++)
            {
                for (int i = tid; i < 864; i += 128)
                {
                    s_conv1_out[i] = d_conv1_out[b * 864 + i];
                }

                __half out_grad = zero;
                __half out_val = zero;
                if (tid < 64 && px < 4 && py < 4)
                {
                    int pool_idx = (py * 4 + px) * 16 + filter_idx;
                    out_grad = d_conv2_out_grad[b * 256 + pool_idx];
                    out_val = d_conv2_out_val[b * 256 + pool_idx];
                }

                __half my_val = zero;
                if (tid < 64)
                {
                    my_val = d_conv2_unpooled_vals[(b * 64 + tid) * 16 + filter_idx];
                }

                __half grad = zero;
                if (tid < 64 && __heq(gelu(my_val), out_val))
                {
                    grad = d_gelu(my_val, out_grad);
                }
                if (tid < 64)
                {
                    s_grad[cy][cx] = grad;
                }
                __syncthreads();

                // Accumulate weights gradient
                int idx = 0;
                for (int i = tid; i < 150; i += 128)
                {
                    int c = c_arr[idx];
                    int fx = fx_arr[idx];
                    int fy = fy_arr[idx];
                    idx++;

                    __half w_grad = zero;
                    #pragma unroll
                    for (int y = 0; y < 8; y++)
                    {
                        #pragma unroll
                        for (int x = 0; x < 8; x++)
                        {
                            __half g = s_grad[y][x];
                            int in_x = x + fx;
                            int in_y = y + fy;
                            w_grad += g * s_conv1_out[(in_y * 12 + in_x) * 6 + c];
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

                // Backprop to inputs
                for (int i = tid; i < 864; i += 128)
                {
                    int c = i % 6;
                    int spatial_idx = i / 6;
                    int ix = spatial_idx % 12;
                    int iy = spatial_idx / 12;

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
                                int f_idx = filter_idx * 150 + (fy * 5 + fx) * 6 + c;
                                sum_grad += s_grad[y][x] * d_conv2_filters[f_idx];
                            }
                        }
                    }
                    if (__half2float(sum_grad) != 0.0f)
                    {
                        atomicAdd(&d_conv1_out_grad[b * 864 + i], sum_grad);
                    }
                }
                __syncthreads();
            }

            for (int i = tid; i < 150; i += 128)
            {
                atomicAdd(&d_conv2_filters_grad[filter_idx * 150 + i], s_filter_grad[i]);
            }
            if (tid == 0)
            {
                atomicAdd(&d_conv2_biases_grad[filter_idx], s_bias_grad);
            }
        }

        // Conv1 Backward Pass (6 filters -> 1-bit input gradients)
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

            const int start_b = chunk_idx * CONV1_BATCH_PER_CHUNK;
            const int end_b = start_b + CONV1_BATCH_PER_CHUNK;

            int fx = tid % 5;
            int fy = tid / 5;

            for (int b = start_b; b < end_b; b++)
            {
                if (tid < 28)
                {
                    s_image[tid] = d_inputs[(batchOffset + b) * 28 + tid];
                }

                // Parallel populate s_grad across 256 threads
                for (int i = tid; i < 576; i += 256)
                {
                    int gy = i / 24;
                    int gx = i % 24;
                    int px = gx / 2;
                    int py = gy / 2;
                    int pool_idx = (py * 12 + px) * 6 + filter_idx;

                    __half out_grad = d_conv1_out_grad[b * 864 + pool_idx];
                    __half out_val = d_conv1_out_val[b * 864 + pool_idx];
                    __half my_val = d_conv1_unpooled_vals[(b * 576 + gy * 24 + gx) * 6 + filter_idx];

                    __half grad = zero;
                    if (__heq(gelu(my_val), out_val))
                    {
                        grad = d_gelu(my_val, out_grad);
                    }
                    s_grad[gy][gx] = grad;
                }
                __syncthreads();

                int seed = b + *d_step;
                int dx = (is_training == 1) ? ((seed * 1103515245 + 12345) % 3 - 1) : 0;
                int dy = (is_training == 1) ? (((seed * 1103515245 + 12345) / 3) % 3 - 1) : 0;

                if (tid < 25)
                {
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
