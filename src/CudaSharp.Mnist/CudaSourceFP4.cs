namespace CudaSharp.Mnist;

public static partial class Program
{
    public static string CudaSourceFP4 =>
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

        // EXPERT BIT MAGIC: Convert 1 bit directly to __half without any float operations.
        // 0x3C00 is 1.0 in FP16.
        __device__ __forceinline__ __half bit_to_half(uint32_t bit)
        {
            unsigned short h_val = bit * 0x3C00; 
            return __ushort_as_half(h_val);
        }

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
            if (blockIdx.y > 0) return;
            int tid = threadIdx.y * blockDim.x + threadIdx.x;
            if (tid >= 128) return;

            const int batch_idx = blockIdx.x;
            const int warp_id = tid / 32;
            const int lane_id = tid % 32;

            __shared__ __half s_A[16][32];
            __shared__ __half s_B[32][576];
            __shared__ uint32_t s_image[28];

            int batchOffset = ((*d_step) % BATCHES_PER_EPOCH) * BATCH_SIZE;

            for (int i = tid; i < 400; i += 128) {
                int r = i / 25;
                int c = i % 25;
                s_A[r][c] = d_filters[i];
            }
            for (int i = tid; i < 512; i += 128) {
                int r = i / 32;
                int c = i % 32;
                if (c >= 25) s_A[r][c] = __float2half(0.0f);
            }
            for (int i = tid; i < 28; i += 128) {
                s_image[i] = d_inputs[(batchOffset + batch_idx) * 28 + i];
            }

            int seed = batch_idx + *d_step;
            int dx = (is_training == 1) ? ((seed * 1103515245 + 12345) % 3 - 1) : 0;
            int dy = (is_training == 1) ? (((seed * 1103515245 + 12345) / 3) % 3 - 1) : 0;

            __syncthreads();

            for (int i = tid; i < 18432; i += 128) {
                int k = i / 576;
                int n = i % 576;

                if (k >= 25) {
                    s_B[k][n] = __float2half(0.0f);
                } else {
                    int fy = k / 5;
                    int fx = k % 5;
                    int out_y = n / 24;
                    int out_x = n % 24;

                    int in_y = out_y + fy + dy;
                    int in_x = out_x + fx + dx;

                    __half val = __float2half(0.0f);
                    if (in_y >= 0 && in_y < 28 && in_x >= 0 && in_x < 28) {
                        uint32_t pixel = (s_image[in_y] >> in_x) & 1u;
                        val = __float2half((float)pixel);
                    }
                    s_B[k][n] = val;
                }
            }
            __syncthreads();

            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag_0;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag_1;
            wmma::load_matrix_sync(a_frag_0, &s_A[0][0], 32);
            wmma::load_matrix_sync(a_frag_1, &s_A[0][16], 32);

            __shared__ float s_C_float[16][576];

            for (int t = warp_id; t < 36; t += 4) {
                int n_offset = t * 16;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag_0;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag_1;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
            
            wmma::fill_fragment(c_frag, 0.0f);

            wmma::load_matrix_sync(b_frag_0, &s_B[0][n_offset], 576);
            wmma::load_matrix_sync(b_frag_1, &s_B[16][n_offset], 576);

            wmma::mma_sync(c_frag, a_frag_0, b_frag_0, c_frag);
            wmma::mma_sync(c_frag, a_frag_1, b_frag_1, c_frag);

            wmma::store_matrix_sync(&s_C_float[0][n_offset], c_frag, 576, wmma::mem_row_major);
        }
        __syncthreads();

        for (int i = tid; i < 2304; i += 128) {
            int c = i / 144;
            int pool_spatial = i % 144;
            int p_y = pool_spatial / 12;
            int p_x = pool_spatial % 12;

            __half max_v = __float2half(-1e9f);
            for (int pdy = 0; pdy < 2; pdy++) {
                for (int pdx = 0; pdx < 2; pdx++) {
                    int n = (p_y * 2 + pdy) * 24 + (p_x * 2 + pdx);
                    __half val = __float2half(s_C_float[c][n]) + d_biases[c];
                    d_unpooled_vals[(batch_idx * 576 + n) * 16 + c] = val;
                    
                    __half act = gelu(val);
                    if (__hgt(act, max_v)) max_v = act;
                }
            }
            d_outputs[batch_idx * 2304 + pool_spatial * 16 + c] = max_v;
        }
        }

        extern "C" __global__ void conv2_forward(
            const __half* __restrict__ d_inputs,
            const __half* __restrict__ d_filters,
            const __half* __restrict__ d_biases,
            __half* __restrict__ d_outputs,
            __half* __restrict__ d_unpooled_vals)
        {
            int tid = threadIdx.x;
            if (tid >= 128) return;

            const int batch_idx = blockIdx.x;
            const int warp_id = tid / 32;
            const int lane_id = tid % 32;

            __shared__ __half s_B[400][64];
            __shared__ __half s_A[16][400];

            for (int i = tid; i < 6400; i += 128) {
                int r = i / 400;
                int c = i % 400;
                s_A[r][c] = d_filters[i];
            }

            for (int i = tid; i < 25600; i += 128) {
                int k = i / 64;
                int n = i % 64;

                int in_c = k % 16;
                int f_spatial = k / 16;
                int fy = f_spatial / 5;
                int fx = f_spatial % 5;

                int out_y = n / 8;
                int out_x = n % 8;

                int in_y = out_y + fy;
                int in_x = out_x + fx;

                s_B[k][n] = d_inputs[batch_idx * 2304 + (in_y * 12 + in_x) * 16 + in_c];
            }

            __syncthreads();

            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, __half> c_frag;
            wmma::fill_fragment(c_frag, __float2half(0.0f));

            int n_offset = warp_id * 16;

            #pragma unroll 1
            for (int k_offset = 0; k_offset < 400; k_offset += 16) {
                wmma::load_matrix_sync(a_frag, &s_A[0][k_offset], 400);
                wmma::load_matrix_sync(b_frag, &s_B[k_offset][n_offset], 64);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

            __shared__ __half s_C[16][64];
            wmma::store_matrix_sync(&s_C[0][n_offset], c_frag, 64, wmma::mem_row_major);
            __syncthreads();

            if (tid < 256) {
                int c = tid / 16;
                int pool_spatial = tid % 16;
                int p_y = pool_spatial / 4;
                int p_x = pool_spatial % 4;

                __half max_v = __float2half(-1e9f);
                for (int dy = 0; dy < 2; dy++) {
                    for (int dx = 0; dx < 2; dx++) {
                        int n = (p_y * 2 + dy) * 8 + (p_x * 2 + dx);
                        __half val = s_C[c][n] + d_biases[c];
                        d_unpooled_vals[(batch_idx * 64 + n) * 16 + c] = val; 
                        
                        __half act = gelu(val);
                        if (__hgt(act, max_v)) max_v = act;
                    }
                }
                d_outputs[batch_idx * 256 + pool_spatial * 16 + c] = max_v;
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
            const int batch_idx = blockIdx.x; // We use block for batch, so 1 block = 1 sample! Wait! 
            // In CudaSharp config, fc1 gridX is BATCH_SIZE (128). blockX is 128.
            // If blockIdx.x is batch_idx, then we are doing 1 sample per block.
            // We shouldn't use WMMA if we only do 1 sample per block because M=1.
            // Let's fallback to parallel vector dot product if we are locked to BatchSize blocks!
            // Wait, I can just write the standard dot product but perfectly aligned.
            const int tid = threadIdx.x;

            __shared__ __half s_input[256];
            
            // Cooperatively load the 256 inputs for this sample
            if (tid < 256) {
                s_input[tid] = d_inputs[batch_idx * 256 + tid];
            }
            __syncthreads();

            // Each thread computes 1 output feature
            if (tid < 128)
            {
                __half sum = d_biases[tid];
                #pragma unroll 4
                for (int i = 0; i < 256; i++)
                {
                    sum += s_input[i] * d_weights[i * 128 + tid];
                }
                d_unpooled_vals[batch_idx * 128 + tid] = sum;
                d_outputs[batch_idx * 128 + tid] = gelu(sum);
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

            __shared__ __half s_input[128];
            if (tid < 128)
            {
                s_input[tid] = d_inputs[batch_idx * 128 + tid];
            }
            __syncthreads();

            if (tid < 10)
            {
                __half sum = d_biases[tid];
                #pragma unroll 4
                for (int i = 0; i < 128; i++)
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
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x; // 0..255

            __shared__ __half s_grad[128];
            __shared__ __half s_weights[256][128];

            if (tid < 128)
            {
                s_grad[tid] = d_fc1_out_grad[batch_idx * 128 + tid];
            }

            int warp_id = tid / 32;
            int lane_id = tid % 32;
            int warp_row_start = warp_id * 32;
            
            int total_warp_elements = 32 * 128;
            #pragma unroll
            for (int i = lane_id; i < total_warp_elements; i += 32)
            {
                int r = i / 128;
                int c = i % 128;
                s_weights[warp_row_start + r][c] = d_fc1_weights[(warp_row_start + r) * 128 + c];
            }
            __syncthreads();

            __half sum_grad = __float2half(0.0f);
            #pragma unroll
            for (int c = 0; c < 128; c++)
            {
                sum_grad += s_grad[c] * s_weights[tid][c];
            }

            d_conv2_out_grad[batch_idx * 256 + tid] = sum_grad;

            if (tid < 128)
            {
                __half b_grad = s_grad[tid];
                if (__half2float(b_grad) != 0.0f)
                {
                    atomicAdd(&d_fc1_biases_grad[tid], b_grad);
                }
            }
        }

        extern "C" __global__ void fc1_backward_weights(
            const __half* __restrict__ d_fc1_out_grad,
            const __half* __restrict__ d_fc1_outputs,
            const __half* __restrict__ d_fc1_inputs,
            __half* __restrict__ d_fc1_weights_grad)
        {
            const int input_idx = blockIdx.x; // 0..255
            const int tid = threadIdx.x;      // 0..63

            __shared__ float s_accum[64][128];

            #pragma unroll
            for (int c = 0; c < 128; c++)
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
                for (int c = 0; c < 128; c++)
                {
                    s_accum[tid][c] += x_val * __half2float(d_fc1_out_grad[b * 128 + c]);
                }
            }
            __syncthreads();

            for (int stride = 32; stride > 0; stride >>= 1)
            {
                if (tid < stride)
                {
                    #pragma unroll
                    for (int c = 0; c < 128; c++)
                    {
                        s_accum[tid][c] += s_accum[tid + stride][c];
                    }
                }
                __syncthreads();
            }

            if (tid == 0)
            {
                #pragma unroll
                for (int c = 0; c < 128; c++)
                {
                    d_fc1_weights_grad[input_idx * 128 + c] = __float2half(s_accum[0][c]);
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

            __shared__ __half s_conv1_out[2304];           
            __shared__ __half s_filter_grad[400];         
            __shared__ __half s_bias_grad;
            __shared__ __half s_grad[8][8];

            __half zero = __float2half(0.0f);

            for (int i = tid; i < 400; i += 128)
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

            int c_arr[4], fx_arr[4], fy_arr[4];
            int i_count = 0;
            for (int i = tid; i < 400; i += 128)
            {
                c_arr[i_count] = i % 16;
                fx_arr[i_count] = (i / 16) % 5;
                fy_arr[i_count] = i / 80;
                i_count++;
            }

            for (int b = start_b; b < end_b; b++)
            {
                for (int i = tid; i < 2304; i += 128)
                {
                    s_conv1_out[i] = d_conv1_out[b * 2304 + i];
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

                int idx = 0;
                for (int i = tid; i < 400; i += 128)
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
                            w_grad += g * s_conv1_out[(in_y * 12 + in_x) * 16 + c];
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

                for (int i = tid; i < 2304; i += 128)
                {
                    int c = i % 16;
                    int spatial_idx = i / 16;
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
                                int f_idx = filter_idx * 400 + (fy * 5 + fx) * 16 + c;
                                sum_grad += s_grad[y][x] * d_conv2_filters[f_idx];
                            }
                        }
                    }
                    if (__half2float(sum_grad) != 0.0f)
                    {
                        atomicAdd(&d_conv1_out_grad[b * 2304 + i], sum_grad);
                    }
                }
                __syncthreads();
            }

            for (int i = tid; i < 400; i += 128)
            {
                atomicAdd(&d_conv2_filters_grad[filter_idx * 400 + i], s_filter_grad[i]);
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
            const int tid = threadIdx.y * 16 + threadIdx.x; 

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

                for (int i = tid; i < 576; i += 256)
                {
                    int gy = i / 24;
                    int gx = i % 24;
                    int px = gx / 2;
                    int py = gy / 2;
                    int pool_idx = (py * 12 + px) * 16 + filter_idx;

                    __half out_grad = d_conv1_out_grad[b * 2304 + pool_idx];
                    __half out_val = d_conv1_out_val[b * 2304 + pool_idx];
                    __half my_val = d_conv1_unpooled_vals[(b * 576 + gy * 24 + gx) * 16 + filter_idx];

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