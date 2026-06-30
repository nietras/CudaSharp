namespace CudaSharp.Mnist;

public static partial class Program
{
    public static string CudaSourceV23 =>
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

                    int unpooled_idx = (batch_idx * 2304 + cy * 24 + cx) * 16 + filter_idx;
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

                        int unpooled_idx = (batch_idx * 256 + cy * 8 + cx) * 16 + filter_idx;
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
            if (blockIdx.x > 0) return;

            const int warp_id = threadIdx.x / 32;
            const int lane_id = threadIdx.x % 32;

            if (warp_id >= 8) return;

            __shared__ __half s_tile[8][16][16];

            #pragma unroll
            for (int c = 0; c < 8; c++)
            {
                wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc_frag;
                wmma::fill_fragment(acc_frag, __float2half(0.0f));

                #pragma unroll
                for (int k = 0; k < 16; k++)
                {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;

                    wmma::load_matrix_sync(a_frag, d_inputs + warp_id * 4096 + k * 16, 256);
                    wmma::load_matrix_sync(b_frag, d_weights + k * 2048 + c * 16, 128);

                    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
                }

                wmma::store_matrix_sync(&s_tile[warp_id][0][0], acc_frag, 16, wmma::mem_row_major);
                __syncwarp();

                #pragma unroll
                for (int i = 0; i < 8; i++)
                {
                    int idx = lane_id * 8 + i;
                    int r_local = idx / 16;
                    int c_local = idx % 16;

                    __half val = s_tile[warp_id][r_local][c_local];
                    __half bias = d_biases[c * 16 + c_local];
                    __half sum = val + bias;

                    int global_row = warp_id * 16 + r_local;
                    int global_col = c * 16 + c_local;

                    d_unpooled_vals[global_row * 128 + global_col] = sum;
                    d_outputs[global_row * 128 + global_col] = gelu(sum);
                }
                __syncwarp();
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
            const unsigned char* __restrict__ d_labels,
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
            if (blockIdx.x > 0) return;

            const int tid = threadIdx.x;
            const int warp_id = tid / 32;
            const int lane_id = tid % 32;

            // Bias gradient accumulation (run by first 128 threads)
            if (tid < 128)
            {
                float sum = 0.0f;
                #pragma unroll 8
                for (int b = 0; b < 128; b++)
                {
                    sum += __half2float(d_fc1_out_grad[b * 128 + tid]);
                }
                d_fc1_biases_grad[tid] = __float2half(sum);
            }

            if (warp_id >= 8) return;

            __shared__ __half s_tile[8][16][16];

            #pragma unroll
            for (int c = 0; c < 16; c++)
            {
                wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc_frag;
                wmma::fill_fragment(acc_frag, __float2half(0.0f));

                #pragma unroll
                for (int k = 0; k < 8; k++)
                {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;

                    wmma::load_matrix_sync(a_frag, d_fc1_out_grad + warp_id * 2048 + k * 16, 128);
                    wmma::load_matrix_sync(b_frag, d_fc1_weights + c * 2048 + k * 16, 128);

                    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
                }

                wmma::store_matrix_sync(&s_tile[warp_id][0][0], acc_frag, 16, wmma::mem_row_major);
                __syncwarp();

                #pragma unroll
                for (int i = 0; i < 8; i++)
                {
                    int idx = lane_id * 8 + i;
                    int r_local = idx / 16;
                    int c_local = idx % 16;

                    __half val = s_tile[warp_id][r_local][c_local];

                    int global_row = warp_id * 16 + r_local;
                    int global_col = c * 16 + c_local;

                    d_conv2_out_grad[global_row * 256 + global_col] = val;
                }
                __syncwarp();
            }
        }

        extern "C" __global__ void fc1_backward_weights(
            const __half* __restrict__ d_fc1_out_grad,
            const __half* __restrict__ d_fc1_outputs,
            const __half* __restrict__ d_fc1_inputs,
            __half* __restrict__ d_fc1_weights_grad)
        {
            if (blockIdx.x >= 64) return;

            const int warp_id = threadIdx.x / 32;
            const int lane_id = threadIdx.x % 32;

            if (warp_id >= 2) return;

            __shared__ __half s_tile[2][16][16];

            const int tile_idx = blockIdx.x * 2 + warp_id;
            const int r = tile_idx / 8;
            const int c = tile_idx % 8;

            wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc_frag;
            wmma::fill_fragment(acc_frag, __float2half(0.0f));

            #pragma unroll
            for (int k = 0; k < 8; k++)
            {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;

                wmma::load_matrix_sync(a_frag, d_fc1_inputs + k * 4096 + r * 16, 256);
                wmma::load_matrix_sync(b_frag, d_fc1_out_grad + k * 2048 + c * 16, 128);

                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

            wmma::store_matrix_sync(&s_tile[warp_id][0][0], acc_frag, 16, wmma::mem_row_major);
            __syncwarp();

            #pragma unroll
            for (int i = 0; i < 8; i++)
            {
                int idx = lane_id * 8 + i;
                int r_local = idx / 16;
                int c_local = idx % 16;

                __half val = s_tile[warp_id][r_local][c_local];

                int global_row = r * 16 + r_local;
                int global_col = c * 16 + c_local;

                d_fc1_weights_grad[global_row * 128 + global_col] = val;
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
