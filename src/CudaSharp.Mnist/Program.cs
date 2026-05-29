using System;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Net.Http;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using CudaSharp;
using static CudaSharp.nvcuda;
using static CudaSharp.nvrtc;

namespace CudaSharp.Mnist;

public unsafe partial class Program
{
    static int BatchSize = 128;
    const int ClassCount = 10;
    const int ImageRows = 28;
    const int ImageCols = 28;
    const int TrainImagesCount = 51200; // 400 batches of size 128
    const int TestImagesCount = 10240;   // Padded to multiple of BatchSize (128 * 80 = 10240)

    static readonly string CudaSourceV1 =
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

            // Parallel perfectly coalesced load of 2,304 input elements (no division/modulo!)
            for (int i = tid; i < 2304; i += 256)
            {
                s_input[i] = d_inputs[batch_idx * 2304 + i];
            }

            // Parallel load of all 2,304 filter weights
            for (int i = tid; i < 2304; i += 256)
            {
                s_filters[i] = d_filters[i];
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
            for (int i = tid; i < 400; i += 256)
            {
                s_input[i] = d_inputs[batch_idx * 400 + i];
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
            s_input[tid] = d_inputs[batch_idx * 256 + tid];
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

        // FC1 Weights Gradient Kernel (chunked batch accumulation!)
        extern "C" __global__ void fc1_backward_weights(
            const float* __restrict__ d_fc1_out_grad, // [BatchSize x 256]
            const float* __restrict__ d_fc1_outputs,  // [BatchSize x 256]
            const float* __restrict__ d_conv2_outputs, // [BatchSize x 400]
            float* __restrict__ d_fc1_weights_grad)   // [400 x 256]
        {
            #define FC1_CHUNKS 8
            #define FC1_BATCH_PER_CHUNK (BATCH_SIZE / FC1_CHUNKS)

            // Grid: 8 x 8 x 8 = 512 blocks.
            // blockIdx.x: row-chunk (0..7, each handles 50 rows)
            // blockIdx.y: col-chunk (0..7, each handles 32 cols)
            // blockIdx.z: batch-chunk (0..7, each handles 32 batch elements)
            // Block size: 128 threads (1D)
            
            const int row_start = blockIdx.x * 50;
            const int col_start = blockIdx.y * 32;
            const int chunk_idx = blockIdx.z;
            const int tid = threadIdx.x;
            
            __shared__ float s_w_grad[50][32];
            
            // Initialize shared memory
            for (int i = tid; i < 1600; i += 128)
            {
                int r = i / 32;
                int c = i % 32;
                s_w_grad[r][c] = 0.0f;
            }
            __syncthreads();
            
            const int start_b = chunk_idx * FC1_BATCH_PER_CHUNK;
            const int end_b = start_b + FC1_BATCH_PER_CHUNK;
            
            for (int b = start_b; b < end_b; b++)
            {
                __shared__ float s_input[50];
                __shared__ float s_out_grad[32];
                
                // Parallel load inputs and out_grads for this batch element
                if (tid < 50)
                {
                    s_input[tid] = d_conv2_outputs[b * 400 + row_start + tid];
                }
                if (tid < 32)
                {
                    float out_val = d_fc1_outputs[b * 256 + col_start + tid];
                    s_out_grad[tid] = out_val > 0.0f ? d_fc1_out_grad[b * 256 + col_start + tid] : 0.0f;
                }
                __syncthreads();
                
                // Accumulate products
                for (int i = tid; i < 1600; i += 128)
                {
                    int r = i / 32;
                    int c = i % 32;
                    s_w_grad[r][c] += s_input[r] * s_out_grad[c];
                }
                __syncthreads();
            }
            
            // Write to global memory using atomicAdd
            for (int i = tid; i < 1600; i += 128)
            {
                int r = i / 32;
                int c = i % 32;
                int global_row = row_start + r;
                int global_col = col_start + c;
                float val = s_w_grad[r][c];
                if (val != 0.0f)
                {
                    atomicAdd(&d_fc1_weights_grad[global_row * 256 + global_col], val);
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
                // Parallel coalesced load of 2,304 elements from d_conv1_out to shared memory (no division/modulo!)
                for (int i = tid; i < 2304; i += 128)
                {
                    s_conv1_out[i] = d_conv1_out[b * 2304 + i];
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
            float max_lr = 0.028f; 
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

    static readonly string CudaSourceV2 =
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

            // Parallel perfectly coalesced load of 1,152 input elements
            for (int i = tid; i < 1152; i += 256)
            {
                s_input[i] = d_inputs[batch_idx * 1152 + i];
            }

            // Parallel load of all 3,200 filter weights
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
            s_input[tid] = d_inputs[batch_idx * 256 + tid];
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
                // Parallel coalesced load of 1,152 elements
                for (int i = tid; i < 1152; i += 128)
                {
                    s_conv1_out[i] = d_conv1_out[b * 1152 + i];
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

    static readonly string CudaSourceV3 =
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

    static readonly string CudaSourceV4 = CudaSourceV1
        .Replace("\r\n", "\n")
        .Replace("#define BATCH_SIZE 128", "#define BATCH_SIZE 128")
        .Replace("#define TOTAL_STEPS 400", "#define TOTAL_STEPS 190")
        .Replace("float max_lr = 0.028f;", "float max_lr = 0.028f;")
        .Replace("float beta1 = 0.7f;", "float beta1 = 0.9f;")
        .Replace("float beta2 = 0.9f;", "float beta2 = 0.999f;")
        .Replace(@"            float pct = (float)step_val / total_steps;
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
            }", @"            float lr = 0.0f;
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
            }");

    public static readonly NetworkConfig ConfigV9 = new()
    {
        Name = "V9",
        CudaSource = CudaSourceV9,
        IsHalf = true,
        IsV7Based = true,
        BatchSize = 256,
        Conv1FilterCount = 6,
        Conv1FilterSize = 5,
        Conv2FilterCount = 16,
        Conv2FilterSize = 5,
        Pool2OutSize = 4,
        HasFC1 = true,
        FC1Outputs = 120,
        FC1Inputs = 256,
        BatchesPerEpoch = 200,
        TotalSteps = 300,
        MaxLR = 0.006f
    };

    public static readonly NetworkConfig ConfigV8 = new()
    {
        Name = "V8",
        CudaSource = CudaSourceV8,
        IsHalf = true,
        IsV7Based = true,
        BatchSize = 256,
        Conv1FilterCount = 16,
        Conv1FilterSize = 3,
        Conv2FilterCount = 16,
        Conv2FilterSize = 3,
        Pool2OutSize = 7,
        HasFC1 = true,
        FC1Outputs = 16,
        FC1Inputs = 144,
        BatchesPerEpoch = 200,
        TotalSteps = 300,
        MaxLR = 0.006f
    };

    public static readonly NetworkConfig ConfigV7 = new()
    {
        Name = "V7",
        CudaSource = CudaSourceV7,
        IsHalf = true,
        IsV7Based = true,
        BatchSize = 256,
        Conv1FilterCount = 8,
        Conv1FilterSize = 3,
        Conv2FilterCount = 16,
        Conv2FilterSize = 3,
        Pool2OutSize = 5,
        HasFC1 = false,
        BatchesPerEpoch = 200,
        TotalSteps = 300,
        MaxLR = 0.006f
    };

    public static readonly NetworkConfig ConfigV6 = new()
    {
        Name = "V6",
        CudaSource = CudaSourceV6,
        IsHalf = true,
        IsV7Based = false,
        BatchSize = 128,
        Conv1FilterCount = 16,
        Conv1FilterSize = 5,
        Conv2FilterCount = 32,
        Conv2FilterSize = 5,
        Pool2OutSize = 4,
        HasFC1 = false,
        BatchesPerEpoch = 240,
        TotalSteps = 240,
        MaxLR = 0.007f
    };

    public static readonly NetworkConfig ConfigV5 = new()
    {
        Name = "V5",
        CudaSource = CudaSourceV5,
        IsHalf = false,
        IsV7Based = false,
        BatchSize = 128,
        Conv1FilterCount = 1,
        Conv1FilterSize = 1,
        Conv2FilterCount = 1,
        Conv2FilterSize = 1,
        Pool2OutSize = 1,
        HasFC1 = true,
        FC1Outputs = 256,
        BatchesPerEpoch = 180,
        TotalSteps = 180,
        MaxLR = 0.009f
    };

    public static readonly NetworkConfig ConfigV4 = new()
    {
        Name = "V4",
        CudaSource = CudaSourceV4,
        IsHalf = false,
        IsV7Based = false,
        BatchSize = 128,
        Conv1FilterCount = 16,
        Conv1FilterSize = 5,
        Conv2FilterCount = 16,
        Conv2FilterSize = 3,
        Pool2OutSize = 5,
        HasFC1 = true,
        FC1Outputs = 256,
        BatchesPerEpoch = 200,
        TotalSteps = 190,
        MaxLR = 0.028f
    };

    public static readonly NetworkConfig ConfigV3 = new()
    {
        Name = "V3",
        CudaSource = CudaSourceV3,
        IsHalf = true,
        IsV7Based = false,
        Conv1FilterCount = 8,
        Conv1FilterSize = 5,
        Conv2FilterCount = 16,
        Conv2FilterSize = 5,
        Pool2OutSize = 4,
        HasFC1 = false,
        BatchSize = 128,
        BatchesPerEpoch = 200,
        TotalSteps = 200,
        MaxLR = 0.05f
    };

    public static readonly NetworkConfig ConfigV1 = new()
    {
        Name = "V1",
        CudaSource = CudaSourceV1,
        IsHalf = false,
        IsV7Based = false,
        Conv1FilterCount = 16,
        Conv1FilterSize = 5,
        Conv2FilterCount = 16,
        Conv2FilterSize = 3,
        Pool2OutSize = 5,
        HasFC1 = true,
        FC1Outputs = 256,
        BatchesPerEpoch = 200,
        TotalSteps = 400,
        MaxLR = 0.028f
    };

    public static readonly NetworkConfig ConfigV2 = new()
    {
        Name = "V2",
        CudaSource = CudaSourceV2,
        IsHalf = false,
        IsV7Based = false,
        Conv1FilterCount = 8,
        Conv1FilterSize = 5,
        Conv2FilterCount = 16,
        Conv2FilterSize = 5,
        Pool2OutSize = 4,
        HasFC1 = false,
        BatchesPerEpoch = 300,
        TotalSteps = 600,
        MaxLR = 0.06f
    };

    public static void Main(string[] args)
    {
        Console.WriteLine("==================================================");
        Console.WriteLine("CudaSharp Ultra-Fast MNIST CNN Training Simulator");
        Console.WriteLine("==================================================");

        string version = "V2";
        bool profile = false;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--version")
            {
                version = args[i + 1].ToUpperInvariant();
            }
            else if (args[i] == "--profile")
            {
                profile = true;
            }
        }
        Console.WriteLine($"[CONFIG] Network Version: {version} (Profile Mode: {profile})");

        NetworkConfig activeConfig;
        if (version.StartsWith("V0") || (version.StartsWith("V") && version.Length > 2 && int.TryParse(version.AsSpan(1), out _)))
        {
            if (int.TryParse(version.AsSpan(1), out int num) && num >= 1 && num <= 99)
            {
                if (num == 1) activeConfig = ConfigV1;
                else if (num == 2) activeConfig = ConfigV2;
                else if (num == 3) activeConfig = ConfigV3;
                else if (num == 4) activeConfig = ConfigV4;
                else if (num == 5) activeConfig = ConfigV5;
                else if (num == 6) activeConfig = ConfigV6;
                else if (num == 7) activeConfig = ConfigV7;
                else if (num == 8) activeConfig = ConfigV8;
                else if (num == 9) activeConfig = ConfigV9;
                else
                {
                    int batchSize = (num % 4) switch
                    {
                        0 => 64,
                        1 => 128,
                        2 => 256,
                        3 => 512,
                        _ => 256
                    };
                    float maxLR = ((num / 4) % 4) switch
                    {
                        0 => 0.003f,
                        1 => 0.006f,
                        2 => 0.009f,
                        3 => 0.012f,
                        _ => 0.006f
                    };
                    int totalSteps = ((num / 16) % 3) switch
                    {
                        0 => 150,
                        1 => 300,
                        2 => 450,
                        _ => 300
                    };
                    activeConfig = new NetworkConfig
                    {
                        Name = version,
                        CudaSource = CudaSourceV7,
                        IsHalf = true,
                        IsV7Based = true,
                        BatchSize = batchSize,
                        Conv1FilterCount = 8,
                        Conv1FilterSize = 3,
                        Conv2FilterCount = 16,
                        Conv2FilterSize = 3,
                        Pool2OutSize = 5,
                        HasFC1 = false,
                        BatchesPerEpoch = 51200 / batchSize,
                        TotalSteps = totalSteps,
                        MaxLR = maxLR
                    };
                }
            }
            else
            {
                throw new ArgumentException($"Unknown version: {version}");
            }
        }
        else
        {
            activeConfig = version switch
            {
                "V1" => ConfigV1,
                "V2" => ConfigV2,
                "V3" => ConfigV3,
                "V4" => ConfigV4,
                "V5" => ConfigV5,
                "V6" => ConfigV6,
                "V7" => ConfigV7,
                "V8" => ConfigV8,
                "V9" => ConfigV9,
                _ => throw new ArgumentException($"Unknown version: {version}")
            };
        }
        BatchSize = activeConfig.BatchSize;

        CuInit.EnsureInit();

        cuDeviceGet(out var device, 0).Ok();
        Span<byte> deviceNameBytes = stackalloc byte[256];
        cuDeviceGetName(deviceNameBytes, 256, device).Ok();
        string deviceName = Encoding.UTF8.GetString(deviceNameBytes).TrimEnd('\0');
        cuDeviceComputeCapability(out var major, out var minor, device).Ok();

        Console.WriteLine($"[DEVICE] Loaded active GPU: {deviceName} (sm_{major}{minor})");

        string dataDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "mnist_data");
        if (!Directory.Exists(dataDir))
        {
            Directory.CreateDirectory(dataDir);
        }

        string trainImagesPath = Path.Combine(dataDir, "train-images-idx3-ubyte.gz");
        string trainLabelsPath = Path.Combine(dataDir, "train-labels-idx1-ubyte.gz");
        string testImagesPath = Path.Combine(dataDir, "t10k-images-idx3-ubyte.gz");
        string testLabelsPath = Path.Combine(dataDir, "t10k-labels-idx1-ubyte.gz");

        EnsureDatasetFile(trainImagesPath, "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz");
        EnsureDatasetFile(trainLabelsPath, "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz");
        EnsureDatasetFile(testImagesPath, "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz");
        EnsureDatasetFile(testLabelsPath, "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz");

        Console.WriteLine("[DATA] Parsing Gzip compressed idx dataset files in-memory...");
        var (h_trainImages, trainImagesLoaded) = ParseImagesGz(trainImagesPath, TrainImagesCount);
        var h_trainLabels = ParseLabelsGz(trainLabelsPath, TrainImagesCount);
        var (h_testImages, testImagesLoaded) = ParseImagesGz(testImagesPath, TestImagesCount);
        var h_testLabels = ParseLabelsGz(testLabelsPath, TestImagesCount);

        Console.WriteLine($"[DATA] Loaded {trainImagesLoaded} train images and {testImagesLoaded} test images successfully!");

        int conv1Chunks = Math.Max(16, activeConfig.BatchSize / 8);
        int conv2Chunks = Math.Max(16, activeConfig.BatchSize / 8);

        Console.WriteLine("[JIT] Compiling fused CUDA kernels...");
        nvrtcCreateProgram(out var program, activeConfig.CudaSource, "mnist_kernels", 0, [], []).Ok();
        CUcontext context = default;
        try
        {
            var optionsList = new System.Collections.Generic.List<string>
            {
                $"--gpu-architecture=compute_{major}{minor}",
                "--std=c++17",
                "--use_fast_math",
                $"-DBATCH_SIZE={activeConfig.BatchSize}",
                $"-DBATCHES_PER_EPOCH={activeConfig.BatchesPerEpoch}",
                $"-DTOTAL_STEPS={activeConfig.TotalSteps}",
                $"-DMAX_LR={activeConfig.MaxLR}f",
                $"-DFC1_OUTPUTS={activeConfig.FC1Outputs}",
                $"-DCONV1_CHUNKS={conv1Chunks}",
                $"-DCONV2_CHUNKS={conv2Chunks}"
            };
            string cudaPath = Environment.GetEnvironmentVariable("CUDA_PATH");
            if (!string.IsNullOrEmpty(cudaPath))
            {
                optionsList.Add($"-I{Path.Combine(cudaPath, "include")}");
            }

            var options = optionsList.ToArray();
            var optionBytes = new byte[options.Length][];
            var optionPointers = stackalloc byte*[options.Length];
            for (int i = 0; i < options.Length; i++)
            {
                optionBytes[i] = Encoding.UTF8.GetBytes($"{options[i]}\0");
                fixed (byte* optPtr = optionBytes[i])
                {
                    optionPointers[i] = optPtr;
                }
            }

            var compileResult = nvrtcCompileProgram(program, options.Length, optionPointers);
            if (compileResult.IsError())
            {
                nvrtcGetProgramLogSize(program, out var logSize).Ok();
                var logBuffer = new byte[logSize];
                nvrtcGetProgramLog(program, logBuffer).Ok();
                throw new InvalidOperationException($"NVRTC Compilation failed:\n{Encoding.UTF8.GetString(logBuffer)}");
            }

            nvrtcGetPTXSize(program, out var ptxSize).Ok();
            var ptx = new byte[ptxSize];
            nvrtcGetPTX(program, ptx).Ok();

            Console.WriteLine("[DEVICE] Creating CUDA context and command stream...");
            cuCtxCreate(out context, CUctx_flags.CU_CTX_SCHED_SPIN, device).Ok();
            cuCtxSetCurrent(context).Ok();
            cuStreamCreate(out var stream, 0).Ok();
            int isTrainingTrue = 0;
            int isTrainingFalse = 0;

            cuModuleLoadData(out var module, ptx).Ok();

            cuModuleGetFunction(out var f_clear, module, "clear_gradient").Ok();
            cuModuleGetFunction(out var f_conv1, module, "conv1_forward").Ok();
            cuModuleGetFunction(out var f_conv2, module, "conv2_forward").Ok();
            cuModuleGetFunction(out var f_fc2, module, "fc2_forward").Ok();

            cuModuleGetFunction(out var f_fc2_bwd, module, "fc2_backward").Ok();
            cuModuleGetFunction(out var f_conv2_bwd, module, "conv2_backward").Ok();
            cuModuleGetFunction(out var f_conv1_bwd, module, "conv1_backward").Ok();

            cuModuleGetFunction(out var f_adam, module, "adam_update").Ok();

            CUfunction f_fc2_bwd_weights = default;
            if (activeConfig.IsV7Based)
            {
                cuModuleGetFunction(out f_fc2_bwd_weights, module, "fc2_backward_weights").Ok();
            }

            CUfunction f_fc1 = default, f_fc1_bwd = default, f_fc1_bwd_weights = default;
            if (activeConfig.HasFC1)
            {
                string fc1FwdName = activeConfig.Name == "V8" ? "pool2_forward" : "fc1_forward";
                string fc1BwdName = activeConfig.Name == "V8" ? "pool2_backward" : "fc1_backward";
                cuModuleGetFunction(out f_fc1, module, fc1FwdName).Ok();
                cuModuleGetFunction(out f_fc1_bwd, module, fc1BwdName).Ok();
                if (activeConfig.Name == "V8")
                {
                    cuModuleGetFunction(out f_fc1_bwd_weights, module, "fc1_backward").Ok();
                }
                else
                {
                    cuModuleGetFunction(out f_fc1_bwd_weights, module, "fc1_backward_weights").Ok();
                }
            }

            uint conv1BlockX = (uint)activeConfig.Pool1OutSize;
            uint conv1BlockY = (uint)activeConfig.Pool1OutSize;
            uint conv1BwdBlockX = activeConfig.IsV7Based ? 16u : 24u;
            uint conv1BwdBlockY = activeConfig.IsV7Based ? 16u : 24u;

            Console.WriteLine("[MEM] Allocating GPU memory buffers...");

            cuMemAlloc(out var d_trainImages, (nuint)(h_trainImages.Length * sizeof(uint))).Ok();
            cuMemAlloc(out var d_trainLabels, (nuint)(h_trainLabels.Length * sizeof(int))).Ok();
            cuMemAlloc(out var d_testImages, (nuint)(h_testImages.Length * sizeof(uint))).Ok();
            cuMemAlloc(out var d_testLabels, (nuint)(h_testLabels.Length * sizeof(int))).Ok();

            fixed (uint* pTrainImages = h_trainImages)
            fixed (int* pTrainLabels = h_trainLabels)
            fixed (uint* pTestImages = h_testImages)
            fixed (int* pTestLabels = h_testLabels)
            {
                cuMemcpyHtoD(d_trainImages, (IntPtr)pTrainImages, (nuint)(h_trainImages.Length * sizeof(uint))).Ok();
                cuMemcpyHtoD(d_trainLabels, (IntPtr)pTrainLabels, (nuint)(h_trainLabels.Length * sizeof(int))).Ok();
                cuMemcpyHtoD(d_testImages, (IntPtr)pTestImages, (nuint)(h_testImages.Length * sizeof(uint))).Ok();
                cuMemcpyHtoD(d_testLabels, (IntPtr)pTestLabels, (nuint)(h_testLabels.Length * sizeof(int))).Ok();
            }

            int conv1FilterCount = activeConfig.Conv1FilterCount;
            int conv2FilterCount = activeConfig.Conv2FilterCount;
            int totalParamElements = activeConfig.TotalParamElements;

            int elementSize = activeConfig.IsHalf ? sizeof(ushort) : sizeof(float);
            nuint paramBytes = (nuint)(totalParamElements * elementSize);
            cuMemAlloc(out var d_allParams, paramBytes).Ok();
            cuMemAlloc(out var d_allParamGrads, paramBytes).Ok();
            cuMemAlloc(out var d_allParamM, paramBytes).Ok();
            cuMemAlloc(out var d_allParamV, paramBytes).Ok();

            cuMemsetD8(d_allParamGrads, 0, paramBytes).Ok();
            cuMemsetD8(d_allParamM, 0, paramBytes).Ok();
            cuMemsetD8(d_allParamV, 0, paramBytes).Ok();

            var conv1Param = activeConfig.GetParam("conv1");
            var conv2Param = activeConfig.GetParam("conv2");
            var fc2Param = activeConfig.GetParam("fc2");

            CUdeviceptr d_conv1Filters = SliceDevicePtr(d_allParams, conv1Param.WeightOffset, elementSize);
            CUdeviceptr d_conv1Biases = SliceDevicePtr(d_allParams, conv1Param.BiasOffset, elementSize);
            CUdeviceptr d_conv2Filters = SliceDevicePtr(d_allParams, conv2Param.WeightOffset, elementSize);
            CUdeviceptr d_conv2Biases = SliceDevicePtr(d_allParams, conv2Param.BiasOffset, elementSize);
            CUdeviceptr d_fc2Weights = SliceDevicePtr(d_allParams, fc2Param.WeightOffset, elementSize);
            CUdeviceptr d_fc2Biases = SliceDevicePtr(d_allParams, fc2Param.BiasOffset, elementSize);

            CUdeviceptr d_conv1FiltersGrad = SliceDevicePtr(d_allParamGrads, conv1Param.WeightOffset, elementSize);
            CUdeviceptr d_conv1BiasesGrad = SliceDevicePtr(d_allParamGrads, conv1Param.BiasOffset, elementSize);
            CUdeviceptr d_conv2FiltersGrad = SliceDevicePtr(d_allParamGrads, conv2Param.WeightOffset, elementSize);
            CUdeviceptr d_conv2BiasesGrad = SliceDevicePtr(d_allParamGrads, conv2Param.BiasOffset, elementSize);
            CUdeviceptr d_fc2WeightsGrad = SliceDevicePtr(d_allParamGrads, fc2Param.WeightOffset, elementSize);
            CUdeviceptr d_fc2BiasesGrad = SliceDevicePtr(d_allParamGrads, fc2Param.BiasOffset, elementSize);

            CUdeviceptr d_fc1Weights = default, d_fc1Biases = default;
            CUdeviceptr d_fc1WeightsGrad = default, d_fc1BiasesGrad = default;

            if (activeConfig.HasFC1)
            {
                var fc1Param = activeConfig.GetParam("fc1");
                d_fc1Weights = SliceDevicePtr(d_allParams, fc1Param.WeightOffset, elementSize);
                d_fc1Biases = SliceDevicePtr(d_allParams, fc1Param.BiasOffset, elementSize);
                d_fc1WeightsGrad = SliceDevicePtr(d_allParamGrads, fc1Param.WeightOffset, elementSize);
                d_fc1BiasesGrad = SliceDevicePtr(d_allParamGrads, fc1Param.BiasOffset, elementSize);
            }

            int conv1OutSize = BatchSize * activeConfig.Conv1OutPerSample;
            int conv1UnpooledSize = BatchSize * activeConfig.Conv1UnpooledPerSample;
            int conv2OutSize = BatchSize * activeConfig.Conv2OutPerSample;
            int conv2UnpooledSize = BatchSize * activeConfig.Conv2UnpooledPerSample;

            cuMemAlloc(out var d_conv1Out, (nuint)(conv1OutSize * elementSize)).Ok();
            cuMemAlloc(out var d_conv1Unpooled, (nuint)(conv1UnpooledSize * elementSize)).Ok();
            cuMemAlloc(out var d_conv2Out, (nuint)(conv2OutSize * elementSize)).Ok();
            cuMemAlloc(out var d_conv2Unpooled, (nuint)(conv2UnpooledSize * elementSize)).Ok();

            CUdeviceptr d_fc1Out = default;
            if (activeConfig.HasFC1)
            {
                int fc1OutCount = activeConfig.Name == "V8" ? 784 : activeConfig.FC1Outputs;
                cuMemAlloc(out d_fc1Out, (nuint)(BatchSize * fc1OutCount * elementSize)).Ok();
            }
            cuMemAlloc(out var d_fc2Out, (nuint)(BatchSize * 10 * elementSize)).Ok();

            int conv1OutGradSize = BatchSize * activeConfig.Conv1OutPerSample;
            cuMemAlloc(out var d_conv1OutGrad, (nuint)(conv1OutGradSize * elementSize)).Ok();

            CUdeviceptr d_fc1OutGrad = default, d_conv2OutGrad = default, d_intermediateGrad = default;
            CUdeviceptr d_fc2InGrad = default;
            if (activeConfig.Name == "V8")
            {
                cuMemAlloc(out d_fc2InGrad, (nuint)(BatchSize * 784 * elementSize)).Ok();
                cuMemAlloc(out d_conv2OutGrad, (nuint)(BatchSize * 3136 * elementSize)).Ok();
                cuMemAlloc(out d_fc1OutGrad, (nuint)(BatchSize * 784 * elementSize)).Ok();
                cuMemAlloc(out d_intermediateGrad, (nuint)(BatchSize * 3136 * elementSize)).Ok();
            }
            else if (activeConfig.HasFC1)
            {
                cuMemAlloc(out d_fc1OutGrad, (nuint)(BatchSize * activeConfig.FC1Outputs * elementSize)).Ok();
                cuMemAlloc(out d_conv2OutGrad, (nuint)(BatchSize * activeConfig.FC1Inputs * elementSize)).Ok();
            }
            else
            {
                cuMemAlloc(out d_fc2InGrad, (nuint)(BatchSize * activeConfig.FC2Inputs * elementSize)).Ok();
            }

            cuMemAlloc(out var d_step, (nuint)sizeof(int)).Ok();

            uint fc2BlockSize = activeConfig.Name == "V5" ? 128u : 256u;
            int fc1Chunks = 8;

            InitializeModelParameters(activeConfig, d_conv1Filters, d_conv1Biases, d_conv2Filters, d_conv2Biases, d_fc1Weights, d_fc1Biases, d_fc2Weights, d_fc2Biases, 42);

            Console.WriteLine("[GRAPH] Capturing training loop into a single optimized CUDA Graph...");

            cuGraphCreate(out var epochGraph, 0).Ok();

            int trainStepCount = activeConfig.TotalSteps;
            int testStepCount = TestImagesCount / BatchSize;

            int localClearGradElements = conv1OutGradSize;
            int localTotalParamsCount = totalParamElements;

            CUgraphNode lastNode = default;
            var currentDependencies = new CUgraphNode[1];

            CUdeviceptr d_fc2In = activeConfig.HasFC1 ? d_fc1Out : d_conv2Out;
            CUdeviceptr d_fc2InGrad_kernel = activeConfig.HasFC1 ? d_fc1OutGrad : d_fc2InGrad;
            CUdeviceptr d_conv2BwdInGrad = activeConfig.HasFC1 ? d_conv2OutGrad : d_fc2InGrad;

            var clearGradParams = new void*[] { &d_conv1OutGrad, &localClearGradElements };
            var conv1Params = new void*[]
            {
                &d_trainImages, &d_conv1Filters, &d_conv1Biases,
                &d_conv1Out, &d_conv1Unpooled, &d_step, &isTrainingTrue
            };
            var conv2Params = activeConfig.Name == "V8"
                ? new void*[]
                {
                    &d_conv1Out, &d_conv2Filters, &d_conv2Biases,
                    &d_conv2Out, &d_conv2Unpooled, &d_fc1Weights, &d_fc1Biases
                }
                : new void*[]
                {
                    &d_conv1Out, &d_conv2Filters, &d_conv2Biases,
                    &d_conv2Out, &d_conv2Unpooled
                };
            var fc1Params = activeConfig.Name == "V8"
                ? new void*[] { &d_conv2Out, &d_fc1Out }
                : (activeConfig.Name == "V5"
                    ? new void*[] { &d_trainImages, &d_fc1Weights, &d_fc1Biases, &d_fc1Out, &d_step, &isTrainingTrue }
                    : new void*[] { &d_conv2Out, &d_fc1Weights, &d_fc1Biases, &d_fc1Out });
            var fc2Params = new void*[]
            {
                &d_fc2In, &d_fc2Weights, &d_fc2Biases, &d_fc2Out
            };
            var fc2BwdParams = new void*[]
            {
                &d_fc2Out, &d_trainLabels, &d_fc2In, &d_fc2Weights,
                &d_fc2WeightsGrad, &d_fc2BiasesGrad, &d_fc2InGrad_kernel, &d_step
            };
            var fc2BwdWeightsParams = new void*[]
            {
                &d_fc2Out, &d_trainLabels, &d_fc2In, &d_fc2WeightsGrad, &d_step
            };
            var fc1BwdParams = activeConfig.Name == "V8"
                ? new void*[] { &d_fc1OutGrad, &d_fc1Out, &d_conv2Out, &d_conv2OutGrad }
                : new void*[]
                {
                    &d_fc1OutGrad, &d_fc1Out, &d_conv2Out, &d_fc1Weights,
                    &d_fc1BiasesGrad, &d_conv2OutGrad
                };
            var fc1BwdWeightsParams = activeConfig.Name == "V8"
                ? new void*[]
                {
                    &d_intermediateGrad, &d_conv2Unpooled, &d_conv1Out, &d_conv2Filters,
                    &d_conv2FiltersGrad, &d_conv2BiasesGrad, &d_conv1OutGrad, &d_conv2OutGrad
                }
                : (activeConfig.Name == "V5"
                    ? new void*[] { &d_fc1OutGrad, &d_fc1Out, &d_trainImages, &d_fc1WeightsGrad, &d_step }
                    : new void*[] { &d_fc1OutGrad, &d_fc1Out, &d_conv2Out, &d_fc1WeightsGrad });
            var conv2BwdParams = activeConfig.Name == "V8"
                ? new void*[]
                {
                    &d_conv2OutGrad, &d_conv2Out, &d_conv2Unpooled, &d_conv1Out,
                    &d_conv2Filters, &d_conv2FiltersGrad, &d_conv2BiasesGrad, &d_conv1OutGrad,
                    &d_fc1Weights, &d_fc1WeightsGrad, &d_fc1BiasesGrad, &d_intermediateGrad
                }
                : new void*[]
                {
                    &d_conv2BwdInGrad, &d_conv2Out, &d_conv2Unpooled, &d_conv1Out,
                    &d_conv2Filters, &d_conv2FiltersGrad, &d_conv2BiasesGrad,
                    &d_conv1OutGrad
                };
            var conv1BwdParams = new void*[]
            {
                &d_conv1OutGrad, &d_conv1Out, &d_conv1Unpooled, &d_trainImages,
                &d_conv1FiltersGrad, &d_conv1BiasesGrad, &d_step, &isTrainingTrue
            };
            var adamParams = new void*[]
            {
                &d_allParams, &d_allParamGrads, &d_allParamM, &d_allParamV,
                &localTotalParamsCount, &d_step
            };

            for (int step = 0; step < trainStepCount; step++)
            {
                var depsClear = step == 0
                    ? Array.Empty<CUgraphNode>() : [lastNode];
                lastNode = AddKernelNode(epochGraph, depsClear, f_clear,
                    (uint)((conv1OutGradSize + 255) / 256), 1u, 1u,
                    256u, 1u, 1u, clearGradParams);

                if (activeConfig.Name == "V8")
                {
                    currentDependencies[0] = lastNode;
                    int intermediateGradSize = BatchSize * 3136;
                    var clearIntermediateParams = new void*[] { &d_intermediateGrad, &intermediateGradSize };
                    lastNode = AddKernelNode(epochGraph, currentDependencies, f_clear,
                        (uint)((intermediateGradSize + 255) / 256), 1u, 1u,
                        256u, 1u, 1u, clearIntermediateParams);
                }

                if (activeConfig.Name != "V5")
                {
                    currentDependencies[0] = lastNode;
                    lastNode = AddKernelNode(epochGraph, currentDependencies,
                        f_conv1, (uint)BatchSize, (uint)conv1FilterCount, 1u,
                        conv1BlockX, conv1BlockY, 1u, conv1Params);

                    currentDependencies[0] = lastNode;
                    lastNode = AddKernelNode(epochGraph, currentDependencies,
                        f_conv2, (uint)BatchSize, 1u, 1u,
                        256u, 1u, 1u, conv2Params);
                }

                if (activeConfig.HasFC1)
                {
                    currentDependencies[0] = lastNode;
                    uint fc1BlockSize = activeConfig.Name == "V8" ? 784u : (activeConfig.Name == "V9" ? 256u : 128u);
                    lastNode = AddKernelNode(epochGraph, currentDependencies,
                        f_fc1, (uint)BatchSize, 1u, 1u,
                        fc1BlockSize, 1u, 1u, fc1Params);
                }



                currentDependencies[0] = lastNode;
                lastNode = AddKernelNode(epochGraph, currentDependencies,
                    f_fc2, (uint)BatchSize, 1u, 1u,
                    fc2BlockSize, 1u, 1u, fc2Params);

                currentDependencies[0] = lastNode;
                lastNode = AddKernelNode(epochGraph, currentDependencies,
                    f_fc2_bwd, (uint)BatchSize, 1u, 1u,
                    fc2BlockSize, 1u, 1u, fc2BwdParams);

                if (activeConfig.IsV7Based)
                {
                    currentDependencies[0] = lastNode;
                    uint fc2BwdWGridX = activeConfig.Name == "V8" ? 784u : (activeConfig.Name == "V9" ? 120u : 400u);
                    lastNode = AddKernelNode(epochGraph, currentDependencies,
                        f_fc2_bwd_weights, fc2BwdWGridX, 1u, 1u,
                        128u, 1u, 1u, fc2BwdWeightsParams);
                }

                if (activeConfig.HasFC1)
                {
                    currentDependencies[0] = lastNode;
                    if (activeConfig.Name == "V8")
                    {
                        lastNode = AddKernelNode(epochGraph, currentDependencies,
                            f_fc1_bwd, (uint)BatchSize, 1u, 1u,
                            784u, 1u, 1u, fc1BwdParams);
                    }
                    else if (activeConfig.Name == "V5")
                    {
                        lastNode = AddKernelNode(epochGraph, currentDependencies,
                            f_fc1_bwd, 1u, 1u, 1u,
                            128u, 1u, 1u, fc1BwdParams);
                    }
                    else
                    {
                        lastNode = AddKernelNode(epochGraph, currentDependencies,
                            f_fc1_bwd, (uint)BatchSize, 1u, 1u,
                            256u, 1u, 1u, fc1BwdParams);
                    }

                    if (activeConfig.Name != "V8")
                    {
                        currentDependencies[0] = lastNode;
                        if (activeConfig.Name == "V9")
                        {
                            lastNode = AddKernelNode(epochGraph, currentDependencies,
                                f_fc1_bwd_weights, 256u, 1u, 1u,
                                64u, 1u, 1u, fc1BwdWeightsParams);
                        }
                        else if (activeConfig.Name == "V5")
                        {
                            lastNode = AddKernelNode(epochGraph, currentDependencies,
                                f_fc1_bwd_weights, 784u, 1u, 1u,
                                128u, 1u, 1u, fc1BwdWeightsParams);
                        }
                        else
                        {
                            lastNode = AddKernelNode(epochGraph, currentDependencies,
                                f_fc1_bwd_weights, 8u, 8u, (uint)fc1Chunks,
                                128u, 1u, 1u, fc1BwdWeightsParams);
                        }
                    }
                }

                if (activeConfig.Name != "V5")
                {
                    currentDependencies[0] = lastNode;
                    lastNode = AddKernelNode(epochGraph, currentDependencies,
                        f_conv2_bwd, (uint)conv2FilterCount * (uint)conv2Chunks, 1u, 1u,
                        128u, 1u, 1u, conv2BwdParams);

                    if (activeConfig.Name == "V8")
                    {
                        currentDependencies[0] = lastNode;
                        lastNode = AddKernelNode(epochGraph, currentDependencies,
                            f_fc1_bwd_weights, (uint)conv2FilterCount * (uint)conv2Chunks, 1u, 1u,
                            128u, 1u, 1u, fc1BwdWeightsParams);
                    }

                    currentDependencies[0] = lastNode;
                    lastNode = AddKernelNode(epochGraph, currentDependencies,
                        f_conv1_bwd, (uint)conv1FilterCount * (uint)conv1Chunks, 1u, 1u,
                        conv1BwdBlockX, conv1BwdBlockY, 1u, conv1BwdParams);
                }

                currentDependencies[0] = lastNode;
                lastNode = AddKernelNode(epochGraph, currentDependencies,
                    f_adam, (uint)((totalParamElements + 255) / 256), 1u, 1u,
                    256u, 1u, 1u, adamParams);
            }

            Console.WriteLine("[GRAPH] Instantiating executable graph...");
            Span<byte> graphLogBuffer = stackalloc byte[2048];
            var instantiateResult = cuGraphInstantiate(out var epochGraphExec,
                epochGraph, out var errorNode, graphLogBuffer,
                (nuint)graphLogBuffer.Length);
            if (instantiateResult.IsError())
            {
                var log = Encoding.UTF8.GetString(graphLogBuffer).TrimEnd('\0');
                throw new InvalidOperationException(
                    $"Graph instantiation failed with {instantiateResult.ToStringFast()} at node {errorNode.Value}:\n{log}");
            }

            int bestSeed = 42;
            double bestAccuracy = 0.0;
            double trainingTime = 0.0;
            int[] seedsToTry = [42, 1337, 7, 100, 2026, 12345, 999, 8888,
                12, 1111, 19, 37, 73, 97, 101, 223, 317, 503, 709, 883];

            var h_fcOut = new float[BatchSize * ClassCount];
            var h_fcOutHalf = new Half[BatchSize * ClassCount];

            var argsTestConv1 = stackalloc void*[]
            {
                &d_testImages, &d_conv1Filters, &d_conv1Biases,
                &d_conv1Out, &d_conv1Unpooled, &d_step, &isTrainingFalse
            };
            var argsConv2 = stackalloc void*[]
            {
                &d_conv1Out, &d_conv2Filters, &d_conv2Biases,
                &d_conv2Out, &d_conv2Unpooled
            };
            var argsFc2V2 = stackalloc void*[]
            {
                &d_conv2Out, &d_fc2Weights, &d_fc2Biases, &d_fc2Out
            };
            var argsFc1V1 = stackalloc void*[]
            {
                &d_conv2Out, &d_fc1Weights, &d_fc1Biases, &d_fc1Out
            };
            var argsFc1V5 = stackalloc void*[]
            {
                &d_testImages, &d_fc1Weights, &d_fc1Biases, &d_fc1Out, &d_step, &isTrainingFalse
            };
            var argsFc2V1 = stackalloc void*[]
            {
                &d_fc1Out, &d_fc2Weights, &d_fc2Biases, &d_fc2Out
            };

            var measuredTimes = new System.Collections.Generic.List<double>();
            var measuredAccuracies = new System.Collections.Generic.List<double>();

            for (int sIndex = 0; sIndex < 4; sIndex++)
            {
                int currentSeed = seedsToTry[sIndex];
                bool isWarmup = sIndex == 0;

                if (isWarmup)
                {
                    Console.WriteLine($"[TRAIN] Launching Warmup Run (Seed: {currentSeed})...");
                }
                else
                {
                    Console.WriteLine($"[TRAIN] Launching Measured Run #{sIndex} (Seed: {currentSeed})...");
                }

                cuMemsetD8(d_allParamGrads, 0, paramBytes).Ok();
                cuMemsetD8(d_allParamM, 0, paramBytes).Ok();
                cuMemsetD8(d_allParamV, 0, paramBytes).Ok();

                InitializeModelParameters(activeConfig, d_conv1Filters, d_conv1Biases, d_conv2Filters, d_conv2Biases, d_fc1Weights, d_fc1Biases, d_fc2Weights, d_fc2Biases, currentSeed);

                int zero = 0;
                cuMemcpyHtoD(d_step, (IntPtr)(&zero), (nuint)sizeof(int)).Ok();





                var stopwatch = Stopwatch.StartNew();
                if (profile)
                {
                    CUevent startEvent, stopEvent;
                    cuEventCreate(out startEvent, 0).Ok();
                    cuEventCreate(out stopEvent, 0).Ok();

                    var clearTimes = new System.Collections.Generic.List<float>();
                    var conv1Times = new System.Collections.Generic.List<float>();
                    var conv2Times = new System.Collections.Generic.List<float>();
                    var fc1Times = new System.Collections.Generic.List<float>();
                    var fc2Times = new System.Collections.Generic.List<float>();
                    var fc2BwdTimes = new System.Collections.Generic.List<float>();
                    var fc2BwdWTimes = new System.Collections.Generic.List<float>();
                    var fc1BwdTimes = new System.Collections.Generic.List<float>();
                    var fc1BwdWTimes = new System.Collections.Generic.List<float>();
                    var conv2BwdTimes = new System.Collections.Generic.List<float>();
                    var conv1BwdTimes = new System.Collections.Generic.List<float>();
                    var adamTimes = new System.Collections.Generic.List<float>();

                    float MeasureKernel(CUfunction function, uint gridX, uint gridY, uint gridZ, uint blockX, uint blockY, uint blockZ, void*[] args)
                    {
                        fixed (void** pArgs = args)
                        {
                            cuEventRecord(startEvent, stream).Ok();
                            cuLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ, 0u, stream, pArgs, null).Ok();
                            cuEventRecord(stopEvent, stream).Ok();
                            cuStreamSynchronize(stream).Ok();
                            float ms;
                            cuEventElapsedTime(out ms, startEvent, stopEvent).Ok();
                            return ms;
                        }
                    }

                    for (int step = 0; step < trainStepCount; step++)
                    {
                        clearTimes.Add(MeasureKernel(f_clear, (uint)((conv1OutGradSize + 255) / 256), 1u, 1u, 256u, 1u, 1u, clearGradParams));
                        if (activeConfig.Name == "V8")
                        {
                            int intermediateGradSize = BatchSize * 3136;
                            var clearIntermediateParams = new void*[] { &d_intermediateGrad, &intermediateGradSize };
                            clearTimes.Add(MeasureKernel(f_clear, (uint)((intermediateGradSize + 255) / 256), 1u, 1u, 256u, 1u, 1u, clearIntermediateParams));
                        }

                        if (activeConfig.Name != "V5")
                        {
                            conv1Times.Add(MeasureKernel(f_conv1, (uint)BatchSize, (uint)conv1FilterCount, 1u, conv1BlockX, conv1BlockY, 1u, conv1Params));
                            conv2Times.Add(MeasureKernel(f_conv2, (uint)BatchSize, 1u, 1u, 256u, 1u, 1u, conv2Params));
                        }

                        if (activeConfig.HasFC1)
                        {
                            uint fc1FwdBlockSize = activeConfig.Name == "V8" ? 784u : (activeConfig.Name == "V9" ? 256u : 128u);
                            fc1Times.Add(MeasureKernel(f_fc1, (uint)BatchSize, 1u, 1u, fc1FwdBlockSize, 1u, 1u, fc1Params));
                        }

                        fc2Times.Add(MeasureKernel(f_fc2, (uint)BatchSize, 1u, 1u, fc2BlockSize, 1u, 1u, fc2Params));

                        fc2BwdTimes.Add(MeasureKernel(f_fc2_bwd, (uint)BatchSize, 1u, 1u, fc2BlockSize, 1u, 1u, fc2BwdParams));

                        if (activeConfig.IsV7Based)
                        {
                            uint fc2BwdWGridX = activeConfig.Name == "V8" ? 784u : (activeConfig.Name == "V9" ? 120u : 400u);
                            fc2BwdWTimes.Add(MeasureKernel(f_fc2_bwd_weights, fc2BwdWGridX, 1u, 1u, 128u, 1u, 1u, fc2BwdWeightsParams));
                        }

                        if (activeConfig.HasFC1)
                        {
                            if (activeConfig.Name == "V8")
                            {
                                fc1BwdTimes.Add(MeasureKernel(f_fc1_bwd, (uint)BatchSize, 1u, 1u, 784u, 1u, 1u, fc1BwdParams));
                                fc1BwdWTimes.Add(MeasureKernel(f_fc1_bwd_weights, (uint)conv2FilterCount * (uint)conv2Chunks, 1u, 1u, 128u, 1u, 1u, fc1BwdWeightsParams));
                            }
                            else if (activeConfig.Name == "V9")
                            {
                                fc1BwdTimes.Add(MeasureKernel(f_fc1_bwd, (uint)BatchSize, 1u, 1u, 256u, 1u, 1u, fc1BwdParams));
                                fc1BwdWTimes.Add(MeasureKernel(f_fc1_bwd_weights, 256u, 1u, 1u, 64u, 1u, 1u, fc1BwdWeightsParams));
                            }
                            else if (activeConfig.Name == "V5")
                            {
                                fc1BwdTimes.Add(MeasureKernel(f_fc1_bwd, 1u, 1u, 1u, 128u, 1u, 1u, fc1BwdParams));
                                fc1BwdWTimes.Add(MeasureKernel(f_fc1_bwd_weights, 784u, 1u, 1u, 128u, 1u, 1u, fc1BwdWeightsParams));
                            }
                            else
                            {
                                fc1BwdTimes.Add(MeasureKernel(f_fc1_bwd, (uint)BatchSize, 1u, 1u, 256u, 1u, 1u, fc1BwdParams));
                                fc1BwdWTimes.Add(MeasureKernel(f_fc1_bwd_weights, 8u, 8u, (uint)fc1Chunks, 128u, 1u, 1u, fc1BwdWeightsParams));
                            }
                        }

                        if (activeConfig.Name != "V5")
                        {
                            conv2BwdTimes.Add(MeasureKernel(f_conv2_bwd, (uint)conv2FilterCount * (uint)conv2Chunks, 1u, 1u, 128u, 1u, 1u, conv2BwdParams));
                            conv1BwdTimes.Add(MeasureKernel(f_conv1_bwd, (uint)conv1FilterCount * (uint)conv1Chunks, 1u, 1u, conv1BwdBlockX, conv1BwdBlockY, 1u, conv1BwdParams));
                        }

                        adamTimes.Add(MeasureKernel(f_adam, (uint)((totalParamElements + 255) / 256), 1u, 1u, 256u, 1u, 1u, adamParams));
                    }

                    stopwatch.Stop();
                    trainingTime = stopwatch.Elapsed.TotalMilliseconds;

                    cuEventDestroy(startEvent).Ok();
                    cuEventDestroy(stopEvent).Ok();

                    void PrintStats(string name, System.Collections.Generic.List<float> times)
                    {
                        if (times.Count == 0) return;
                        float min = float.MaxValue, max = float.MinValue, sum = 0;
                        for (int i = 0; i < times.Count; i++)
                        {
                            float t = times[i];
                            if (t < min) min = t;
                            if (t > max) max = t;
                            sum += t;
                        }
                        float mean = sum / times.Count;
                        Console.WriteLine($"[PROFILE] {name,-20} | Min = {min,8:F3} ms | Mean = {mean,8:F3} ms | Max = {max,8:F3} ms | Total = {sum,8:F2} ms");
                    }

                    if (!isWarmup)
                    {
                        Console.WriteLine("==================================================");
                        Console.WriteLine("### GPU KERNEL PROFILING REPORT (Measured Run) ###");
                        Console.WriteLine("==================================================");
                        PrintStats("clear_gradient", clearTimes);
                        PrintStats("conv1_forward", conv1Times);
                        PrintStats("conv2_forward", conv2Times);
                        PrintStats("fc1_forward", fc1Times);
                        PrintStats("fc2_forward", fc2Times);
                        PrintStats("fc2_backward", fc2BwdTimes);
                        PrintStats("fc2_bwd_weights", fc2BwdWTimes);
                        PrintStats("fc1_backward", fc1BwdTimes);
                        PrintStats("fc1_bwd_weights", fc1BwdWTimes);
                        PrintStats("conv2_backward", conv2BwdTimes);
                        PrintStats("conv1_backward", conv1BwdTimes);
                        PrintStats("adam_update", adamTimes);
                        Console.WriteLine("==================================================");
                    }
                }
                else
                {
                    cuGraphLaunch(epochGraphExec, stream).Ok();
                    cuStreamSynchronize(stream).Ok();

                    var h_params = new Half[10];
                    var h_grads = new Half[10];
                    cuMemcpyDtoH((IntPtr)Unsafe.AsPointer(ref h_params[0]), d_allParams, (nuint)(10 * sizeof(Half))).Ok();
                    cuMemcpyDtoH((IntPtr)Unsafe.AsPointer(ref h_grads[0]), d_allParamGrads, (nuint)(10 * sizeof(Half))).Ok();
                    string paramsStr = "", gradsStr = "";
                    for (int di = 0; di < 10; di++)
                    {
                        paramsStr += ((float)h_params[di]).ToString("F5") + (di < 9 ? ", " : "");
                        gradsStr += ((float)h_grads[di]).ToString("F5") + (di < 9 ? ", " : "");
                    }
                    Console.WriteLine($"[DEBUG] Params: {paramsStr}");
                    Console.WriteLine($"[DEBUG] Grads:  {gradsStr}");

                    stopwatch.Stop();
                    trainingTime = stopwatch.Elapsed.TotalMilliseconds;
                }

                int correctPredictions = 0;

                for (int valStep = 0; valStep < testStepCount; valStep++)
                {
                    int batchOffset = valStep * BatchSize;
                    cuMemcpyHtoD(d_step, (IntPtr)(&valStep), (nuint)sizeof(int)).Ok();

                    if (activeConfig.Name == "V5")
                    {
                        cuLaunchKernel(f_fc1, (uint)BatchSize, 1u, 1u,
                            128u, 1u, 1u, 0u, stream, argsFc1V5, null).Ok();
                        cuLaunchKernel(f_fc2, (uint)BatchSize, 1u, 1u,
                            128u, 1u, 1u, 0u, stream, argsFc2V1, null).Ok();
                    }
                    else
                    {
                        cuLaunchKernel(f_conv1, (uint)BatchSize, (uint)conv1FilterCount, 1u,
                            conv1BlockX, conv1BlockY, 1u, 0u, stream, argsTestConv1, null).Ok();

                        if (activeConfig.Name == "V8")
                        {
                            var argsConv2_v8 = new void*[]
                            {
                                &d_conv1Out, &d_conv2Filters, &d_conv2Biases,
                                &d_conv2Out, &d_conv2Unpooled, &d_fc1Weights, &d_fc1Biases
                            };
                            var argsFc1V1_v8 = new void*[] { &d_conv2Out, &d_fc1Out };
                            var argsFc2V1_v8 = new void*[] { &d_fc1Out, &d_fc2Weights, &d_fc2Biases, &d_fc2Out };

                            fixed (void** pArgs2 = argsConv2_v8)
                            fixed (void** pArgs1 = argsFc1V1_v8)
                            fixed (void** pArgsLogits = argsFc2V1_v8)
                            {
                                cuLaunchKernel(f_conv2, (uint)BatchSize, 1u, 1u,
                                    256u, 1u, 1u, 0u, stream, pArgs2, null).Ok();
                                cuLaunchKernel(f_fc1, (uint)BatchSize, 1u, 1u,
                                    784u, 1u, 1u, 0u, stream, pArgs1, null).Ok();
                                cuLaunchKernel(f_fc2, (uint)BatchSize, 1u, 1u,
                                    256u, 1u, 1u, 0u, stream, pArgsLogits, null).Ok();
                            }
                        }
                        else
                        {
                            cuLaunchKernel(f_conv2, (uint)BatchSize, 1u, 1u,
                                256u, 1u, 1u, 0u, stream, argsConv2, null).Ok();

                            if (activeConfig.HasFC1)
                            {
                                cuLaunchKernel(f_fc1, (uint)BatchSize, 1u, 1u,
                                    256u, 1u, 1u, 0u, stream, argsFc1V1, null).Ok();
                                cuLaunchKernel(f_fc2, (uint)BatchSize, 1u, 1u,
                                    256u, 1u, 1u, 0u, stream, argsFc2V1, null).Ok();
                            }
                            else
                            {
                                cuLaunchKernel(f_fc2, (uint)BatchSize, 1u, 1u,
                                    256u, 1u, 1u, 0u, stream, argsFc2V2, null).Ok();
                            }
                        }
                    }

                    if (activeConfig.IsHalf)
                    {
                        cuMemcpyDtoH((IntPtr)Unsafe.AsPointer(ref h_fcOutHalf[0]),
                            d_fc2Out, (nuint)(h_fcOutHalf.Length * sizeof(Half))).Ok();
                        for (int i = 0; i < h_fcOut.Length; i++)
                        {
                            h_fcOut[i] = (float)h_fcOutHalf[i];
                        }
                    }
                    else
                    {
                        cuMemcpyDtoH((IntPtr)Unsafe.AsPointer(ref h_fcOut[0]),
                            d_fc2Out, (nuint)(h_fcOut.Length * sizeof(float))).Ok();
                    }

                    for (int b = 0; b < BatchSize; b++)
                    {
                        float maxVal = -1e9f;
                        int predLabel = -1;
                        for (int c = 0; c < ClassCount; c++)
                        {
                            float val = h_fcOut[b * ClassCount + c];
                            if (val > maxVal)
                            {
                                maxVal = val;
                                predLabel = c;
                            }
                        }
                        if (predLabel == h_testLabels[batchOffset + b])
                            correctPredictions++;
                    }
                }

                double accuracy = (double)correctPredictions / (testStepCount * BatchSize) * 100.0;
                Console.WriteLine("==================================================");
                if (isWarmup)
                {
                    Console.WriteLine($"[WARMUP RESULTS] Accuracy: {accuracy:F2}%, GPU Time: {trainingTime:F3} ms");
                }
                else
                {
                    Console.WriteLine($"[MEASURED RESULTS] Run #{sIndex} - Accuracy: {accuracy:F2}%, GPU Time: {trainingTime:F3} ms");
                    measuredTimes.Add(trainingTime);
                    measuredAccuracies.Add(accuracy);
                }
                Console.WriteLine("==================================================");
            }

            double minTime = double.MaxValue, maxTime = double.MinValue, sumTime = 0;
            double minAcc = double.MaxValue, maxAcc = double.MinValue, sumAcc = 0;

            for (int i = 0; i < measuredTimes.Count; i++)
            {
                double t = measuredTimes[i];
                double a = measuredAccuracies[i];
                if (t < minTime) minTime = t;
                if (t > maxTime) maxTime = t;
                sumTime += t;

                if (a < minAcc) minAcc = a;
                if (a > maxAcc) maxAcc = a;
                sumAcc += a;
            }

            double meanTime = sumTime / measuredTimes.Count;
            double meanAcc = sumAcc / measuredAccuracies.Count;

            Console.WriteLine("==================================================");
            Console.WriteLine("### SUMMARY METRICS FOR MEASURED RUNS ###");
            Console.WriteLine($"GPU Training Time: Min = {minTime:F3} ms | Mean = {meanTime:F3} ms | Max = {maxTime:F3} ms");
            Console.WriteLine($"Test Accuracy:     Min = {minAcc:F2}% | Mean = {meanAcc:F2}% | Max = {maxAcc:F2}%");
            Console.WriteLine("==================================================");

            cuMemFree(d_trainImages).Ok();
            cuMemFree(d_trainLabels).Ok();
            cuMemFree(d_testImages).Ok();
            cuMemFree(d_testLabels).Ok();

            cuMemFree(d_allParams).Ok();
            cuMemFree(d_allParamGrads).Ok();
            cuMemFree(d_allParamM).Ok();
            cuMemFree(d_allParamV).Ok();

            cuMemFree(d_conv1Out).Ok();
            cuMemFree(d_conv1Unpooled).Ok();
            cuMemFree(d_conv2Out).Ok();
            cuMemFree(d_conv2Unpooled).Ok();
            if (activeConfig.HasFC1)
            {
                cuMemFree(d_fc1Out).Ok();
            }
            cuMemFree(d_fc2Out).Ok();

            if (activeConfig.HasFC1)
            {
                cuMemFree(d_fc1OutGrad).Ok();
                cuMemFree(d_conv2OutGrad).Ok();
                if (d_intermediateGrad.Value != IntPtr.Zero)
                {
                    cuMemFree(d_intermediateGrad).Ok();
                }
            }
            else
            {
                cuMemFree(d_fc2InGrad).Ok();
            }
            cuMemFree(d_conv1OutGrad).Ok();

            cuMemFree(d_step).Ok();

            cuGraphExecDestroy(epochGraphExec).Ok();
            cuGraphDestroy(epochGraph).Ok();
            cuModuleUnload(module).Ok();
        }
        finally
        {
            nvrtcDestroyProgram(ref program).Ok();
            if (context.Value != IntPtr.Zero)
            {
                cuCtxDestroy(context).Ok();
            }
        }
    }

    static CUgraphNode AddKernelNode(
        CUgraph graph,
        ReadOnlySpan<CUgraphNode> dependencies,
        CUfunction function,
        uint gridX, uint gridY, uint gridZ,
        uint blockX, uint blockY, uint blockZ,
        void*[] args)
    {
        fixed (void** pArgs = args)
        {
            var nodeParams = new CUDA_KERNEL_NODE_PARAMS
            {
                func = function,
                gridDimX = gridX,
                gridDimY = gridY,
                gridDimZ = gridZ,
                blockDimX = blockX,
                blockDimY = blockY,
                blockDimZ = blockZ,
                sharedMemBytes = 0,
                kernelParams = (IntPtr)pArgs,
                extra = IntPtr.Zero
            };

            cuGraphAddKernelNode(out var node, graph, dependencies, (nuint)dependencies.Length, nodeParams).Ok();
            return node;
        }
    }

    static CUdeviceptr SliceDevicePtr(CUdeviceptr ptr, int offsetElements, int elementSize)
    {
        return new CUdeviceptr((IntPtr)(ptr.Value.ToInt64() + offsetElements * elementSize));
    }

    static void InitializeParameters(CUdeviceptr d_weights, CUdeviceptr d_biases, int outFeatures, int inFeatures, int seed, bool isHalf)
    {
        var rand = new Random(seed);
        double stdDev = Math.Sqrt(2.0 / inFeatures);

        if (isHalf)
        {
            Half[] h_weights = new Half[outFeatures * inFeatures];
            for (int i = 0; i < h_weights.Length; i++)
            {
                double u1 = 1.0 - rand.NextDouble();
                double u2 = 1.0 - rand.NextDouble();
                double normalRand = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                h_weights[i] = (Half)(normalRand * stdDev);
            }

            Half[] h_biases = new Half[outFeatures];
            for (int i = 0; i < h_biases.Length; i++)
            {
                h_biases[i] = (Half)0.0f;
            }

            cuMemcpyHtoD(d_weights, (IntPtr)Unsafe.AsPointer(ref h_weights[0]), (nuint)(h_weights.Length * sizeof(Half))).Ok();
            cuMemcpyHtoD(d_biases, (IntPtr)Unsafe.AsPointer(ref h_biases[0]), (nuint)(h_biases.Length * sizeof(Half))).Ok();
        }
        else
        {
            float[] h_weights = new float[outFeatures * inFeatures];
            for (int i = 0; i < h_weights.Length; i++)
            {
                double u1 = 1.0 - rand.NextDouble();
                double u2 = 1.0 - rand.NextDouble();
                double normalRand = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                h_weights[i] = (float)(normalRand * stdDev);
            }

            float[] h_biases = new float[outFeatures];
            for (int i = 0; i < h_biases.Length; i++)
            {
                h_biases[i] = 0.0f;
            }

            cuMemcpyHtoD(d_weights, (IntPtr)Unsafe.AsPointer(ref h_weights[0]), (nuint)(h_weights.Length * sizeof(float))).Ok();
            cuMemcpyHtoD(d_biases, (IntPtr)Unsafe.AsPointer(ref h_biases[0]), (nuint)(h_biases.Length * sizeof(float))).Ok();
        }
    }

    static void InitializeModelParameters(
        NetworkConfig activeConfig,
        CUdeviceptr d_conv1Filters, CUdeviceptr d_conv1Biases,
        CUdeviceptr d_conv2Filters, CUdeviceptr d_conv2Biases,
        CUdeviceptr d_fc1Weights, CUdeviceptr d_fc1Biases,
        CUdeviceptr d_fc2Weights, CUdeviceptr d_fc2Biases,
        int seed)
    {
        bool isHalf = activeConfig.IsHalf;
        var conv1 = activeConfig.GetParam("conv1");
        InitializeParameters(d_conv1Filters, d_conv1Biases, conv1.OutFeatures, conv1.InFeatures, seed, isHalf);

        var conv2 = activeConfig.GetParam("conv2");
        InitializeParameters(d_conv2Filters, d_conv2Biases, conv2.OutFeatures, conv2.InFeatures, seed, isHalf);

        if (activeConfig.HasFC1)
        {
            var fc1 = activeConfig.GetParam("fc1");
            InitializeParameters(d_fc1Weights, d_fc1Biases, fc1.OutFeatures, fc1.InFeatures, seed, isHalf);
            if (activeConfig.Name == "V8" || activeConfig.Name == "V9")
            {
                ScaleDownDeviceBuffer(d_fc1Weights, fc1.OutFeatures * fc1.InFeatures, 0.05f, isHalf);
            }
        }

        var fc2 = activeConfig.GetParam("fc2");
        InitializeParameters(d_fc2Weights, d_fc2Biases, fc2.OutFeatures, fc2.InFeatures, seed, isHalf);
    }

    static unsafe void ScaleDownDeviceBuffer(CUdeviceptr ptr, int size, float scale, bool isHalf)
    {
        if (isHalf)
        {
            Half[] host = new Half[size];
            cuMemcpyDtoH((IntPtr)Unsafe.AsPointer(ref host[0]), ptr, (nuint)(size * sizeof(Half))).Ok();
            for (int i = 0; i < size; i++)
            {
                host[i] = (Half)((float)host[i] * scale);
            }
            cuMemcpyHtoD(ptr, (IntPtr)Unsafe.AsPointer(ref host[0]), (nuint)(size * sizeof(Half))).Ok();
        }
        else
        {
            float[] host = new float[size];
            cuMemcpyDtoH((IntPtr)Unsafe.AsPointer(ref host[0]), ptr, (nuint)(size * sizeof(float))).Ok();
            for (int i = 0; i < size; i++)
            {
                host[i] = host[i] * scale;
            }
            cuMemcpyHtoD(ptr, (IntPtr)Unsafe.AsPointer(ref host[0]), (nuint)(size * sizeof(float))).Ok();
        }
    }

    static void EnsureDatasetFile(string filePath, string url)
    {
        if (File.Exists(filePath)) return;

        Console.WriteLine($"[DOWNLOAD] MNIST dataset file missing. Fetching from: {url}");
        using var client = new HttpClient();
        var response = client.GetAsync(url).GetAwaiter().GetResult();
        response.EnsureSuccessStatusCode();

        using var fs = File.Create(filePath);
        response.Content.CopyToAsync(fs).GetAwaiter().GetResult();
        Console.WriteLine($"[DOWNLOAD] Download complete. Saved to: {filePath}");
    }

    static (uint[] images, int count) ParseImagesGz(string filePath, int maxCount)
    {
        using var fileStream = File.OpenRead(filePath);
        using var gzStream = new GZipStream(fileStream, CompressionMode.Decompress);
        using var ms = new MemoryStream();
        gzStream.CopyTo(ms);
        var bytes = ms.ToArray();

        int magic = (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
        if (magic != 0x00000803)
            throw new InvalidOperationException($"Invalid images magic number: {magic:X}");

        int count = (bytes[4] << 24) | (bytes[5] << 16) | (bytes[6] << 8) | bytes[7];
        int rows = (bytes[8] << 24) | (bytes[9] << 16) | (bytes[10] << 8) | bytes[11];
        int cols = (bytes[12] << 24) | (bytes[13] << 16) | (bytes[14] << 8) | bytes[15];

        if (rows != 28 || cols != 28)
            throw new InvalidOperationException($"Expected 28x28 images, but got {rows}x{cols}");

        int imageCountToLoad = maxCount;
        uint[] packedImages = new uint[imageCountToLoad * 28];

        for (int i = 0; i < imageCountToLoad; i++)
        {
            int sourceImageIdx = i % count;
            int sourcePixelOffset = 16 + sourceImageIdx * 28 * 28;

            for (int r = 0; r < 28; r++)
            {
                uint rowBits = 0;
                for (int c = 0; c < 28; c++)
                {
                    byte pixelVal = bytes[sourcePixelOffset++];
                    if (pixelVal > 127)
                    {
                        rowBits |= (1u << c);
                    }
                }
                packedImages[i * 28 + r] = rowBits;
            }
        }

        return (packedImages, imageCountToLoad);
    }

    static int[] ParseLabelsGz(string filePath, int maxCount)
    {
        using var fileStream = File.OpenRead(filePath);
        using var gzStream = new GZipStream(fileStream, CompressionMode.Decompress);
        using var ms = new MemoryStream();
        gzStream.CopyTo(ms);
        var bytes = ms.ToArray();

        int magic = (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
        if (magic != 0x00000801)
            throw new InvalidOperationException($"Invalid labels magic number: {magic:X}");

        int count = (bytes[4] << 24) | (bytes[5] << 16) | (bytes[6] << 8) | bytes[7];

        int labelCountToLoad = maxCount;
        int[] labels = new int[labelCountToLoad];

        for (int i = 0; i < labelCountToLoad; i++)
        {
            labels[i] = bytes[8 + (i % count)];
        }

        return labels;
    }
}
