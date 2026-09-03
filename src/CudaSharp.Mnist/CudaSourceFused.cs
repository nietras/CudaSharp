using System;

namespace CudaSharp.Mnist;

public static class CudaSourceFused
{
    public static readonly string FusedForward =
        """
        extern "C" __global__ void fused_forward(
            const uint32_t* __restrict__ d_images,
            const __half* __restrict__ d_conv1_weights,
            const __half* __restrict__ d_conv1_biases,
            __half* __restrict__ d_conv1_out,
            __half* __restrict__ d_conv1_unpooled,
            const __half* __restrict__ d_conv2_weights,
            const __half* __restrict__ d_conv2_biases,
            __half* __restrict__ d_conv2_out,
            __half* __restrict__ d_conv2_unpooled,
            const __half* __restrict__ d_fc1_weights,
            const __half* __restrict__ d_fc1_biases,
            __half* __restrict__ d_fc1_out,
            __half* __restrict__ d_fc1_unpooled,
            const __half* __restrict__ d_fc2_weights,
            const __half* __restrict__ d_fc2_biases,
            __half* __restrict__ d_fc2_out,
            const int* __restrict__ d_step,
            int is_training)
        {
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x;

            // Shared memory pool, reused across layers. Max needed is 3456 halves (6.9 KB).
            __shared__ __half s_pool[3456];
            __shared__ uint32_t s_image[28];

            // 1. Load Image
            if (tid < 28)
            {
                s_image[tid] = d_images[batch_idx * 28 + tid];
            }
            __syncthreads();

            // 2. Conv1: 28x28 -> 24x24x6 = 3456 elements
            for (int i = tid; i < 3456; i += 256)
            {
                int c = i % 6;
                int x = (i / 6) % 24;
                int y = i / 144;

                __half sum = d_conv1_biases[c];
                #pragma unroll
                for (int fy = 0; fy < 5; fy++)
                {
                    uint32_t row = s_image[y + fy];
                    #pragma unroll
                    for (int fx = 0; fx < 5; fx++)
                    {
                        uint32_t pixel = (row >> (x + fx)) & 1u;
                        sum += __half((float)pixel) * d_conv1_weights[c * 25 + fy * 5 + fx];
                    }
                }
                d_conv1_unpooled[batch_idx * 3456 + i] = sum;
                s_pool[i] = ACTIVATION_FWD(sum);
            }
            __syncthreads();

            // 3. Pool1: 24x24x6 -> 12x12x6 = 864 elements
            // To reuse s_pool, we must read all values before overwriting. 
            // We can write to the start of s_pool since 864 < 3456, but we must be careful not to overwrite values another thread needs!
            // Wait, we can't do in-place without risk unless we write to a separate array or wait until everyone read.
            // But we have plenty of shared memory! Let's just use s_pool_2 for the 864 elements.
            __shared__ __half s_pool_2[1024];

            for (int i = tid; i < 864; i += 256)
            {
                int c = i % 6;
                int x = (i / 6) % 12;
                int y = i / 72;

                int base_idx = (y * 2) * 144 + (x * 2) * 6 + c;
                __half v0 = s_pool[base_idx];
                __half v1 = s_pool[base_idx + 6];
                __half v2 = s_pool[base_idx + 144];
                __half v3 = s_pool[base_idx + 150];

                __half max1 = __hgt(v0, v1) ? v0 : v1;
                __half max2 = __hgt(v2, v3) ? v2 : v3;
                __half max_val = __hgt(max1, max2) ? max1 : max2;

                d_conv1_out[batch_idx * 864 + i] = max_val;
                s_pool_2[i] = max_val;
            }
            __syncthreads();

            // 4. Conv2: 12x12x6 -> 8x8x16 = 1024 elements
            for (int i = tid; i < 1024; i += 256)
            {
                int c = i % 16;
                int x = (i / 16) % 8;
                int y = i / 128;

                __half sum = d_conv2_biases[c];
                #pragma unroll
                for (int ic = 0; ic < 6; ic++)
                {
                    #pragma unroll
                    for (int fy = 0; fy < 5; fy++)
                    {
                        #pragma unroll
                        for (int fx = 0; fx < 5; fx++)
                        {
                            __half val = s_pool_2[(y + fy) * 72 + (x + fx) * 6 + ic];
                            __half w = d_conv2_weights[c * 150 + ic * 25 + fy * 5 + fx];
                            sum += val * w;
                        }
                    }
                }
                d_conv2_unpooled[batch_idx * 1024 + i] = sum;
                s_pool[i] = ACTIVATION_FWD(sum); // Overwrite s_pool safely
            }
            __syncthreads();

            // 5. Pool2: 8x8x16 -> 4x4x16 = 256 elements
            if (tid < 256)
            {
                int c = tid % 16;
                int x = (tid / 16) % 4;
                int y = tid / 64;

                int base_idx = (y * 2) * 128 + (x * 2) * 16 + c;
                __half v0 = s_pool[base_idx];
                __half v1 = s_pool[base_idx + 16];
                __half v2 = s_pool[base_idx + 128];
                __half v3 = s_pool[base_idx + 144];

                __half max1 = __hgt(v0, v1) ? v0 : v1;
                __half max2 = __hgt(v2, v3) ? v2 : v3;
                __half max_val = __hgt(max1, max2) ? max1 : max2;

                d_conv2_out[batch_idx * 256 + tid] = max_val;
                s_pool_2[tid] = max_val; // Overwrite s_pool_2 safely
            }
            __syncthreads();

            // 6. FC1: 256 -> 120
            if (tid < 120)
            {
                __half sum = d_fc1_biases[tid];
                #if USE_HALF2 == 1
                if (tid < 60) // 60 active threads computing 2 outputs each
                {
                    __half2 sum2 = __float2half2_rn(0.0f);
                    __half2 bias2 = ((const __half2*)d_fc1_biases)[tid];
                    sum2 = sum2 + bias2; // Wait, half2 addition
                    // Note: This needs exact syntax. Let's just use half for now to avoid compilation issues, or write it carefully:
                    // sum2 = __hadd2(sum2, bias2);
                }
                #endif
                
                // Let's stick to safe FP16 first to ensure logic works:
                #pragma unroll
                for (int i = 0; i < 256; i++)
                {
                    sum += s_pool_2[i] * d_fc1_weights[i * 120 + tid];
                }
                
                d_fc1_unpooled[batch_idx * 120 + tid] = sum;
                __half act = ACTIVATION_FWD(sum);
                
                #if HAS_DROPOUT == 1
                if (is_training == 1)
                {
                    // LCG PRNG for dropout
                    uint32_t seed = (batch_idx * 120 + tid) * (*d_step + 1) * 19937;
                    seed ^= seed >> 11;
                    seed ^= seed << 7;
                    seed ^= seed >> 15;
                    float rand_val = (float)(seed & 0xFFFFFF) / 16777216.0f;
                    if (rand_val < DROPOUT_RATE) {
                        act = __float2half(-1e9f); // Marked as dropped
                    } else {
                        act = __float2half(__half2float(act) * (1.0f / (1.0f - DROPOUT_RATE)));
                    }
                }
                #endif
                
                d_fc1_out[batch_idx * 120 + tid] = act;
                s_pool[tid] = act; // Overwrite s_pool
            }
            __syncthreads();

            // 7. FC2: 120 -> 10
            if (tid < 10)
            {
                __half sum = d_fc2_biases[tid];
                #pragma unroll
                for (int i = 0; i < 120; i++)
                {
                    __half act = s_pool[i];
                    #if HAS_DROPOUT == 1
                    if (__half2float(act) <= -1e8f) {
                        act = __float2half(0.0f);
                    }
                    #endif
                    sum += act * d_fc2_weights[i * 10 + tid];
                }
                d_fc2_out[batch_idx * 10 + tid] = sum;
            }
        }
        """;
}
