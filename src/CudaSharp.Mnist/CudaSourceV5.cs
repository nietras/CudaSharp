using System;

namespace CudaSharp.Mnist;

public static partial class Program
{
    public static readonly string CudaSourceV5 =
        """
        typedef unsigned int uint32_t;
        
        #ifndef BATCH_SIZE
        #define BATCH_SIZE 128
        #endif
        #define FC1_INPUTS 784
        #ifndef FC1_OUTPUTS
        #define FC1_OUTPUTS 256
        #endif
        #define FC2_OUTPUTS 10

        #ifndef BATCHES_PER_EPOCH
        #define BATCHES_PER_EPOCH 350
        #endif
        #ifndef TOTAL_STEPS
        #define TOTAL_STEPS 350
        #endif

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

        // Dummy empty conv kernels to satisfy nvcc/nvrtc entrypoints in program.cs
        extern "C" __global__ void conv1_forward(
            const uint32_t* __restrict__ d_inputs,
            const float* __restrict__ d_filters,
            const float* __restrict__ d_biases,
            float* __restrict__ d_outputs,
            float* __restrict__ d_unpooled_vals,
            const int* __restrict__ d_step,
            int is_training) {}

        extern "C" __global__ void conv2_forward(
            const float* __restrict__ d_inputs,
            const float* __restrict__ d_filters,
            const float* __restrict__ d_biases,
            float* __restrict__ d_outputs,
            float* __restrict__ d_unpooled_vals) {}

        // FC1 Forward Pass (1-bit packed input 784 -> 256, ReLU)
        extern "C" __global__ void fc1_forward(
            const uint32_t* __restrict__ d_inputs,   // [TotalImages x 28] packed uints
            const float* __restrict__ d_weights,     // [784 x FC1_OUTPUTS]
            const float* __restrict__ d_biases,      // [FC1_OUTPUTS]
            float* __restrict__ d_outputs,           // [BatchSize x FC1_OUTPUTS]
            const int* __restrict__ d_step,
            int is_training)
        {
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x; // 0..FC1_OUTPUTS-1

            if (tid >= FC1_OUTPUTS) return;

            int batchOffset = 0;
            if (is_training == 1)
            {
                batchOffset = ((*d_step) % BATCHES_PER_EPOCH) * BATCH_SIZE;
            }
            else
            {
                batchOffset = (*d_step) * BATCH_SIZE;
            }

            // Load packed 1-bit input image rows to shared memory
            __shared__ uint32_t s_image[28];
            if (tid < 28)
            {
                s_image[tid] = d_inputs[(batchOffset + batch_idx) * 28 + tid];
            }
            __syncthreads();

            float sum = d_biases[tid];

            #pragma unroll
            for (int r = 0; r < 28; r++)
            {
                uint32_t row_bits = s_image[r];

                #pragma unroll
                for (int c = 0; c < 28; c++)
                {
                    uint32_t pixel = (row_bits >> c) & 1u;

                    sum += (float)pixel * d_weights[(r * 28 + c) * FC1_OUTPUTS + tid];
                }
            }

            d_outputs[batch_idx * FC1_OUTPUTS + tid] = sum > 0.0f ? sum : 0.0f;
        }

        // FC2 Forward Pass (FC1_OUTPUTS -> 10, Linear)
        extern "C" __global__ void fc2_forward(
            const float* __restrict__ d_inputs,      // [BatchSize x FC1_OUTPUTS]
            const float* __restrict__ d_weights,     // [FC1_OUTPUTS x 10]
            const float* __restrict__ d_biases,      // [10]
            float* __restrict__ d_outputs)           // [BatchSize x 10]
        {
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x; // 0..255 (block size FC1_OUTPUTS)

            __shared__ float s_input[FC1_OUTPUTS];
            if (tid < FC1_OUTPUTS)
            {
                s_input[tid] = d_inputs[batch_idx * FC1_OUTPUTS + tid];
            }
            __syncthreads();

            if (tid < 10)
            {
                float sum = d_biases[tid];
                #pragma unroll 4
                for (int i = 0; i < FC1_OUTPUTS; i++)
                {
                    sum += s_input[i] * d_weights[i * 10 + tid];
                }
                d_outputs[batch_idx * 10 + tid] = sum;
            }
        }

        // FC2 Backward Pass (FC1_OUTPUTS -> 10, Linear)
        extern "C" __global__ void fc2_backward(
            const float* __restrict__ d_fc2_outputs,  // [BatchSize x 10]
            const int* __restrict__ d_labels,        // [TotalImages]
            const float* __restrict__ d_fc1_outputs,   // [BatchSize x FC1_OUTPUTS]
            const float* __restrict__ d_fc2_weights,   // [FC1_OUTPUTS x 10]
            float* __restrict__ d_fc2_weights_grad,    // [FC1_OUTPUTS x 10]
            float* __restrict__ d_fc2_biases_grad,     // [10]
            float* __restrict__ d_fc1_out_grad,       // [BatchSize x FC1_OUTPUTS]
            const int* __restrict__ d_step)
        {
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x; // 0..FC1_OUTPUTS-1

            if (tid >= FC1_OUTPUTS) return;

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

            float x_val = d_fc1_outputs[batch_idx * FC1_OUTPUTS + tid];

            float sum_input_grad = 0.0f;
            #pragma unroll
            for (int c = 0; c < 10; c++)
            {
                sum_input_grad += s_grad[c] * d_fc2_weights[tid * 10 + c];
            }
            d_fc1_out_grad[batch_idx * FC1_OUTPUTS + tid] = sum_input_grad;

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

        // FC1 Backward Pass (No input gradient required, block-level bias reduction!)
        extern "C" __global__ void fc1_backward(
            const float* __restrict__ d_fc1_out_grad, // [BatchSize x FC1_OUTPUTS]
            const float* __restrict__ d_fc1_outputs,  // [BatchSize x FC1_OUTPUTS]
            const float* __restrict__ d_dummy_inputs,
            const float* __restrict__ d_dummy_weights,
            float* __restrict__ d_fc1_biases_grad,
            float* __restrict__ d_dummy_out_grad)
        {
            const int tid = threadIdx.x; // 0..FC1_OUTPUTS-1
            if (tid >= FC1_OUTPUTS) return;

            float sum = 0.0f;
            #pragma unroll 4
            for (int b = 0; b < BATCH_SIZE; b++)
            {
                float out_val = d_fc1_outputs[b * FC1_OUTPUTS + tid];
                sum += out_val > 0.0f ? d_fc1_out_grad[b * FC1_OUTPUTS + tid] : 0.0f;
            }

            d_fc1_biases_grad[tid] = sum;
        }

        // FC1 Weights Gradient Kernel (Highly optimized, zero atomic weight updates!)
        extern "C" __global__ void fc1_backward_weights(
            const float* __restrict__ d_fc1_out_grad, // [BatchSize x FC1_OUTPUTS]
            const float* __restrict__ d_fc1_outputs,  // unused
            const uint32_t* __restrict__ d_inputs,    // [TotalImages x 28] packed uints
            float* __restrict__ d_fc1_weights_grad,   // [784 x FC1_OUTPUTS]
            const int* __restrict__ d_step)
        {
            const int input_idx = blockIdx.x; // 0..783
            const int tid = threadIdx.x;      // 0..FC1_OUTPUTS-1

            if (tid >= FC1_OUTPUTS) return;

            int batchOffset = ((*d_step) % BATCHES_PER_EPOCH) * BATCH_SIZE;

            __shared__ unsigned char s_pixels[128];

            if (tid < 128)
            {
                const int b = tid;
                int img_row = input_idx / 28;
                int img_col = input_idx % 28;
                uint32_t row_bits = d_inputs[(batchOffset + b) * 28 + img_row];
                s_pixels[b] = (row_bits >> img_col) & 1u;
            }
            __syncthreads();

            float sum = 0.0f;
            #pragma unroll 4
            for (int j = 0; j < 128; j++)
            {
                sum += (float)s_pixels[j] * d_fc1_out_grad[j * FC1_OUTPUTS + tid];
            }

            d_fc1_weights_grad[input_idx * FC1_OUTPUTS + tid] = sum;
        }

        // Dummy empty conv backward kernels
        extern "C" __global__ void conv2_backward(
            const float* __restrict__ d_conv2_out_grad,
            const float* __restrict__ d_conv2_out_val,
            const float* __restrict__ d_conv2_unpooled_vals,
            const float* __restrict__ d_conv1_out,
            const float* __restrict__ d_conv2_filters,
            float* __restrict__ d_conv2_filters_grad,
            float* __restrict__ d_conv2_biases_grad,
            float* __restrict__ d_conv1_out_grad) {}

        extern "C" __global__ void conv1_backward(
            const float* __restrict__ d_conv1_out_grad,
            const float* __restrict__ d_conv1_out,
            const float* __restrict__ d_conv1_unpooled,
            const uint32_t* __restrict__ d_inputs,
            float* __restrict__ d_conv1_filters_grad,
            float* __restrict__ d_conv1_biases_grad,
            const int* __restrict__ d_step,
            int is_training) {}

        // Adam parameter updates
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
