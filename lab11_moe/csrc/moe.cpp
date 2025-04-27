#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <limits>
#include <vector> 
#include <c10/hip/HIPStream.h>

// Configuration for Expert-centric approach (Large Inter Dim Strategy)
#define EXPERT_BLOCK_SIZE 256      // Thread-per-block cho expert kernels

// GEMM Tile dimensions handled by one block
// WorkGroup
#define EXPERT_WG_TILE_M 64
#define EXPERT_WG_TILE_N 64
#define EXPERT_WG_TILE_K 32

// Each THREAD computes a small block of the output tile
#define EXPERT_THREAD_TILE_M 4
#define EXPERT_THREAD_TILE_N 4


// --- Checking HIP errors ---
#define HIP_CHECK(cmd) do { \
    hipError_t e = cmd; \
    if(e != hipSuccess) { \
        printf("Failed: HIP error %s:%d '%s'\n", __FILE__, __LINE__, hipGetErrorString(e)); \
        throw std::runtime_error(hipGetErrorString(e)); \
    } \
} while(0)

__global__ void small_process_kernel(
    const float* __restrict__ hidden,
    const float* __restrict__ w1,
    const float* __restrict__ w2,
    const int32_t* __restrict__ topk_ids,
    float* __restrict__ expert_output,
    int token_num,
    int topk,
    int model_dim,
    int inter_dim
) {
    using float4_t = float4;
    constexpr int VEC_SIZE = 4;
    constexpr int WARP_SIZE = 64;
    constexpr int CHUNK_SIZE = 512; // Optimal chunk size for cache usage
    
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;
    
    int branch_idx = blockIdx.x;
    int t = branch_idx / topk;    
    int k = branch_idx % topk;    
    
    if (t >= token_num) return;
    
    int E_id = topk_ids[t * topk + k];

    // Shared memory layout optimization
    extern __shared__ float smem[];
    float* s_hidden = smem;
    float* s_act_out = &smem[((model_dim + 127) / 128) * 128]; 

    // Load hidden states with vectorized access
    for (int m = threadIdx.x * VEC_SIZE; m < model_dim; m += blockDim.x * VEC_SIZE) {
        if (m + VEC_SIZE <= model_dim) {
            float4_t vec = reinterpret_cast<const float4_t*>(&hidden[t * model_dim])[m/VEC_SIZE];
            reinterpret_cast<float4_t*>(s_hidden)[m/VEC_SIZE] = vec;
        } else {
            for (int j = 0; j < min(VEC_SIZE, model_dim - m); j++) {
                s_hidden[m + j] = hidden[t * model_dim + m + j];
            }
        }
    }
    __syncthreads();

    const size_t w1_offset = (size_t)E_id * (2 * inter_dim) * model_dim;
    const size_t w2_offset = (size_t)E_id * model_dim * inter_dim;
    const size_t out_offset = (size_t)t * topk * model_dim + k * model_dim;

    // Process in smaller chunks for better cache utilization
    for (int chunk_start = 0; chunk_start < inter_dim; chunk_start += CHUNK_SIZE) {
        const int chunk_end = min(chunk_start + CHUNK_SIZE, inter_dim);
        const int chunk_len = chunk_end - chunk_start;

        // Clear temporary buffer
        for (int j = threadIdx.x; j < chunk_len; j += blockDim.x) {
            s_act_out[j] = 0.0f;
        }
        __syncthreads();

        // Compute gate and up projections in parallel with warp-level parallelism
        for (int j = warp_id; j < chunk_len; j += warps_per_block) {
            float gate = 0.0f, up = 0.0f;
            
            #pragma unroll 4
            for (int m = lane_id; m < model_dim; m += WARP_SIZE) {
                float h_val = s_hidden[m];
                float w1_gate = w1[w1_offset + (chunk_start + j) * model_dim + m];
                float w1_up = w1[w1_offset + (inter_dim + chunk_start + j) * model_dim + m];
                
                gate += h_val * w1_gate;
                up += h_val * w1_up;
            }
            
            // Warp reduction
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                gate += __shfl_down(gate, offset);
                up += __shfl_down(up, offset);
            }
            
            if (lane_id == 0) {
                s_act_out[j] = gate / (1.0f + expf(-gate)) * up;
            }
        }
        __syncthreads();

        // Compute output projection
        for (int m = warp_id; m < model_dim; m += warps_per_block) {
            float sum = 0.0f;
            
            #pragma unroll 4 
            for (int j = lane_id; j < chunk_len; j += WARP_SIZE) {
                float act = s_act_out[j];
                float w2_val = w2[w2_offset + m * inter_dim + chunk_start + j];
                sum += act * w2_val;
            }

            // Warp reduction
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                sum += __shfl_down(sum, offset);
            }

            if (lane_id == 0) {
                expert_output[out_offset + m] = sum;
            }
        }
        __syncthreads();
    }
}

// Kernel 2: Fused kernel for weight application and summation
__global__ void weighted_sum_small_kernel(
    const float* __restrict__ expert_output, 
    const float* __restrict__ topk_weight,
    float* __restrict__ final_output,
    int token_num, 
    int topk, 
    int model_dim) 
{
    // Handle one output element per thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= token_num * model_dim) return;
    
    int t = idx / model_dim;
    int m = idx % model_dim;
    
    // Compute weighted sum across all experts for this element
    float sum = 0.0f;
    for (int k = 0; k < topk; k++) {
        float expert_val = expert_output[t * topk * model_dim + k * model_dim + m];
        float weight = topk_weight[t * topk + k];
        sum += expert_val * weight;
    }
    
    final_output[idx] = sum;
}

// Expert-centric approach kernels (Large Inter Dim Strategy)

// Helper function để tạo masks và indices cho từng expert
__global__ void create_expert_masks(
    const int32_t* __restrict__ topk_ids,    // Input: [token_num, topk]
    const float* __restrict__ topk_w,        // Input: [token_num, topk]
    int32_t* __restrict__ expert_indices,    // Output: [E, expert_info_stride_sz] (stores original token_idx)
    float* __restrict__ expert_weights,      // Output: [E, expert_info_stride_sz] (stores corresponding weight)
    int32_t* __restrict__ expert_counts,     // Output: [E] (stores count per expert)
    int token_num,
    int topk,
    int num_experts,
    size_t expert_info_stride_sz             
) {
    // Each block handles one expert
    int expert_id = blockIdx.x;
    if (expert_id >= num_experts) return;

    // Reset counter
    if (threadIdx.x == 0) {
        expert_counts[expert_id] = 0;
    }
    __syncthreads();

    // Each thread checks a subset of the flattened token/topk pairs
    size_t total_pairs = (size_t)token_num * topk;
    for (size_t flat_idx = threadIdx.x; flat_idx < total_pairs; flat_idx += blockDim.x) {
        int token_idx = flat_idx / topk;
        int k_idx = flat_idx % topk;

        int current_expert_id = topk_ids[token_idx * topk + k_idx];
        if (current_expert_id == expert_id) {
            int offset = atomicAdd(&expert_counts[expert_id], 1);
            // Calculate offset into the expert's segment of the output arrays
            size_t base_offset = (size_t)expert_id * expert_info_stride_sz;
            if (base_offset + offset < (size_t)expert_id * expert_info_stride_sz + expert_info_stride_sz) { // Bounds check
                expert_indices[base_offset + offset] = token_idx;
                expert_weights[base_offset + offset] = topk_w[token_idx * topk + k_idx];
            }
        }
    }
}

// Kernel để tập hợp các tokens thuộc một expert
__global__ void gather_expert_tokens(
    const float* __restrict__ hidden,        // Input: [token_num, model_dim]
    float* __restrict__ expert_tokens,       // Output: [token_count, model_dim]
    const int32_t* __restrict__ expert_indices,// Input: [E, expert_info_stride_sz]
    int expert_id,
    int token_count,
    int token_num,                                                           
    int model_dim,
    size_t expert_info_stride_sz             
) {
    if (token_count == 0) return;

    size_t total_elements = (size_t)token_count * model_dim;
    size_t base_expert_indices_offset = (size_t)expert_id * expert_info_stride_sz;

    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += (size_t)gridDim.x * blockDim.x)
    {
        int expert_token_offset = idx / model_dim; //(0..token_count-1)
        int feature_idx = idx % model_dim;

        if (expert_token_offset < token_count) { 
            // Get the original token index
            int original_token_idx = expert_indices[base_expert_indices_offset + expert_token_offset];

            // Read from original hidden states and write to expert buffer
            if ((size_t)original_token_idx * model_dim + feature_idx < (size_t)token_num * model_dim) {
                 expert_tokens[idx] = hidden[(size_t)original_token_idx * model_dim + feature_idx];
            } else {
                 expert_tokens[idx] = 0.0f; 
            }
        }
    }
}


//GEMM kernel for W1 (Gate and Up) using standard tiling
__global__ void expert_gemm_w1_kernel(
    const float* __restrict__ expert_tokens, // Input A: [token_count, model_dim]
    const float* __restrict__ w1,            // Input B (Weights): [E, 2*inter_dim, model_dim]
    float* __restrict__ gate_out,            // Output C1: [token_count, inter_dim]
    float* __restrict__ up_out,              // Output C2: [token_count, inter_dim]
    int expert_id,
    int token_count, // M 
    int model_dim,   // K 
    int inter_dim    // N 
) {
    if (token_count == 0) return;

    int wg_tile_m = blockIdx.x * EXPERT_WG_TILE_M;
    int wg_tile_n = blockIdx.y * EXPERT_WG_TILE_N;

    // Pointers to weight matrices for the current expert
    const float* w1_expert_base = w1 + (size_t)expert_id * (2 * inter_dim) * model_dim;
    const float* w1_gate_expert = w1_expert_base;
    const float* w1_up_expert   = w1_expert_base + (size_t)inter_dim * model_dim;

    // Thread identification within the block
    int thread_id = threadIdx.x;
    const int threads_per_row = EXPERT_WG_TILE_N / EXPERT_THREAD_TILE_N;
    int thread_tile_m_base = (thread_id / threads_per_row) * EXPERT_THREAD_TILE_M;
    int thread_tile_n_base = (thread_id % threads_per_row) * EXPERT_THREAD_TILE_N;

    // Shared memory allocation
    extern __shared__ float smem[];
    float* token_tile = smem;
    float* w1_gate_tile = token_tile + EXPERT_WG_TILE_M * EXPERT_WG_TILE_K;
    float* w1_up_tile = w1_gate_tile + EXPERT_WG_TILE_K * EXPERT_WG_TILE_N;

    // Accumulators in registers
    float gate_accum[EXPERT_THREAD_TILE_M][EXPERT_THREAD_TILE_N] = {{0.0f}};
    float up_accum[EXPERT_THREAD_TILE_M][EXPERT_THREAD_TILE_N] = {{0.0f}};

    // Loop over K dimension tiles (model_dim)
    for (int k_tile_base = 0; k_tile_base < model_dim; k_tile_base += EXPERT_WG_TILE_K) {
        int k_tile_size = min(EXPERT_WG_TILE_K, model_dim - k_tile_base);

        // Load A tile (expert_tokens) into Shared Memory
        int num_a_elements = EXPERT_WG_TILE_M * k_tile_size;
        for (int load_idx = thread_id; load_idx < num_a_elements; load_idx += blockDim.x) {
            int m_local = load_idx / k_tile_size;
            int k_local = load_idx % k_tile_size;
            int m_global = wg_tile_m + m_local;
            int k_global = k_tile_base + k_local;
            if (m_global < token_count && k_global < model_dim) {
                 token_tile[m_local * EXPERT_WG_TILE_K + k_local] = expert_tokens[(size_t)m_global * model_dim + k_global];
            } else {
                 token_tile[m_local * EXPERT_WG_TILE_K + k_local] = 0.0f;
            }
        }

        // Load B tile (w1_gate) into Shared Memory
        int num_b_gate_elements = k_tile_size * EXPERT_WG_TILE_N;
         for (int load_idx = thread_id; load_idx < num_b_gate_elements; load_idx += blockDim.x) {
            int k_local = load_idx / EXPERT_WG_TILE_N;
            int n_local = load_idx % EXPERT_WG_TILE_N;
            int k_global = k_tile_base + k_local;
            int n_global = wg_tile_n + n_local;
            if (n_global < inter_dim && k_global < model_dim) {
                w1_gate_tile[k_local * EXPERT_WG_TILE_N + n_local] = w1_gate_expert[(size_t)n_global * model_dim + k_global];
            } else {
                w1_gate_tile[k_local * EXPERT_WG_TILE_N + n_local] = 0.0f;
            }
        }

        // Load B tile (w1_up) into Shared Memory
        int num_b_up_elements = k_tile_size * EXPERT_WG_TILE_N;
         for (int load_idx = thread_id; load_idx < num_b_up_elements; load_idx += blockDim.x) {
            int k_local = load_idx / EXPERT_WG_TILE_N;
            int n_local = load_idx % EXPERT_WG_TILE_N;
            int k_global = k_tile_base + k_local;
            int n_global = wg_tile_n + n_local;
            if (n_global < inter_dim && k_global < model_dim) {
                w1_up_tile[k_local * EXPERT_WG_TILE_N + n_local] = w1_up_expert[(size_t)n_global * model_dim + k_global];
            } else {
                w1_up_tile[k_local * EXPERT_WG_TILE_N + n_local] = 0.0f;
            }
        }
        __syncthreads();

        // Compute C tile contribution
        #pragma unroll
        for (int k_local = 0; k_local < k_tile_size; ++k_local) {
            #pragma unroll
            for (int m_offset = 0; m_offset < EXPERT_THREAD_TILE_M; ++m_offset) {
                float a_val = token_tile[(thread_tile_m_base + m_offset) * EXPERT_WG_TILE_K + k_local];
                #pragma unroll
                for (int n_offset = 0; n_offset < EXPERT_THREAD_TILE_N; ++n_offset) {
                    float b_gate_val = w1_gate_tile[k_local * EXPERT_WG_TILE_N + (thread_tile_n_base + n_offset)];
                    float b_up_val   = w1_up_tile[k_local * EXPERT_WG_TILE_N + (thread_tile_n_base + n_offset)];
                    gate_accum[m_offset][n_offset] = std::fma(a_val, b_gate_val, gate_accum[m_offset][n_offset]);
                    up_accum[m_offset][n_offset]   = std::fma(a_val, b_up_val, up_accum[m_offset][n_offset]);
                }
            }
        }
        __syncthreads();
    } 

    // Write results from registers to Global Memory
    #pragma unroll
    for (int m_offset = 0; m_offset < EXPERT_THREAD_TILE_M; ++m_offset) {
        int m_local = thread_tile_m_base + m_offset;
        int m_global = wg_tile_m + m_local;
        if (m_global < token_count) {
            #pragma unroll
            for (int n_offset = 0; n_offset < EXPERT_THREAD_TILE_N; ++n_offset) {
                int n_local = thread_tile_n_base + n_offset;
                int n_global = wg_tile_n + n_local;
                if (n_global < inter_dim) {
                    gate_out[(size_t)m_global * inter_dim + n_global] = gate_accum[m_offset][n_offset];
                    up_out[(size_t)m_global * inter_dim + n_global]   = up_accum[m_offset][n_offset];
                }
            }
        }
    }
}


// Kernel to aplly SiLU and gating element-wise
__global__ void expert_silu_gate_kernel(
    const float* __restrict__ gate_in,     // Input: [token_count, inter_dim]
    const float* __restrict__ up_in,       // Input: [token_count, inter_dim]
    float* __restrict__ activated_out,     // Output: [token_count, inter_dim]
    int token_count,
    int inter_dim
) {
    if (token_count == 0) return;
    size_t total_elements = (size_t)token_count * inter_dim;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += (size_t)gridDim.x * blockDim.x)
    {
        float gate_val = gate_in[idx];
        float up_val = up_in[idx];
        float sig = 1.0f / (1.0f + expf(-gate_val));
        float silu = gate_val * sig;
        activated_out[idx] = silu * up_val;
    }
}

// Fused with Scatter
__global__ void expert_gemm_w2_scatter_kernel(
    // GEMM Inputs
    const float* __restrict__ activated,     // Input A: [token_count, inter_dim]
    const float* __restrict__ w2,            // Input B (Weights): [E, model_dim, inter_dim]
    int expert_id,
    int token_count, 
    int model_dim,   
    int inter_dim,   
    // Scatter Inputs
    float* __restrict__ out,                 // Final Output: [token_num, model_dim]
    const int32_t* __restrict__ expert_indices,// Routing info: [E, expert_info_stride_sz]
    const float* __restrict__ expert_weights,  // Routing info: [E, expert_info_stride_sz]
    int token_num,                           // For scatter indexing                            
    size_t expert_info_stride_sz            
) {
     if (token_count == 0) return;
    int wg_tile_m = blockIdx.x * EXPERT_WG_TILE_M;
    int wg_tile_n = blockIdx.y * EXPERT_WG_TILE_N;
    const float* w2_expert = w2 + (size_t)expert_id * model_dim * inter_dim;
    int thread_id = threadIdx.x;
    const int threads_per_row = EXPERT_WG_TILE_N / EXPERT_THREAD_TILE_N;
    int thread_tile_m_base = (thread_id / threads_per_row) * EXPERT_THREAD_TILE_M;
    int thread_tile_n_base = (thread_id % threads_per_row) * EXPERT_THREAD_TILE_N;

    extern __shared__ float smem[];
    float* activated_tile = smem;
    float* w2_tile = activated_tile + EXPERT_WG_TILE_M * EXPERT_WG_TILE_K;
    float output_accum[EXPERT_THREAD_TILE_M][EXPERT_THREAD_TILE_N] = {{0.0f}};

    for (int k_tile_base = 0; k_tile_base < inter_dim; k_tile_base += EXPERT_WG_TILE_K) {
        int k_tile_size = min(EXPERT_WG_TILE_K, inter_dim - k_tile_base);

        // Load A tile
        int num_a_elements = EXPERT_WG_TILE_M * k_tile_size;
        for (int load_idx = thread_id; load_idx < num_a_elements; load_idx += blockDim.x) {
            int m_local = load_idx / k_tile_size; int k_local = load_idx % k_tile_size;
            int m_global = wg_tile_m + m_local; int k_global = k_tile_base + k_local;
            if (m_global < token_count && k_global < inter_dim) {
                 activated_tile[m_local * EXPERT_WG_TILE_K + k_local] = activated[(size_t)m_global * inter_dim + k_global];
            } else { activated_tile[m_local * EXPERT_WG_TILE_K + k_local] = 0.0f; }
        }

        // Load B tile
        int num_b_elements = k_tile_size * EXPERT_WG_TILE_N;
         for (int load_idx = thread_id; load_idx < num_b_elements; load_idx += blockDim.x) {
            int k_local = load_idx / EXPERT_WG_TILE_N; int n_local = load_idx % EXPERT_WG_TILE_N;
            int k_global = k_tile_base + k_local; int n_global = wg_tile_n + n_local;
            if (n_global < model_dim && k_global < inter_dim) {
                w2_tile[k_local * EXPERT_WG_TILE_N + n_local] = w2_expert[(size_t)n_global * inter_dim + k_global];
            } else { w2_tile[k_local * EXPERT_WG_TILE_N + n_local] = 0.0f; }
        }
        __syncthreads();

        // Compute C contribution
        #pragma unroll
        for (int k_local = 0; k_local < k_tile_size; ++k_local) {
            #pragma unroll
            for (int m_offset = 0; m_offset < EXPERT_THREAD_TILE_M; ++m_offset) {
                float a_val = activated_tile[(thread_tile_m_base + m_offset) * EXPERT_WG_TILE_K + k_local];
                #pragma unroll
                for (int n_offset = 0; n_offset < EXPERT_THREAD_TILE_N; ++n_offset) {
                    float b_val = w2_tile[k_local * EXPERT_WG_TILE_N + (thread_tile_n_base + n_offset)];
                    output_accum[m_offset][n_offset] = std::fma(a_val, b_val, output_accum[m_offset][n_offset]);
                }
            }
        }
        __syncthreads();
    } 

    // --- Fused Scatter Write-Back Logic ---
    size_t base_expert_info_offset = (size_t)expert_id * expert_info_stride_sz;
    #pragma unroll
    for (int m_offset = 0; m_offset < EXPERT_THREAD_TILE_M; ++m_offset) {
        int m_local = thread_tile_m_base + m_offset;
        int m_global_expert = wg_tile_m + m_local; // Index within this expert's batch

        if (m_global_expert < token_count) {
            // Get original token index and weight
            int original_token_idx = expert_indices[base_expert_info_offset + m_global_expert];
            float weight = expert_weights[base_expert_info_offset + m_global_expert];

            #pragma unroll
            for (int n_offset = 0; n_offset < EXPERT_THREAD_TILE_N; ++n_offset) {
                int n_local = thread_tile_n_base + n_offset;
                int n_global = wg_tile_n + n_local; // Final output column index

                if (n_global < model_dim) {
                    float result = output_accum[m_offset][n_offset];
                    // Weighted atomicAdd to final output tensor
                    if (result != 0.0f && weight != 0.0f) {
                        atomicAdd(&out[(size_t)original_token_idx * model_dim + n_global], result * weight);
                    }
                }
            }
        }
    }
}


// --- Host Launcher Function ---
torch::Tensor launch_custom_moe(
    torch::Tensor hidden_states,
    torch::Tensor w1,
    torch::Tensor w2,
    torch::Tensor topk_weight,
    torch::Tensor topk_ids
) {
    // Extract dimensions
    const int token_num = hidden_states.size(0);
    const int model_dim = hidden_states.size(1);
    const int topk = topk_weight.size(1);
    const int E = w1.size(0);
    const int inter_dim = w2.size(2);
    const int N_branches = token_num * topk;

    // Initialize output tensor (zeros)
    auto out = torch::zeros_like(hidden_states);

    // Get default HIP stream from PyTorch
    hipStream_t stream = c10::hip::getCurrentHIPStream();

    // Devide strategy
    const int small_inter_dim_threshold = 512;
    const int small_token_threshold = 32;

    // Flatten inputs for all kernels
    // auto hidden_flat = hidden_states.unsqueeze(1).expand({-1, topk, -1}).reshape({N_branches, model_dim}).contiguous();
    // auto topk_w_flat = topk_weight.reshape({N_branches}).contiguous();
    // auto topk_ids_flat = topk_ids.reshape({N_branches}).contiguous();

    // Case 1: Small token count with small inter_dim
    if (token_num <= small_token_threshold && inter_dim <= small_inter_dim_threshold) {
        // Create expert output tensor
        auto options = torch::TensorOptions()
            .dtype(hidden_states.dtype())
            .device(hidden_states.device())
            .requires_grad(false);
            
        auto expert_output = torch::empty({token_num, topk, model_dim}, options);
        
        // Configure kernel parameters
        int num_blocks = token_num * topk;
        int threads_per_block = 512;
        size_t shared_mem_size = (model_dim + 512) * sizeof(float);

        hipLaunchKernelGGL(
            small_process_kernel,
            dim3(num_blocks),
            dim3(threads_per_block), 
            shared_mem_size,
            stream,
            hidden_states.data_ptr<float>(), 
            w1.data_ptr<float>(), 
            w2.data_ptr<float>(), 
            topk_ids.data_ptr<int32_t>(), 
            expert_output.data_ptr<float>(),
            token_num, topk, model_dim, inter_dim
        );
        HIP_CHECK(hipGetLastError());

        int sum_blocks = (token_num * model_dim + 255) / 256;
        hipLaunchKernelGGL(
            weighted_sum_small_kernel,
            dim3(sum_blocks),
            dim3(256),
            0,
            stream,
            expert_output.data_ptr<float>(), 
            topk_weight.data_ptr<float>(), 
            out.data_ptr<float>(), 
            token_num, topk, model_dim
        );
        HIP_CHECK(hipGetLastError());
    }
    // Case 2: Large inter_dim (Expert-Centric - Fused)
    else {
        // Allocate GPU buffers for expert routing information
        size_t expert_info_stride_sz = (size_t)token_num * topk;
        int64_t expert_info_stride = static_cast<int64_t>(expert_info_stride_sz);
        auto expert_indices_options = torch::dtype(torch::kInt32).device(hidden_states.device());
        auto expert_weights_options = torch::dtype(torch::kFloat32).device(hidden_states.device());
        auto expert_counts_options = torch::dtype(torch::kInt32).device(hidden_states.device());

        auto expert_indices = torch::empty({E, expert_info_stride}, expert_indices_options);
        auto expert_weights = torch::empty({E, expert_info_stride}, expert_weights_options);
        auto expert_counts = torch::zeros({E}, expert_counts_options);

        // Launch kernel to populate routing info
        {
            dim3 grid_mask(E);
            dim3 block_mask(EXPERT_BLOCK_SIZE); // Use specific block size for this kernel
            hipLaunchKernelGGL(
                 create_expert_masks, grid_mask, block_mask, 0, stream,
                 topk_ids.data_ptr<int32_t>(), topk_weight.data_ptr<float>(),
                 expert_indices.data_ptr<int32_t>(), expert_weights.data_ptr<float>(),
                 expert_counts.data_ptr<int32_t>(), token_num, topk, E,
                 expert_info_stride_sz); // Pass stride
            HIP_CHECK(hipGetLastError());
        }

        // Copy expert counts from GPU to CPU
        auto expert_counts_cpu = expert_counts.to(torch::kCPU, false);
        auto expert_counts_data = expert_counts_cpu.accessor<int32_t, 1>();

        // Loop through each expert and process its assigned tokens
        for (int expert_id = 0; expert_id < E; ++expert_id) {
            int token_count = expert_counts_data[expert_id];
            if (token_count == 0) continue;

            // Allocate temporary buffers for this expert
            auto expert_tokens = torch::empty({token_count, model_dim}, hidden_states.options());
            auto gate_out = torch::empty({token_count, inter_dim}, hidden_states.options());
            auto up_out = torch::empty({token_count, inter_dim}, hidden_states.options());
            auto activated = torch::empty({token_count, inter_dim}, hidden_states.options());

            //Gather tokens
            {
                size_t gather_elements = (size_t)token_count * model_dim;
                
                // Caculate to ensure that no exceed limits
                unsigned int max_grid_dim_x = 65535; // Or query device
                int gather_grid_size = std::min<unsigned int>((gather_elements + EXPERT_BLOCK_SIZE - 1) / EXPERT_BLOCK_SIZE, max_grid_dim_x);

                dim3 grid_gather(gather_grid_size);
                dim3 block_gather(EXPERT_BLOCK_SIZE);
                hipLaunchKernelGGL(
                     gather_expert_tokens, grid_gather, block_gather, 0, stream,
                     hidden_states.data_ptr<float>(), expert_tokens.data_ptr<float>(),
                     expert_indices.data_ptr<int32_t>(), expert_id, token_count,
                     token_num, model_dim, expert_info_stride_sz); // Pass stride
                 HIP_CHECK(hipGetLastError());
            }

            // Compute W1 GEMM
            {
                int m_tiles = (token_count + EXPERT_WG_TILE_M - 1) / EXPERT_WG_TILE_M;
                int n_tiles = (inter_dim + EXPERT_WG_TILE_N - 1) / EXPERT_WG_TILE_N;
                dim3 grid_w1(m_tiles, n_tiles);
                dim3 block_w1(EXPERT_BLOCK_SIZE);
                size_t smem_w1 = sizeof(float) * (EXPERT_WG_TILE_M * EXPERT_WG_TILE_K + 2 * EXPERT_WG_TILE_K * EXPERT_WG_TILE_N);
                hipLaunchKernelGGL(
                     expert_gemm_w1_kernel, grid_w1, block_w1, smem_w1, stream,
                     expert_tokens.data_ptr<float>(), w1.data_ptr<float>(),
                     gate_out.data_ptr<float>(), up_out.data_ptr<float>(),
                     expert_id, token_count, model_dim, inter_dim);
                 HIP_CHECK(hipGetLastError());
            }

            //Apply SiLU and Gating
            {
                size_t silu_elements = (size_t)token_count * inter_dim;
                unsigned int max_grid_dim_x = 65535;
                int silu_grid_size = std::min<unsigned int>((silu_elements + EXPERT_BLOCK_SIZE - 1) / EXPERT_BLOCK_SIZE, max_grid_dim_x);
                dim3 grid_silu(silu_grid_size);
                dim3 block_silu(EXPERT_BLOCK_SIZE);
                hipLaunchKernelGGL(
                     expert_silu_gate_kernel, grid_silu, block_silu, 0, stream,
                     gate_out.data_ptr<float>(), up_out.data_ptr<float>(),
                     activated.data_ptr<float>(), token_count, inter_dim);
                 HIP_CHECK(hipGetLastError());
            }

            // Compute W2 GEMM and Scatter (Fused Kernel)
            {
                int m_tiles = (token_count + EXPERT_WG_TILE_M - 1) / EXPERT_WG_TILE_M;
                int n_tiles = (model_dim + EXPERT_WG_TILE_N - 1) / EXPERT_WG_TILE_N;
                dim3 grid_w2(m_tiles, n_tiles);
                dim3 block_w2(EXPERT_BLOCK_SIZE);
                size_t smem_w2 = sizeof(float) * (EXPERT_WG_TILE_M * EXPERT_WG_TILE_K + EXPERT_WG_TILE_K * EXPERT_WG_TILE_N);
                hipLaunchKernelGGL(
                    expert_gemm_w2_scatter_kernel, 
                    grid_w2, block_w2, smem_w2, stream,
                    // GEMM args
                    activated.data_ptr<float>(), w2.data_ptr<float>(),
                    expert_id, token_count, model_dim, inter_dim,
                    // Scatter args
                    out.data_ptr<float>(), expert_indices.data_ptr<int32_t>(),
                    expert_weights.data_ptr<float>(), token_num,
                    expert_info_stride_sz 
                );
                 HIP_CHECK(hipGetLastError());
            }
        } 
    }
    return out;
}

// PyTorch binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_custom_moe", &launch_custom_moe,
          "Launch optimized MoE kernel");
}