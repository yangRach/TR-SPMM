#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <string>

namespace py = pybind11;

// ============================================================================
// Data structures
// ============================================================================

struct MatchStats {
    int       total_rows = 0;
    int       total_cols = 0;
    long long total_nnz = 0;
    double    sparsity = 0.0;
    int       num_row_panels = 0;
    int       num_groups = 0;
    int       num_16x16_blocks = 0;
    int       block_padding_groups = 0;
    int       fine_fb = 0;
    int       coarse_fb = 0;
    int       fake_zeros = 0;
};

struct Pair1x2Device {
    int      col1;
    int      col2;
    uint16_t mask1;
    uint16_t mask2;
    uint16_t G1;
    uint16_t G2;
    uint8_t  is_fallback;
};

struct Group2x4Device {
    uint16_t masks[4];
};

// ============================================================================
// Device helpers
// ============================================================================

__device__ __forceinline__ int popcount16(uint16_t x) {
    return __popc(static_cast<unsigned int>(x));
}

__device__ int find_column_index(
    const int* col_ids,
    int num_cols,
    int target_col)
{
    for (int i = 0; i < num_cols; ++i) {
        if (col_ids[i] == target_col) {
            return i;
        }
    }
    return -1;
}

__device__ void insertion_sort_columns(
    int* col_ids,
    uint16_t* masks,
    int num_cols)
{
    for (int i = 1; i < num_cols; ++i) {
        int key_col = col_ids[i];
        uint16_t key_mask = masks[i];
        int j = i - 1;

        while (j >= 0 && col_ids[j] > key_col) {
            col_ids[j + 1] = col_ids[j];
            masks[j + 1] = masks[j];
            --j;
        }

        col_ids[j + 1] = key_col;
        masks[j + 1] = key_mask;
    }
}

// ============================================================================
// CUDA kernel
//
// Baseline implementation:
// - one block handles one row panel (window)
// - thread 0 in that block executes the same logic as the CPU reference
// This preserves the matching order and makes it easy to validate against the
// CPU implementation before later block-level parallelization.
// ============================================================================

__global__ void match_2to4_kernel(
    const int* row_ptr,
    const int* col_ind,
    int rows,
    int window_size,
    MatchStats* stats)
{
    if (threadIdx.x != 0) {
        return;
    }

    const int win_id = blockIdx.x;
    const int row_start = win_id * window_size;
    if (row_start >= rows) {
        return;
    }

    const int row_end = min(row_start + window_size, rows);
    const int win_rows = row_end - row_start;
    const int window_nnz = row_ptr[row_end] - row_ptr[row_start];

    if (window_nnz <= 0) {
        return;
    }

    int* col_ids = new int[window_nnz];
    uint16_t* masks = new uint16_t[window_nnz];

    if (col_ids == nullptr || masks == nullptr) {
        return;
    }

    int num_cols = 0;

    // Step 0: build per-window (col_id -> mask) with a simple linear dictionary.
    for (int r = row_start; r < row_end; ++r) {
        const int local_row = r - row_start;
        const uint16_t bit = static_cast<uint16_t>(1u << local_row);

        for (int e = row_ptr[r]; e < row_ptr[r + 1]; ++e) {
            const int c = col_ind[e];
            const int idx = find_column_index(col_ids, num_cols, c);

            if (idx >= 0) {
                masks[idx] |= bit;
            } else {
                col_ids[num_cols] = c;
                masks[num_cols] = bit;
                ++num_cols;
            }
        }
    }

    if (num_cols == 0) {
        delete[] col_ids;
        delete[] masks;
        return;
    }

    // Step 1: keep original column locality by sorting by col_id.
    insertion_sort_columns(col_ids, masks, num_cols);

    // Step 2: 1:2 pair matching.
    bool* matched = new bool[num_cols];
    Pair1x2Device* pairs = new Pair1x2Device[num_cols];

    if (matched == nullptr || pairs == nullptr) {
        delete[] col_ids;
        delete[] masks;
        delete[] matched;
        delete[] pairs;
        return;
    }

    for (int i = 0; i < num_cols; ++i) {
        matched[i] = false;
    }

    int num_pairs = 0;
    int local_fine_fb = 0;

    const int SEARCH_WINDOW = 32;
    const int T_MAX = 12;

    for (int i = 0; i < num_cols; ++i) {
        if (matched[i]) {
            continue;
        }

        const uint16_t mask_i = masks[i];
        int best_j = -1;
        int min_cost = T_MAX + 1;
        const int j_end = min(i + SEARCH_WINDOW, num_cols);

        for (int j = i + 1; j < j_end; ++j) {
            if (matched[j]) {
                continue;
            }

            const uint16_t and_val = static_cast<uint16_t>(mask_i & masks[j]);
            const int cost = popcount16(and_val);

            if (cost == 0) {
                best_j = j;
                min_cost = 0;
                break;
            }

            if (cost <= T_MAX && cost < min_cost) {
                min_cost = cost;
                best_j = j;
            }
        }

        Pair1x2Device& p = pairs[num_pairs++];
        p.col1 = col_ids[i];
        p.mask1 = mask_i;

        if (best_j != -1) {
            const uint16_t and_val = static_cast<uint16_t>(mask_i & masks[best_j]);
            p.col2 = col_ids[best_j];
            p.mask2 = masks[best_j];
            p.G1 = and_val;
            p.G2 = static_cast<uint16_t>(mask_i | masks[best_j]);
            p.is_fallback = 0;

            matched[i] = true;
            matched[best_j] = true;
        } else {
            p.col2 = -1;
            p.mask2 = 0;
            p.G1 = 0;
            p.G2 = mask_i;
            p.is_fallback = 1;

            matched[i] = true;
            ++local_fine_fb;
        }
    }

    // Step 3: 2:4 group matching.
    bool* pair_used = new bool[num_pairs];
    Group2x4Device* groups = new Group2x4Device[num_pairs];

    if (pair_used == nullptr || groups == nullptr) {
        delete[] col_ids;
        delete[] masks;
        delete[] matched;
        delete[] pairs;
        delete[] pair_used;
        delete[] groups;
        return;
    }

    for (int i = 0; i < num_pairs; ++i) {
        pair_used[i] = false;
    }

    int num_groups_local = 0;
    int local_coarse_fb = 0;
    const int GROUP_SEARCH_WINDOW = 1024;

    for (int i = 0; i < num_pairs; ++i) {
        if (pair_used[i]) {
            continue;
        }

        const Pair1x2Device& pa = pairs[i];
        bool found = false;
        const int j_end = min(i + GROUP_SEARCH_WINDOW, num_pairs);

        for (int j = i + 1; j < j_end; ++j) {
            if (pair_used[j]) {
                continue;
            }

            const Pair1x2Device& pb = pairs[j];
            const uint16_t Z =
                static_cast<uint16_t>((pa.G1 & pb.G2) | (pa.G2 & pb.G1));

            if (Z == 0) {
                Group2x4Device& g = groups[num_groups_local++];
                g.masks[0] = pa.mask1;
                g.masks[1] = pa.mask2;
                g.masks[2] = pb.mask1;
                g.masks[3] = pb.mask2;

                pair_used[i] = true;
                pair_used[j] = true;
                found = true;
                break;
            }
        }

        if (!found) {
            Group2x4Device& g = groups[num_groups_local++];
            g.masks[0] = pa.mask1;
            g.masks[1] = pa.mask2;
            g.masks[2] = 0;
            g.masks[3] = 0;

            pair_used[i] = true;
            local_coarse_fb += 2;
        }
    }

    // Step 4: fake zero padding.
    int local_fake_zeros = 0;

    for (int g = 0; g < num_groups_local; ++g) {
        for (int r = 0; r < win_rows; ++r) {
            const uint16_t row_bit = static_cast<uint16_t>(1u << r);
            int count = 0;

            for (int c = 0; c < 4; ++c) {
                if (groups[g].masks[c] & row_bit) {
                    ++count;
                }
            }

            if (count < 2) {
                local_fake_zeros += (2 - count);
            }
        }
    }

    const int local_blocks = (num_groups_local + 3) / 4;
    const int local_pad_groups = local_blocks * 4 - num_groups_local;

    atomicAdd(&stats->num_groups, num_groups_local);
    atomicAdd(&stats->num_16x16_blocks, local_blocks);
    atomicAdd(&stats->block_padding_groups, local_pad_groups);
    atomicAdd(&stats->fine_fb, local_fine_fb);
    atomicAdd(&stats->coarse_fb, local_coarse_fb);
    atomicAdd(&stats->fake_zeros, local_fake_zeros);

    delete[] col_ids;
    delete[] masks;
    delete[] matched;
    delete[] pairs;
    delete[] pair_used;
    delete[] groups;
}

// ============================================================================
// Host wrapper
// ============================================================================

py::dict match_2to4_cuda(
    torch::Tensor row_ptr,
    torch::Tensor col_ind,
    torch::Tensor values,
    int window_size = 16)
{
    TORCH_CHECK(row_ptr.is_cuda(), "row_ptr must be a CUDA tensor");
    TORCH_CHECK(col_ind.is_cuda(), "col_ind must be a CUDA tensor");
    TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");

    TORCH_CHECK(row_ptr.scalar_type() == torch::kInt32, "row_ptr must be int32");
    TORCH_CHECK(col_ind.scalar_type() == torch::kInt32, "col_ind must be int32");
    TORCH_CHECK(values.scalar_type() == torch::kFloat32, "values must be float32");

    TORCH_CHECK(row_ptr.dim() == 1, "row_ptr must be a 1D tensor");
    TORCH_CHECK(col_ind.dim() == 1, "col_ind must be a 1D tensor");
    TORCH_CHECK(values.dim() == 1, "values must be a 1D tensor");
    TORCH_CHECK(col_ind.size(0) == values.size(0), "col_ind and values must have the same length");

    TORCH_CHECK(window_size >= 1 && window_size <= 16,
                "window_size must be in [1, 16] for the uint16_t mask implementation");

    row_ptr = row_ptr.contiguous();
    col_ind = col_ind.contiguous();
    values = values.contiguous();

    const int rows = static_cast<int>(row_ptr.size(0)) - 1;
    const int nnz = static_cast<int>(col_ind.size(0));
    const int cols = (nnz > 0) ? (col_ind.max().item<int>() + 1) : 0;

    TORCH_CHECK(rows > 0, "invalid matrix shape: rows must be > 0");
    TORCH_CHECK(cols > 0, "invalid matrix shape: cols must be > 0");

    MatchStats host_stats{};
    host_stats.total_rows = rows;
    host_stats.total_cols = cols;
    host_stats.total_nnz = static_cast<long long>(nnz);
    host_stats.sparsity =
        1.0 - static_cast<double>(host_stats.total_nnz) / (static_cast<double>(rows) * cols);
    host_stats.num_row_panels = (rows + window_size - 1) / window_size;

    MatchStats* device_stats = nullptr;
    cudaError_t err = cudaMalloc(&device_stats, sizeof(MatchStats));
    TORCH_CHECK(err == cudaSuccess, "cudaMalloc failed for device_stats: ", cudaGetErrorString(err));

    err = cudaMemcpy(device_stats, &host_stats, sizeof(MatchStats), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(device_stats);
        TORCH_CHECK(false, "cudaMemcpy host->device failed: ", cudaGetErrorString(err));
    }

    // The baseline kernel uses device-side new/delete. Reserve a larger heap so
    // moderate-sized windows can be handled reliably.
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, static_cast<size_t>(512) << 20);

    const int num_windows = host_stats.num_row_panels;
    match_2to4_kernel<<<num_windows, 1>>>(
        row_ptr.data_ptr<int>(),
        col_ind.data_ptr<int>(),
        rows,
        window_size,
        device_stats);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(device_stats);
        TORCH_CHECK(false, "match_2to4_kernel launch failed: ", cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(device_stats);
        TORCH_CHECK(false, "match_2to4_kernel execution failed: ", cudaGetErrorString(err));
    }

    err = cudaMemcpy(&host_stats, device_stats, sizeof(MatchStats), cudaMemcpyDeviceToHost);
    cudaFree(device_stats);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpy device->host failed: ", cudaGetErrorString(err));

    std::printf("\n");
    std::printf("============================================================\n");
    std::printf("  2:4 Structured Sparse Matching Stats (CUDA baseline)\n");
    std::printf("============================================================\n");
    std::printf("  Matrix shape:          %d x %d\n", host_stats.total_rows, host_stats.total_cols);
    std::printf("  Total nnz:             %lld\n", host_stats.total_nnz);
    std::printf("  Sparsity:              %.4f%%\n", host_stats.sparsity * 100.0);
    std::printf("  Row panels:            %d\n", host_stats.num_row_panels);
    std::printf("  16x4 groups:           %d\n", host_stats.num_groups);
    std::printf("  16x16 sparse blocks:   %d\n", host_stats.num_16x16_blocks);
    std::printf("  Block padding groups:  %d\n", host_stats.block_padding_groups);
    std::printf("  Fine fallback:         %d\n", host_stats.fine_fb);
    std::printf("  Coarse fallback:       %d\n", host_stats.coarse_fb);
    std::printf("  Fake zeros:            %d\n", host_stats.fake_zeros);
    std::printf("============================================================\n\n");

    py::dict result;
    result["total_rows"] = host_stats.total_rows;
    result["total_cols"] = host_stats.total_cols;
    result["total_nnz"] = host_stats.total_nnz;
    result["sparsity"] = host_stats.sparsity;
    result["num_row_panels"] = host_stats.num_row_panels;
    result["num_groups"] = host_stats.num_groups;
    result["num_16x16_blocks"] = host_stats.num_16x16_blocks;
    result["block_padding_groups"] = host_stats.block_padding_groups;
    result["fine_fallback"] = host_stats.fine_fb;
    result["coarse_fallback"] = host_stats.coarse_fb;
    result["fake_zeros"] = host_stats.fake_zeros;
    return result;
}

// ============================================================================
// Module definition
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA baseline implementation of 2:4 structured sparse matching";

    m.def("match_2to4",
          &match_2to4_cuda,
          py::arg("row_ptr"),
          py::arg("col_ind"),
          py::arg("values"),
          py::arg("window_size") = 16,
          "Run 2:4 structured sparse matching on CUDA tensors");
}
