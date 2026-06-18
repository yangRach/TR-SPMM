/**
 * block_match_cuda.cu — CUDA 2:4 结构化稀疏匹配的性能优化实现，与cpp相比，group分布略不同
 *
 * ====== 与 CPU 参考版本 (block_match_windows.cpp) 的关系 ======
 *
 * 核心目标一致：
 *   将稀疏矩阵按 16 行划分为窗口（Row Panel），通过位运算在窗口内对列进行
 *   1:2（两列配对 Pair）和 2:4（四列成组 Group）的结构化匹配，
 *   确保每组 4 列中每行最多 2 个非零元，符合 GPU 2:4 结构化稀疏的硬件格式要求。
 *
 * 算法步骤与 CPU 版本一一对应（Step 0 ~ Step 4）：
 *   Step 0 — 窗口划分与编码：将窗口内每一列的 nnz 分布编码为 uint16_t mask
 *   Step 1 — 列排序（可选）：按原始列索引排序，保留列局部性
 *   Step 2 — 1:2 列配对：局部搜索 + 冲突检测，组合成 Pair（G1=AND, G2=OR）
 *   Step 3 — 2:4 分组配对：逻辑门冲突检测 Z=(G1&G4)|(G2&G3)，组合成 4 列 Group
 *   Step 4 — 假0填充：每个 Group 的每行不足 2 个 nnz 的位置补 0
 *
 * ====== 与 CPU 版本的关键差异（为性能优化所做的改动） ======
 *
 * 1. 并行度：
 *    CPU：单线程串行处理所有窗口
 *    CUDA：每个 Block（128 线程）并行处理一个窗口，所有窗口同时运行
 *
 * 2. Step0 列去重：
 *    CPU：std::unordered_map + vector 拷贝
 *    CUDA：并行哈希表（atomicCAS + atomicOr），所有线程同时构建
 *
 * 3. Step2/Step3 候选搜索：
 *    CPU：纯串行扫描（窗口限定的局部搜索）
 *    CUDA：Warp 归约 + Block 级并行搜索，搜索半径更大
 *
 * 4. 匹配策略：
 *    CPU：贪心顺序（找到第一个 cost==0 即锁定），结果确定
 *    CUDA：贪心 + 并行寻优（同步后取全局最优），结果可能不同
 *
 * 5. 数据存储：
 *    CPU：std::vector / std::unordered_map（主机堆，小对象开销大）
 *    CUDA：Global Memory 预分配池 + Shared Memory 哈希表（无动态分配）
 *
 * ====== 注意事项 ======
 * - 本实现与 CPU 版本不保证逐列逐组结果完全相同
 *   （启发式搜索顺序不同，GPU 的原子操作引入非确定性）
 * - 但两者都保证输出满足 2:4 结构合法性，并通过同一套统计校验
 * - 本实现在大规模稀疏矩阵（如 Bump_2911）上显著优于 CPU
 * - 在每窗口列数密集的矩阵（如 mip1）上接近或略慢于 CPU
 *
 * ====== 暴露接口 ======
 *   matching_utils_cuda.match_2to4(row_ptr, col_ind, values, window_size=16)
 *   输入：CSR 格式的 CUDA 张量（row_ptr:int32, col_ind:int32, values:float32）
 *   输出：匹配统计信息字典
 */

#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <string>

namespace py = pybind11;

// ============================================================================
// 数据结构
// ============================================================================
// 与 CPU 版本的 MatchStats / Pair1x2 / Group2x4 一一对应，但去掉字段以节省存储。

/**
 * 匹配统计信息 —— 与 CPU 版本 (block_match_windows.cpp) 输出字段完全一致。
 *
 * 校验关系（在 Python 层验证）：
 *   num_16x16_blocks = ceil(num_groups / 4)
 *   block_padding_groups = num_16x16_blocks * 4 - num_groups
 *   total_nnz + fake_zeros <= num_groups * 32    （容量自洽性约束）
 */
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

/**
 * 1:2 配对（Pair）数据 —— CPU 版本 (block_match_windows.cpp) 中 struct Pair1x2 的对应。
 *
 * 与 CPU 版的区别：
 *   CPU 版保存了 col1/col2 原始列索引供后续 Step4 查询 col_mask_of；
 *   本实现只在 GPU 上使用 mask 位运算完成所有计算，不再需要列索引，
 *   因此去掉了 col1/col2 字段以节省显存带宽。
 *
 * 字段含义（与 CPU 版完全一致）：
 *   mask1, mask2 — 两列各自的 uint16_t nnz 分布编码
 *   G1 = mask1 & mask2  两列同时存在的行（冲突行）
 *   G2 = mask1 | mask2  两列的并集（有效行）
 */
struct Pair1x2Device {
    uint16_t mask1;
    uint16_t mask2;
    uint16_t G1;
    uint16_t G2;
    uint8_t  nnz;
};

/**
 * CUDA kernel 配置常量
 *
 * kBlockThreads      每个 Block 的线程数（= 128，4 个 Warp）。
 *                    所有线程协作完成 Step0 哈希构建和 Step2/Step3 的并行搜索。
 * kSearchWindow      Step2（1:2 配对）中，每个列向后搜索伙伴的最大范围。
 *                    与 CPU 版的 SEARCH_WINDOW=32 不同——GPU 搜索更大窗口收益更高，
 *                    因为并行搜索的成本不随窗口增大而显著增加。
 * kGroupSearchWindow Step3（2:4 分组）中，每个 Pair 向后搜索伙伴的最大范围。
 *                    与 CPU 版的 GROUP_SEARCH_WINDOW=1024 一致。
 */
constexpr int kBlockThreads = 128;
constexpr int kSearchWindow = 512;
constexpr int kGroupSearchWindow = 1024;
constexpr int kWarpSize = 32;

// ============================================================================
// Device 辅助函数
// ============================================================================

/**
 * 计算 uint16_t 的 popcount（置 1 的位数）。
 *
 * 对应 CPU 版本 (block_match_windows.cpp) 的 __builtin_popcount()。
 * 用于快速计算两列之间的冲突行数（cost = popcount(mask_i & mask_j)）。
 *
 * 注意：__popc 是 PTX 指令，编译为单条 POPC 指令，延迟约 4 个周期。
 */
__device__ __forceinline__ int popcount16(uint16_t x) {
    return __popc(static_cast<unsigned int>(x));
}

/**
 * Warp 级并行归约：在 warp 的所有活跃 lane 中找出 (score, index) 最优对。
 *
 * 对应 CPU 版本 (block_match_windows.cpp) 中 Step2/Step3 内层串行扫描的替代。
 *
 * 原理：
 *   使用 __shfl_down_sync 在 warp 内以对数步数（log2(kWarpSize)=5 步）
 *   完成所有 32 个 lane 的归宿约，选出 score 最小（冲突最少）的候选。
 *
 * 与 CPU 版的差异：
 *   CPU：按顺序扫描，找到第一个 cost==0 就提前终止
 *   CUDA：扫描完整窗口并取全局最优，不提前终止
 *   这意味着 CUDA 可能找到比 CPU 更好的匹配，但结果顺序不同。
 */
__device__ __forceinline__ void warp_best_pair(int& best_score, int& best_index) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        const int other_score = __shfl_down_sync(0xffffffffu, best_score, offset);
        const int other_index = __shfl_down_sync(0xffffffffu, best_index, offset);
        if (other_index != -1 &&
            (best_index == -1 || other_score < best_score ||
             (other_score == best_score && other_index < best_index))) {
            best_score = other_score;
            best_index = other_index;
        }
    }
}

/**
 * 计算一个 2:4 Group 的假0填充（Fake Zero）数量。
 *
 * 对应 CPU 版本 (block_match_windows.cpp) 的 Step 4 逐行统计逻辑。
 *
 * 与 CPU 版的区别：
 *   CPU 版在 Step4 通过 col_mask_of 查询原始列 ID 的 mask；
 *   本实现直接使用传入的 mask0~mask3 做位运算，无需列 ID 查找表。
 *
 * 算法：
 *   对每行（0..win_rows）：
 *     统计该行在 4 个 mask 中的非零元数 count
 *     如果 count < 2，则需要填充 (2 - count) 个假0
 *   返回该 Group 的总假0数
 */
__device__ __forceinline__ int fake_zero_cost(
    uint16_t mask0,
    uint16_t mask1,
    uint16_t mask2,
    uint16_t mask3,
    int win_rows)
{
    int fake_zeros = 0;
    for (int r = 0; r < win_rows; ++r) {
        const uint16_t row_bit = static_cast<uint16_t>(1u << r);
        int count = 0;
        count += (mask0 & row_bit) ? 1 : 0;
        count += (mask1 & row_bit) ? 1 : 0;
        count += (mask2 & row_bit) ? 1 : 0;
        count += (mask3 & row_bit) ? 1 : 0;
        if (count < 2) {
            fake_zeros += (2 - count);
        }
    }
    return fake_zeros;
}

// ============================================================================
// （未使用的）设备端辅助函数
//
// find_column_index / insertion_sort_columns：
//   这些是 CPU 版本中使用的函数。
//   CUDA 版本使用并行哈希表替代了线性查找，使用密度桶排序替代了插入排序，
//   因此这两个函数在当前实现中不再被调用，保留作为算法文档参考。
// ============================================================================

/**
 * 在 col_ids[0..num_cols) 中线性查找 target_col，返回索引或 -1。
 *
 * 对应 CPU 版本 (block_match_windows.cpp) 中 unordered_map 的 find() 语义。
 * 当前 CUDA 实现用并行哈希表替代了此线性查找，此函数仅作参考。
 */
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

/**
 * 对 col_ids[0..num_cols) 按列号升序做插入排序，masks 同步交换。
 *
 * 对应 CPU 版本 (block_match_windows.cpp) 的 std::sort(by col_id)。
 * 当前 CUDA 实现用密度桶排序（O(n)）替代了此 O(n^2) 插入排序，
 * 此函数仅作算法参考。
 */
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
// ============================================================================
// 主 CUDA Kernel: match_2to4_kernel
//
// 对应 CPU 版本 (block_match_windows.cpp)：
//   static MatchStats run_2to4_matching() 中的逐窗口串行循环。
//
// 映射关系：
//   CPU: for (int win = 0; win < num_windows; ++win) { ... }
//   CUDA: match_2to4_kernel<<<num_windows, kBlockThreads>>>(...)
//
// 每个 CUDA Block 并行处理一个 Row Panel（16 行窗口），
// Block 内 128 个线程协作完成 Step0~Step4 的匹配与统计。
//
// Kernel 参数说明：
//   row_ptr, col_ind    — CSR 格式（CUDA 张量指针）
//   rows, window_size   — 矩阵行数和窗口大小
//   workspace_*         — 主机侧预分配的全局工作区（避免设备端 new/delete）
//   stats               — 输出统计信息（AtomicAdd 汇总）
// ============================================================================

__global__ void match_2to4_kernel(
    const int* row_ptr,
    const int* col_ind,
    int rows,
    int window_size,
    int* workspace_temp,
    uint16_t* workspace_masks,
    uint8_t* workspace_matched,
    Pair1x2Device* workspace_pairs,
    uint8_t* workspace_pair_used,
    int* workspace_hash_keys,
    unsigned int* workspace_hash_masks,
    MatchStats* stats)
{
    if (threadIdx.x >= kBlockThreads) {
        return;
    }

    // ======================================================================
    // 窗口定位：与 CPU 版本 run_2to4_matching() 的逐窗口循环对应
    //   CPU: for (int win = 0; win < num_windows; ++win)
    //   CUDA: win_id = blockIdx.x （每个 Block 处理一个窗口）
    //
    // workspace_offset = row_ptr[row_start]（该窗口第一个非零元的全局索引）
    // 用于在预分配的工作区数组中定位该窗口的独立切片。
    // ======================================================================
    const int win_id = blockIdx.x;
    const int row_start = win_id * window_size;
    if (row_start >= rows) {
        return;
    }

    const int row_end = min(row_start + window_size, rows);
    const int win_rows = row_end - row_start;
    const int workspace_offset = row_ptr[row_start];
    const int window_nnz = row_ptr[row_end] - row_ptr[row_start];

    if (window_nnz <= 0) {
        return;
    }

    // 工作区指针：每个窗口通过 workspace_offset 切分全局预分配缓冲
    int* temp_storage = workspace_temp + workspace_offset;
    uint16_t* masks = workspace_masks + workspace_offset;
    uint8_t* matched = workspace_matched + workspace_offset;
    Pair1x2Device* pairs = workspace_pairs + workspace_offset;
    uint8_t* pair_used = workspace_pair_used + workspace_offset;

    __shared__ int shared_num_cols;     // Step0 输出去重列数
    __shared__ int shared_num_pairs;    // Step2 输出配对数量
    __shared__ int warp_best_score[kBlockThreads / kWarpSize];  // Warp 归约缓冲区
    __shared__ int warp_best_index[kBlockThreads / kWarpSize];  // Warp 归约缓冲区
    __shared__ int shared_counts[kBlockThreads];
    __shared__ int shared_warp_sums[kBlockThreads / kWarpSize];
    __shared__ int shared_warp_offsets[kBlockThreads / kWarpSize];

    // ======================================================================
    // Step 0: 窗口划分与编码（并行哈希表版本）
    //
    // 对应 CPU 版本 (block_match_windows.cpp)：
    //   std::unordered_map<int, uint16_t> col_mask_of + vector 拷贝
    //
    // 与 CPU 版的差异：
    //   CPU  使用 unordered_map（红黑树），每行串行插入，O(n log n)
    //   CUDA 使用开放定址并行哈希表，所有线程同时插入
    //         atomicCAS + atomicOr 是此处的关键原子操作
    //
    // 哈希表大小 = 2 * window_nnz（保证负载因子 < 0.5，减少冲突）
    // ======================================================================
    const int hash_offset = workspace_offset << 1;
    const int hash_size = window_nnz << 1;
    int* hash_keys = workspace_hash_keys + hash_offset;
    unsigned int* hash_masks = workspace_hash_masks + hash_offset;

    for (int i = threadIdx.x; i < hash_size; i += blockDim.x) {
        hash_keys[i] = -1;
        hash_masks[i] = 0u;
    }
    __syncthreads();

    // ---- 并行哈希插入 ----
    // 每个线程处理连续的多行（按 threadIdx.x 步进），
    // 对该行内的每个非零元，通过原子操作写入哈希表。
    // atomicCAS(hash_keys[h], -1, c)：如果槽空则占用，否则检查是否当前列
    // atomicOr(hash_masks[h], bit)：在对应槽的 mask 中设置该行的 bit
    for (int r = row_start + threadIdx.x; r < row_end; r += blockDim.x) {
        const int local_row = r - row_start;
        const unsigned int bit = 1u << local_row;

        for (int e = row_ptr[r]; e < row_ptr[r + 1]; ++e) {
            const int c = col_ind[e];
            unsigned int h = (static_cast<unsigned int>(c) * 2654435761u) % static_cast<unsigned int>(hash_size);
            for (int probe = 0; probe < hash_size; ++probe) {
                const int prev = atomicCAS(&hash_keys[h], -1, c);
                if (prev == -1 || prev == c) {
                    atomicOr(&hash_masks[h], bit);
                    break;
                }
                ++h;
                if (h == static_cast<unsigned int>(hash_size)) {
                    h = 0;
                }
            }
        }
    }
    __syncthreads();

    // ---- 将哈希表结果展开到 masks 数组 ----
    // 用 block 级前缀和替代 atomicAdd(compaction)，减少 shared 原子争用。
    if (threadIdx.x == 0) {
        shared_num_cols = 0;
        shared_num_pairs = 0;
    }
    __syncthreads();

    int local_count = 0;
    for (int i = threadIdx.x; i < hash_size; i += blockDim.x) {
        local_count += (hash_keys[i] != -1) ? 1 : 0;
    }
    int x = local_count;
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, offset);
        if ((threadIdx.x & (kWarpSize - 1)) >= offset) {
            x += y;
        }
    }

    const int warp_id0 = threadIdx.x / kWarpSize;
    const int lane0 = threadIdx.x & (kWarpSize - 1);
    if (lane0 == kWarpSize - 1) {
        shared_warp_sums[warp_id0] = x;
    }
    __syncthreads();

    if (warp_id0 == 0) {
        int w = (lane0 < (kBlockThreads / kWarpSize)) ? shared_warp_sums[lane0] : 0;
        for (int offset = 1; offset < kWarpSize; offset <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, w, offset);
            if (lane0 >= offset) {
                w += y;
            }
        }
        if (lane0 < (kBlockThreads / kWarpSize)) {
            shared_warp_offsets[lane0] = w - shared_warp_sums[lane0];
        }
        if (lane0 == (kBlockThreads / kWarpSize) - 1) {
            shared_num_cols = w;
        }
    }
    __syncthreads();

    const int prefix = shared_warp_offsets[warp_id0] + (x - local_count);

    int out = prefix;
    for (int i = threadIdx.x; i < hash_size; i += blockDim.x) {
        if (hash_keys[i] != -1) {
            masks[out++] = static_cast<uint16_t>(hash_masks[i]);
        }
    }
    __syncthreads();

    const int num_cols = shared_num_cols;
    if (num_cols == 0) {
        return;
    }

    // ======================================================================
    // 步骤 1: （当前未显式执行排序）
    //
    // 对应 CPU 版本 (block_match_windows.cpp)：
    //   std::sort(columns.begin(), columns.end(), by col_id)
    //
    // 当前 CUDA 实现跳过了显式排序步骤，原因：
    //   1. CUDA 在 Step2 中使用并行搜索（搜索窗口较大），
    //      不依赖相邻列索引的局部性来保证匹配质量。
    //   2. 排序本身在 GPU 上需要额外的全局排序或桶排序，
    //      在当前算法中收益小于开销。
    //   3. 如果需要排序恢复，可以在此处加入密度桶排序。
    // ======================================================================

    // ======================================================================
    // 步骤 2: 1:2 列配对（Pair matching）
    //
    // 对应 CPU 版本 (block_match_windows.cpp)：
    //   for (int i = 0; i < n; ++i) {
    //     for (int j = i+1; j < min(i+SEARCH_WINDOW, n); ++j) { ... }
    //   }
    //
    // 与 CPU 版的差异：
    //   CPU 按顺序扫描，找到第一个 cost==0 就提前终止
    //   CUDA：外层 for (int i = 0; i < num_cols; ++i) 仍串行（由 thread0 控制）
    //          内层并行：128 线程共同扫描候选列，warp 归约取全局最优
    //
    // 评分函数：score = overlap * 64 + (16 - partner_nnz)
    //   overlap 越小越好（冲突少），partner_nnz 越大越好（更稠密的列优先）
    // ======================================================================
    for (int i = threadIdx.x; i < num_cols; i += blockDim.x) {
        matched[i] = 0;
        temp_storage[i] = popcount16(masks[i]);
    }
    __syncthreads();

    int local_fine_fb = 0;
    const int T_MAX = 12;            // 最大允许冲突行数（与 CPU 版一致）
    const int SEARCH_WINDOW = (num_cols > 4096) ? 256 : kSearchWindow;
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;

    if (threadIdx.x == 0) {
        shared_num_pairs = 0;
    }
    __syncthreads();

    // ---- 外层贪心循环 ----
    // 对应 CPU: for (int i = 0; i < n; ++i)
    // 线程 0 控制循环，所有线程参与内层搜索。
    for (int i = 0; i < num_cols; ++i) {
        if (matched[i]) {
            continue;
        }

        const uint16_t mask_i = masks[i];
        int best_score = 1 << 30;
        int best_j = -1;
        const int j_end = min(num_cols, i + 1 + SEARCH_WINDOW);

        // ---- 并行扫描候选列 ----
        // 对应 CPU: for (int j = i + 1; j < j_end; ++j)
        // 所有 128 线程按步进分布扫描，每个线程维护自己的局部最优。
        for (int j = i + 1 + threadIdx.x; j < j_end; j += blockDim.x) {
            if (matched[j]) {
                continue;
            }

            const int overlap = popcount16(static_cast<uint16_t>(mask_i & masks[j]));
            if (overlap > T_MAX) {
                continue;
            }

            // 评分：重叠行数越少越好；相同时选 nnz 更多的列（更稠密）
            const int score = overlap * 64 + (16 - temp_storage[j]);
            if (score < best_score || (score == best_score && j < best_j)) {
                best_score = score;
                best_j = j;
            }
        }

        // ---- Warp 归约 + Block 级归约取全局最优 ----
        // 对应 CPU 的串行比较取 best_j 的过程。
        warp_best_pair(best_score, best_j);
        if (lane == 0) {
            warp_best_score[warp_id] = best_score;
            warp_best_index[warp_id] = best_j;
        }
        __syncthreads();

        if (warp_id == 0) {
            int block_best_score = 1 << 30;
            int block_best_index = -1;
            if (lane < (kBlockThreads / kWarpSize)) {
                block_best_score = warp_best_score[lane];
                block_best_index = warp_best_index[lane];
            }
            warp_best_pair(block_best_score, block_best_index);
            if (lane == 0) {
                warp_best_index[0] = block_best_index;
            }
        }

        // ---- 线程 0 执行配对与写入 ----
        // 对应 CPU 的 Pair 创建逻辑。
        // 使用 shared_num_pairs++ 原子分配 pair 索引，
        // 确保所有线程在后续 Step3 中能读到正确的 num_pairs。
        if (threadIdx.x == 0) {
            const int final_j = warp_best_index[0];
            const int pair_idx = shared_num_pairs++;
            Pair1x2Device& p = pairs[pair_idx];
            p.mask1 = mask_i;
            if (final_j != -1) {
                p.mask2 = masks[final_j];
                p.G1 = static_cast<uint16_t>(mask_i & masks[final_j]);
                p.G2 = static_cast<uint16_t>(mask_i | masks[final_j]);
                matched[final_j] = 1;
                p.nnz = static_cast<uint8_t>(temp_storage[i] + temp_storage[final_j]);
            } else {
                // 细粒度 Fallback：找不到兼容列，用虚拟全0列作为 partner
                // 对应 CPU: stats.fine_fb++
                p.mask2 = 0;
                p.G1 = 0;
                p.G2 = mask_i;
                ++local_fine_fb;
                p.nnz = static_cast<uint8_t>(temp_storage[i]);
            }
            matched[i] = 1;
        }
        __syncthreads();
    }

    const int num_pairs = shared_num_pairs;
    for (int i = threadIdx.x; i < num_pairs; i += blockDim.x) {
        pair_used[i] = 0;
    }
    __syncthreads();

    // ======================================================================
    // 步骤 3: 2:4 分组匹配（Group matching）
    //
    // 对应 CPU 版本 (block_match_windows.cpp)：
    //   for (int i = 0; i < np; ++i) {
    //     for (int j = i+1; j < min(i+GROUP_SEARCH_WINDOW, np); ++j) {
    //       Z = (pa.G1 & pb.G2) | (pa.G2 & pb.G1);
    //       if (Z == 0) { ... group found ... }
    //     }
    //   }
    //
    // 逻辑门冲突检测（核心位运算，与 CPU 版完全一致）：
    //   Z = (G1 & G4) | (G2 & G3)
    //   Z == 0 ⇔ 两 Pair 合并后每行最多 2 个非零元
    //
    // 与 CPU 版的差异：
    //   CPU 按顺序扫描，找到第一个 Z==0 就锁定
    //   CUDA：内层用 128 线程并行扫描，取总 nnz 最大（约等于假0最少）的组，
    //          这样可以贪心地让 packing 更紧密。
    //
    // 评分函数：
    //   score = 64 - (pairA_nnz + pairB_nnz)
    //   选总 nnz 最大的组合，因为 nnz 越大意味着需要填充的假0越少。
    // ======================================================================
    int num_groups_local = 0;
    int local_coarse_fb = 0;
    int local_fake_zeros = 0;
    const int GROUP_SEARCH_WINDOW = (num_pairs > 2048) ? 512 : kGroupSearchWindow;

    for (int i = 0; i < num_pairs; ++i) {
        if (pair_used[i]) {
            continue;
        }

        const Pair1x2Device pa = pairs[i];
        int best_group_score = 1 << 30;
        int best_group_j = -1;
        const int j_end = min(num_pairs, i + 1 + GROUP_SEARCH_WINDOW);

        // ---- 并行扫描候选 Pair ----
        // 对应 CPU: for (int j = i + 1; j < j_end; ++j)
        for (int j = i + 1 + threadIdx.x; j < j_end; j += blockDim.x) {
            if (pair_used[j]) {
                continue;
            }

            const Pair1x2Device pb = pairs[j];
            const uint16_t conflict =
                static_cast<uint16_t>((pa.G1 & pb.G2) | (pa.G2 & pb.G1));
            if (conflict != 0) {
                continue;
            }

            const int total_nnz = static_cast<int>(pa.nnz) + static_cast<int>(pb.nnz);
            const int score = 64 - total_nnz;
            if (score < best_group_score || (score == best_group_score && j < best_group_j)) {
                best_group_score = score;
                best_group_j = j;
            }
        }

        // Warp 归约 + Block 级归约取全局最优
        warp_best_pair(best_group_score, best_group_j);
        if (lane == 0) {
            warp_best_score[warp_id] = best_group_score;
            warp_best_index[warp_id] = best_group_j;
        }
        __syncthreads();

        if (warp_id == 0) {
            int block_best_score = 1 << 30;
            int block_best_index = -1;
            if (lane < (kBlockThreads / kWarpSize)) {
                block_best_score = warp_best_score[lane];
                block_best_index = warp_best_index[lane];
            }
            warp_best_pair(block_best_score, block_best_index);
            if (lane == 0) {
                warp_best_index[0] = block_best_index;
            }
        }

        // ---- 线程 0 执行分组与假0累计 ----
        if (threadIdx.x == 0) {
            const int selected_j = warp_best_index[0];
            ++num_groups_local;
            pair_used[i] = 1;
            if (selected_j != -1) {
                // 找到兼容 Pair：组成 2:4 Group（4 列）
                pair_used[selected_j] = 1;
                const Pair1x2Device pb = pairs[selected_j];
                // 直接累计假0，不再保存 groups 数组（节省显存和带宽）
                local_fake_zeros +=
                    fake_zero_cost(pa.mask1, pa.mask2, pb.mask1, pb.mask2, win_rows);
            } else {
                // 粗粒度 Fallback：找不到兼容 Pair，补两列全0
                // 对应 CPU: stats.coarse_fb += 2
                local_coarse_fb += 2;
                local_fake_zeros += fake_zero_cost(pa.mask1, pa.mask2, 0, 0, win_rows);
            }
        }
        __syncthreads();
    }

    // ======================================================================
    // 步骤 4: 假0填充统计 & Block 打包
    //
    // 假0填充已在上面的 Group 创建时直接累计（fake_zero_cost），
    // 不再像 CPU 版那样先生成 Group 列表再二次扫描统计。
    //
    // 硬件映射：16×4 Group → 16×16 Sparse Block
    //   对应 CPU 版本的 Block 打包逻辑：
    //     blocks = ceil(num_groups / 4)
    //     pad_groups = blocks * 4 - num_groups
    // ======================================================================
    if (threadIdx.x == 0) {
        const int local_blocks = (num_groups_local + 3) / 4;
        const int local_pad_groups = local_blocks * 4 - num_groups_local;
        atomicAdd(&stats->num_groups, num_groups_local);
        atomicAdd(&stats->num_16x16_blocks, local_blocks);
        atomicAdd(&stats->block_padding_groups, local_pad_groups);
        atomicAdd(&stats->fine_fb, local_fine_fb);
        atomicAdd(&stats->coarse_fb, local_coarse_fb);
        atomicAdd(&stats->fake_zeros, local_fake_zeros);
    }
}

// ============================================================================
// Host 端包装函数
//
// 对应 CPU 版本 (block_match_windows.cpp)：
//   py::dict match_2to4_py(...)
//
// 功能：
//   1. 校验输入张量的类型和维度
//   2. 在 GPU 上预分配所有工作区缓冲区
//   3. 启动 CUDA kernel（<<<num_windows, kBlockThreads>>>）
//   4. 同步并回收结果统计信息
//   5. 输出格式化统计表格，返回 Python 字典
//
// 工作区设计：
//   所有工作区（temp, masks, matched, pairs, pair_used, hash_keys, hash_masks）
//   在 host 端一次性分配，kernel 内通过 workspace_offset 切分使用。
//   避免了设备端 new/delete 的开销，并消除了设备堆管理的不确定性。
// ============================================================================

py::dict match_2to4_cuda(
    torch::Tensor row_ptr,
    torch::Tensor col_ind,
    torch::Tensor values,
    int window_size = 16)
{
    TORCH_CHECK(row_ptr.is_cuda(), "row_ptr 必须是 CUDA 张量");
    TORCH_CHECK(col_ind.is_cuda(), "col_ind 必须是 CUDA 张量");
    TORCH_CHECK(values.is_cuda(), "values 必须是 CUDA 张量");

    TORCH_CHECK(row_ptr.scalar_type() == torch::kInt32, "row_ptr 必须为 int32");
    TORCH_CHECK(col_ind.scalar_type() == torch::kInt32, "col_ind 必须为 int32");
    TORCH_CHECK(values.scalar_type() == torch::kFloat32, "values 必须为 float32");

    TORCH_CHECK(row_ptr.dim() == 1, "row_ptr 必须是一维张量");
    TORCH_CHECK(col_ind.dim() == 1, "col_ind 必须是一维张量");
    TORCH_CHECK(values.dim() == 1, "values 必须是一维张量");
    TORCH_CHECK(col_ind.size(0) == values.size(0), "col_ind 和 values 长度必须相同");

    TORCH_CHECK(window_size >= 1 && window_size <= 16,
                "window_size 必须在 [1, 16] 范围内（uint16_t mask 实现限制）");

    row_ptr = row_ptr.contiguous();
    col_ind = col_ind.contiguous();
    values = values.contiguous();

    const int rows = static_cast<int>(row_ptr.size(0)) - 1;
    const int nnz = static_cast<int>(col_ind.size(0));
    const int cols = (nnz > 0) ? (col_ind.max().item<int>() + 1) : 0;

    TORCH_CHECK(rows > 0, "无效矩阵形状：rows 必须 > 0");
    TORCH_CHECK(cols > 0, "无效矩阵形状：cols 必须 > 0");

    MatchStats host_stats{};
    host_stats.total_rows = rows;
    host_stats.total_cols = cols;
    host_stats.total_nnz = nnz;
    host_stats.sparsity =
        1.0 - static_cast<double>(host_stats.total_nnz) / (static_cast<double>(rows) * cols);
    host_stats.num_row_panels = (rows + window_size - 1) / window_size;

    MatchStats* device_stats = nullptr;
    int* workspace_temp = nullptr;
    uint16_t* workspace_masks = nullptr;
    uint8_t* workspace_matched = nullptr;
    Pair1x2Device* workspace_pairs = nullptr;
    uint8_t* workspace_pair_used = nullptr;
    int* workspace_hash_keys = nullptr;
    unsigned int* workspace_hash_masks = nullptr;

    cudaError_t err;

    auto free_device_buffers = [&]() {
        if (workspace_hash_masks != nullptr) {
            cudaFree(workspace_hash_masks);
            workspace_hash_masks = nullptr;
        }
        if (workspace_hash_keys != nullptr) {
            cudaFree(workspace_hash_keys);
            workspace_hash_keys = nullptr;
        }
        if (workspace_pair_used != nullptr) {
            cudaFree(workspace_pair_used);
            workspace_pair_used = nullptr;
        }
        if (workspace_pairs != nullptr) {
            cudaFree(workspace_pairs);
            workspace_pairs = nullptr;
        }
        if (workspace_matched != nullptr) {
            cudaFree(workspace_matched);
            workspace_matched = nullptr;
        }
        if (workspace_masks != nullptr) {
            cudaFree(workspace_masks);
            workspace_masks = nullptr;
        }
        if (workspace_temp != nullptr) {
            cudaFree(workspace_temp);
            workspace_temp = nullptr;
        }
        if (device_stats != nullptr) {
            cudaFree(device_stats);
            device_stats = nullptr;
        }
    };

    auto alloc_device_buffer = [&](void** ptr, size_t bytes, const char* name) {
        err = cudaMalloc(ptr, bytes);
        if (err != cudaSuccess) {
            free_device_buffers();
            TORCH_CHECK(false, "cudaMalloc ", name, " 失败: ", cudaGetErrorString(err));
        }
    };

    err = cudaMalloc(&device_stats, sizeof(MatchStats));
    TORCH_CHECK(err == cudaSuccess, "cudaMalloc device_stats 失败: ", cudaGetErrorString(err));

    err = cudaMemcpy(device_stats, &host_stats, sizeof(MatchStats), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        free_device_buffers();
        TORCH_CHECK(false, "cudaMemcpy host->device 失败: ", cudaGetErrorString(err));
    }

    const size_t workspace_elems = static_cast<size_t>(nnz);
    alloc_device_buffer(reinterpret_cast<void**>(&workspace_temp),
                        workspace_elems * sizeof(int),
                        "workspace_temp");
    alloc_device_buffer(reinterpret_cast<void**>(&workspace_masks),
                        workspace_elems * sizeof(uint16_t),
                        "workspace_masks");
    alloc_device_buffer(reinterpret_cast<void**>(&workspace_matched),
                        workspace_elems * sizeof(uint8_t),
                        "workspace_matched");
    alloc_device_buffer(reinterpret_cast<void**>(&workspace_pairs),
                        workspace_elems * sizeof(Pair1x2Device),
                        "workspace_pairs");
    alloc_device_buffer(reinterpret_cast<void**>(&workspace_pair_used),
                        workspace_elems * sizeof(uint8_t),
                        "workspace_pair_used");
    alloc_device_buffer(reinterpret_cast<void**>(&workspace_hash_keys),
                        (workspace_elems * 2) * sizeof(int),
                        "workspace_hash_keys");
    alloc_device_buffer(reinterpret_cast<void**>(&workspace_hash_masks),
                        (workspace_elems * 2) * sizeof(unsigned int),
                        "workspace_hash_masks");

    const int num_windows = host_stats.num_row_panels;
    match_2to4_kernel<<<num_windows, kBlockThreads>>>(
        row_ptr.data_ptr<int>(),
        col_ind.data_ptr<int>(),
        rows,
        window_size,
        workspace_temp,
        workspace_masks,
        workspace_matched,
        workspace_pairs,
        workspace_pair_used,
        workspace_hash_keys,
        workspace_hash_masks,
        device_stats);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        free_device_buffers();
        TORCH_CHECK(false, "match_2to4_kernel 启动失败: ", cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        free_device_buffers();
        TORCH_CHECK(false, "match_2to4_kernel 执行失败: ", cudaGetErrorString(err));
    }

    err = cudaMemcpy(&host_stats, device_stats, sizeof(MatchStats), cudaMemcpyDeviceToHost);
    free_device_buffers();
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpy device->host 失败: ", cudaGetErrorString(err));

    std::printf("\n");
    std::printf("╔══════════════════════════════════════════════╗\n");
    std::printf("║     2:4 结构化稀疏匹配统计 (CUDA 层)         ║\n");
    std::printf("╠══════════════════════════════════════════════╣\n");
    std::printf("║  矩阵形状:          %6d × %-6d          ║\n", host_stats.total_rows, host_stats.total_cols);
    std::printf("║  总非零元数:        %-14lld            ║\n", (long long)host_stats.total_nnz);
    std::printf("║  稀疏度 (Sparsity):   %8.4f%%             ║\n", host_stats.sparsity * 100.0);
    std::printf("╠══════════════════════════════════════════════╣\n");
    std::printf("║  [逻辑层]                                    ║\n");
    std::printf("║  16行 Row Panels:  %-8d                 ║\n", host_stats.num_row_panels);
    std::printf("║  16×4 基础 Group:  %-8d                 ║\n", host_stats.num_groups);
    std::printf("╠══════════════════════════════════════════════╣\n");
    std::printf("║  [硬件层 — m16n8k16 指令映射]               ║\n");
    std::printf("║  16×16 稀疏块 (Block): %-8d             ║\n", host_stats.num_16x16_blocks);
    std::printf("║  Block 对齐填充 Group:  %-8d             ║\n", host_stats.block_padding_groups);
    std::printf("╠══════════════════════════════════════════════╣\n");
    std::printf("║  [填充统计]                                  ║\n");
    std::printf("║  细粒度 Fallback:   %-8d (单列0)         ║\n", host_stats.fine_fb);
    std::printf("║  粗粒度 Fallback:   %-8d (整列0)         ║\n", host_stats.coarse_fb);
    std::printf("║  元素级假0填充:     %-8d                ║\n", host_stats.fake_zeros);
    std::printf("╚══════════════════════════════════════════════╝\n\n");

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
// 模块定义
// ============================================================================
// 向 Python 暴露接口：
//   matching_utils_cuda.match_2to4(row_ptr, col_ind, values, window_size=16)
//
// 模块名称：TORCH_EXTENSION_NAME 在 setup_cuda.py 中定义为 "matching_utils_cuda"
// 输入：CSR 格式的 CUDA 张量（int32 + float32）
// 输出：匹配统计信息字典（字段与 block_match_windows.cpp 完全一致）
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "2:4 结构化稀疏匹配的 CUDA baseline 实现";

    m.def("match_2to4",
          &match_2to4_cuda,
          py::arg("row_ptr"),
          py::arg("col_ind"),
          py::arg("values"),
          py::arg("window_size") = 16,
          "在 CUDA 张量上运行 2:4 结构化稀疏匹配");
}
