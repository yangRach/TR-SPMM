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
#include <vector>

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
 * kMinBlockThreads   轻负载窗口的线程数（= 64，2 个 Warp）。
 * kMaxBlockThreads   重负载窗口的线程数（= 128，4 个 Warp）。
 *                    Host 端会按平均窗口 nnz 自适应选择 64/128 线程版本，
 *                    以平衡同步开销和并行搜索吞吐。
 * kSearchWindow      Step2（1:2 配对）中，每个列向后搜索伙伴的最大范围。
 *                    与 CPU 版的 SEARCH_WINDOW=32 不同——GPU 搜索更大窗口收益更高，
 *                    因为并行搜索的成本不随窗口增大而显著增加。
 * kGroupSearchWindow Step3（2:4 分组）中，每个 Pair 向后搜索伙伴的最大范围。
 *                    与 CPU 版的 GROUP_SEARCH_WINDOW=1024 一致。
 */
constexpr int kMinBlockThreads = 64;
constexpr int kMaxBlockThreads = 1024;
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

__device__ __forceinline__ void update_top2_pair(
    int cand_score,
    int cand_index,
    int& best_score,
    int& best_index,
    int& second_score,
    int& second_index)
{
    if (cand_index == -1) {
        return;
    }

    if (best_index == -1 ||
        cand_score < best_score ||
        (cand_score == best_score && cand_index < best_index)) {
        if (cand_index != best_index) {
            second_score = best_score;
            second_index = best_index;
        }
        best_score = cand_score;
        best_index = cand_index;
        return;
    }

    if (cand_index == best_index) {
        return;
    }

    if (second_index == -1 ||
        cand_score < second_score ||
        (cand_score == second_score && cand_index < second_index)) {
        second_score = cand_score;
        second_index = cand_index;
    }
}

__device__ __forceinline__ bool try_claim_u8(uint8_t* flags, int idx) {
    const int word_index = idx >> 2;
    const int byte_offset = (idx & 3) << 3;
    unsigned int* word_ptr = reinterpret_cast<unsigned int*>(flags) + word_index;
    while (true) {
        const unsigned int old_word = *word_ptr;
        const unsigned int old_byte = (old_word >> byte_offset) & 0xffu;
        if (old_byte != 0u) {
            return false;
        }
        const unsigned int new_word = old_word | (1u << byte_offset);
        const unsigned int prev = atomicCAS(word_ptr, old_word, new_word);
        if (prev == old_word) {
            return true;
        }
    }
}

__device__ __forceinline__ void release_u8(uint8_t* flags, int idx) {
    const int word_index = idx >> 2;
    const int byte_offset = (idx & 3) << 3;
    unsigned int* word_ptr = reinterpret_cast<unsigned int*>(flags) + word_index;
    const unsigned int clear_mask = ~(1u << byte_offset);
    while (true) {
        const unsigned int old_word = *word_ptr;
        const unsigned int new_word = old_word & clear_mask;
        const unsigned int prev = atomicCAS(word_ptr, old_word, new_word);
        if (prev == old_word) {
            return;
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
//   CUDA: match_2to4_kernel<<<batch_size, kBlockThreads>>>(...)
//
// 每个 CUDA Block 并行处理一个 Row Panel（16 行窗口），
// Block 内 64 或 128 个线程协作完成 Step0~Step4 的匹配与统计。
//
// Kernel 参数说明：
//   row_ptr, col_ind    — CSR 格式（CUDA 张量指针）
//   rows, window_size   — 矩阵行数和窗口大小
//   window_ids          — 当前批次要处理的窗口 id 列表
//   workspace_*         — 主机侧预分配的全局工作区（避免设备端 new/delete）
//   stats               — 输出统计信息（AtomicAdd 汇总）
// ============================================================================

template <int kBlockThreads>
__global__ void match_2to4_kernel(
    const int* row_ptr,
    const int* col_ind,
    int rows,
    int window_size,
    const int* window_ids,
    int* workspace_temp,
    uint16_t* workspace_masks,
    uint8_t* workspace_matched,
    Pair1x2Device* workspace_pairs,
    uint8_t* workspace_pair_used,
    int* workspace_hash_keys,
    unsigned int* workspace_hash_masks,
    MatchStats* stats)
{
    static_assert(kBlockThreads == kMinBlockThreads || kBlockThreads == kMaxBlockThreads,
                  "match_2to4_kernel 仅支持 64 或 128 线程");
    constexpr int kNumWarps = kBlockThreads / kWarpSize;

    // ======================================================================
    // 窗口定位：与 CPU 版本 run_2to4_matching() 的逐窗口循环对应
    //   CPU: for (int win = 0; win < num_windows; ++win)
    //   CUDA: win_id = window_ids[blockIdx.x] （每个 Block 处理一个指定窗口）
    //
    // workspace_offset = row_ptr[row_start]（该窗口第一个非零元的全局索引）
    // 用于在预分配的工作区数组中定位该窗口的独立切片。
    // ======================================================================
    const int win_id = window_ids[blockIdx.x];
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
    __shared__ int warp_best_score[kNumWarps];  // Warp 归约缓冲区
    __shared__ int warp_best_index[kNumWarps];  // Warp 归约缓冲区
    __shared__ int shared_warp_sums[kNumWarps];
    __shared__ int shared_warp_offsets[kNumWarps];

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
        int w = (lane0 < kNumWarps) ? shared_warp_sums[lane0] : 0;
        for (int offset = 1; offset < kWarpSize; offset <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, w, offset);
            if (lane0 >= offset) {
                w += y;
            }
        }
        if (lane0 < kNumWarps) {
            shared_warp_offsets[lane0] = w - shared_warp_sums[lane0];
        }
        if (lane0 == kNumWarps - 1) {
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
    const int T_MAX = 12;
    const int SEARCH_WINDOW = (num_cols > 16384) ? 64 : (num_cols > 4096) ? 128 : kSearchWindow;
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;

    __shared__ int shared_next_i;
    __shared__ int shared_fine_fb;
    if (threadIdx.x == 0) {
        shared_num_pairs = 0;
        shared_next_i = 0;
        shared_fine_fb = 0;
    }
    __syncthreads();

    while (true) {
        int i = 0;
        if (lane == 0) {
            i = atomicAdd(&shared_next_i, 1);
        }
        i = __shfl_sync(0xffffffffu, i, 0);
        if (i >= num_cols) {
            break;
        }
        if (matched[i]) {
            continue;
        }

        const uint16_t mask_i = masks[i];
        int best_score = 1 << 30;
        int best_j = -1;
        int second_score = 1 << 30;
        int second_j = -1;
        const int j_end = min(num_cols, i + 1 + SEARCH_WINDOW);
        for (int j = i + 1 + lane; j < j_end; j += kWarpSize) {
            if (matched[j]) {
                continue;
            }
            const int overlap = popcount16(static_cast<uint16_t>(mask_i & masks[j]));
            if (overlap > T_MAX) {
                continue;
            }
            const int score = overlap * 64 + (16 - temp_storage[j]);
            update_top2_pair(score, j, best_score, best_j, second_score, second_j);
        }
        warp_best_pair(best_score, best_j);
        best_score = __shfl_sync(0xffffffffu, best_score, 0);
        best_j = __shfl_sync(0xffffffffu, best_j, 0);

        int retry_score = best_score;
        int retry_j = best_j;
        if (retry_j == best_j) {
            retry_score = second_score;
            retry_j = second_j;
        }
        warp_best_pair(retry_score, retry_j);
        retry_j = __shfl_sync(0xffffffffu, retry_j, 0);

        if (lane == 0) {
            const int gi = workspace_offset + i;
            if (!try_claim_u8(workspace_matched, gi)) {
                continue;
            }

            int final_j = -1;
            if (best_j != -1) {
                const int gj = workspace_offset + best_j;
                if (try_claim_u8(workspace_matched, gj)) {
                    final_j = best_j;
                }
            }
            if (final_j == -1 && retry_j != -1) {
                const int gj = workspace_offset + retry_j;
                if (try_claim_u8(workspace_matched, gj)) {
                    final_j = retry_j;
                }
            }

            const int pair_idx = atomicAdd(&shared_num_pairs, 1);
            Pair1x2Device& p = pairs[pair_idx];
            p.mask1 = mask_i;
            if (final_j != -1) {
                p.mask2 = masks[final_j];
                p.G1 = static_cast<uint16_t>(mask_i & masks[final_j]);
                p.G2 = static_cast<uint16_t>(mask_i | masks[final_j]);
                p.nnz = static_cast<uint8_t>(temp_storage[i] + temp_storage[final_j]);
            } else {
                p.mask2 = 0;
                p.G1 = 0;
                p.G2 = mask_i;
                atomicAdd(&shared_fine_fb, 1);
                p.nnz = static_cast<uint8_t>(temp_storage[i]);
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        local_fine_fb = shared_fine_fb;
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
    const int GROUP_SEARCH_WINDOW = (num_pairs > 8192) ? 128 : (num_pairs > 2048) ? 256 : kGroupSearchWindow;

    __shared__ int shared_next_pair_i;
    __shared__ int shared_num_groups_local;
    __shared__ int shared_coarse_fb;
    __shared__ int shared_fake_zeros;
    if (threadIdx.x == 0) {
        shared_next_pair_i = 0;
        shared_num_groups_local = 0;
        shared_coarse_fb = 0;
        shared_fake_zeros = 0;
    }
    __syncthreads();

    while (true) {
        int i = 0;
        if (lane == 0) {
            i = atomicAdd(&shared_next_pair_i, 1);
        }
        i = __shfl_sync(0xffffffffu, i, 0);
        if (i >= num_pairs) {
            break;
        }
        if (pair_used[i]) {
            continue;
        }

        const Pair1x2Device pa = pairs[i];
        int best_group_score = 1 << 30;
        int best_group_j = -1;
        int second_group_score = 1 << 30;
        int second_group_j = -1;
        const int j_end = min(num_pairs, i + 1 + GROUP_SEARCH_WINDOW);
        for (int j = i + 1 + lane; j < j_end; j += kWarpSize) {
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
            update_top2_pair(
                score, j, best_group_score, best_group_j, second_group_score, second_group_j);
        }
        warp_best_pair(best_group_score, best_group_j);
        best_group_score = __shfl_sync(0xffffffffu, best_group_score, 0);
        best_group_j = __shfl_sync(0xffffffffu, best_group_j, 0);

        int retry_group_score = best_group_score;
        int retry_group_j = best_group_j;
        if (retry_group_j == best_group_j) {
            retry_group_score = second_group_score;
            retry_group_j = second_group_j;
        }
        warp_best_pair(retry_group_score, retry_group_j);
        retry_group_j = __shfl_sync(0xffffffffu, retry_group_j, 0);

        if (lane == 0) {
            const int gi = workspace_offset + i;
            if (!try_claim_u8(workspace_pair_used, gi)) {
                continue;
            }

            int selected_j = -1;
            if (best_group_j != -1) {
                const int gj = workspace_offset + best_group_j;
                if (try_claim_u8(workspace_pair_used, gj)) {
                    selected_j = best_group_j;
                }
            }
            if (selected_j == -1 && retry_group_j != -1) {
                const int gj = workspace_offset + retry_group_j;
                if (try_claim_u8(workspace_pair_used, gj)) {
                    selected_j = retry_group_j;
                }
            }

            atomicAdd(&shared_num_groups_local, 1);
            if (selected_j != -1) {
                const Pair1x2Device pb = pairs[selected_j];
                atomicAdd(&shared_fake_zeros,
                          fake_zero_cost(pa.mask1, pa.mask2, pb.mask1, pb.mask2, win_rows));
            } else {
                atomicAdd(&shared_coarse_fb, 2);
                atomicAdd(&shared_fake_zeros, fake_zero_cost(pa.mask1, pa.mask2, 0, 0, win_rows));
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        num_groups_local = shared_num_groups_local;
        local_coarse_fb = shared_coarse_fb;
        local_fake_zeros = shared_fake_zeros;
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

template <int kBlockThreads>
__global__ void build_window_data_kernel(
    const int* row_ptr,
    const int* col_ind,
    int rows,
    int window_size,
    const int* window_ids,
    int* workspace_temp,
    uint16_t* workspace_masks,
    uint8_t* workspace_matched,
    int* workspace_hash_keys,
    unsigned int* workspace_hash_masks,
    int* window_num_cols)
{
    constexpr int kNumWarps = kBlockThreads / kWarpSize;

    const int win_id = window_ids[blockIdx.x];
    const int row_start = win_id * window_size;
    if (row_start >= rows) {
        return;
    }

    const int row_end = min(row_start + window_size, rows);
    const int workspace_offset = row_ptr[row_start];
    const int window_nnz = row_ptr[row_end] - row_ptr[row_start];
    if (window_nnz <= 0) {
        if (threadIdx.x == 0) {
            window_num_cols[win_id] = 0;
        }
        return;
    }

    int* temp_storage = workspace_temp + workspace_offset;
    uint16_t* masks = workspace_masks + workspace_offset;
    uint8_t* matched = workspace_matched + workspace_offset;
    const int hash_offset = workspace_offset << 1;
    const int hash_size = window_nnz << 1;
    int* hash_keys = workspace_hash_keys + hash_offset;
    unsigned int* hash_masks = workspace_hash_masks + hash_offset;

    __shared__ int shared_num_cols;
    __shared__ int shared_warp_sums[kNumWarps];
    __shared__ int shared_warp_offsets[kNumWarps];

    for (int i = threadIdx.x; i < hash_size; i += blockDim.x) {
        hash_keys[i] = -1;
        hash_masks[i] = 0u;
    }
    __syncthreads();

    for (int r = row_start + threadIdx.x; r < row_end; r += blockDim.x) {
        const int local_row = r - row_start;
        const unsigned int bit = 1u << local_row;
        for (int e = row_ptr[r]; e < row_ptr[r + 1]; ++e) {
            const int c = col_ind[e];
            unsigned int h =
                (static_cast<unsigned int>(c) * 2654435761u) % static_cast<unsigned int>(hash_size);
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

    const int warp_id = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x & (kWarpSize - 1);
    if (lane == kWarpSize - 1) {
        shared_warp_sums[warp_id] = x;
    }
    __syncthreads();

    if (warp_id == 0) {
        int w = (lane < kNumWarps) ? shared_warp_sums[lane] : 0;
        for (int offset = 1; offset < kWarpSize; offset <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, w, offset);
            if (lane >= offset) {
                w += y;
            }
        }
        if (lane < kNumWarps) {
            shared_warp_offsets[lane] = w - shared_warp_sums[lane];
        }
        if (lane == kNumWarps - 1) {
            shared_num_cols = w;
        }
    }
    __syncthreads();

    const int prefix = shared_warp_offsets[warp_id] + (x - local_count);
    int out = prefix;
    for (int i = threadIdx.x; i < hash_size; i += blockDim.x) {
        if (hash_keys[i] != -1) {
            masks[out++] = static_cast<uint16_t>(hash_masks[i]);
        }
    }
    __syncthreads();

    const int num_cols = shared_num_cols;
    for (int i = threadIdx.x; i < num_cols; i += blockDim.x) {
        matched[i] = 0;
        temp_storage[i] = popcount16(masks[i]);
    }
    if (threadIdx.x == 0) {
        window_num_cols[win_id] = num_cols;
    }
}

template <int kBlockThreads>
__global__ void match_pairs_multi_block_kernel(
    const int* row_ptr,
    int rows,
    int window_size,
    const int* window_ids,
    int blocks_per_window,
    const int* window_num_cols,
    int* window_next_i,
    int* window_num_pairs,
    int* window_fine_fb,
    int* workspace_temp,
    uint16_t* workspace_masks,
    uint8_t* workspace_matched,
    Pair1x2Device* workspace_pairs)
{
    const int logical_block = blockIdx.x / blocks_per_window;
    const int win_id = window_ids[logical_block];
    const int row_start = win_id * window_size;
    if (row_start >= rows) {
        return;
    }

    const int workspace_offset = row_ptr[row_start];
    const int num_cols = window_num_cols[win_id];
    if (num_cols <= 0) {
        return;
    }

    int* temp_storage = workspace_temp + workspace_offset;
    uint16_t* masks = workspace_masks + workspace_offset;
    Pair1x2Device* pairs = workspace_pairs + workspace_offset;

    const int T_MAX = 12;
    const int SEARCH_WINDOW = (num_cols > 16384) ? 64 : (num_cols > 4096) ? 128 : kSearchWindow;
    const int lane = threadIdx.x % kWarpSize;

    while (true) {
        int i = 0;
        if (lane == 0) {
            i = atomicAdd(&window_next_i[win_id], 1);
        }
        i = __shfl_sync(0xffffffffu, i, 0);
        if (i >= num_cols) {
            break;
        }

        const int gi = workspace_offset + i;
        if (workspace_matched[gi]) {
            continue;
        }

        const uint16_t mask_i = masks[i];
        int best_score = 1 << 30;
        int best_j = -1;
        int second_score = 1 << 30;
        int second_j = -1;
        const int j_end = min(num_cols, i + 1 + SEARCH_WINDOW);
        for (int j = i + 1 + lane; j < j_end; j += kWarpSize) {
            if (workspace_matched[workspace_offset + j]) {
                continue;
            }
            const int overlap = popcount16(static_cast<uint16_t>(mask_i & masks[j]));
            if (overlap > T_MAX) {
                continue;
            }
            const int score = overlap * 64 + (16 - temp_storage[j]);
            update_top2_pair(score, j, best_score, best_j, second_score, second_j);
        }
        warp_best_pair(best_score, best_j);
        best_j = __shfl_sync(0xffffffffu, best_j, 0);

        int retry_score = second_score;
        int retry_j = second_j;
        warp_best_pair(retry_score, retry_j);
        retry_j = __shfl_sync(0xffffffffu, retry_j, 0);

        if (lane == 0) {
            if (!try_claim_u8(workspace_matched, gi)) {
                continue;
            }

            int final_j = -1;
            if (best_j != -1) {
                const int gj = workspace_offset + best_j;
                if (try_claim_u8(workspace_matched, gj)) {
                    final_j = best_j;
                }
            }
            if (final_j == -1 && retry_j != -1) {
                const int gj = workspace_offset + retry_j;
                if (try_claim_u8(workspace_matched, gj)) {
                    final_j = retry_j;
                }
            }

            const int pair_idx = atomicAdd(&window_num_pairs[win_id], 1);
            Pair1x2Device& p = pairs[pair_idx];
            p.mask1 = mask_i;
            if (final_j != -1) {
                p.mask2 = masks[final_j];
                p.G1 = static_cast<uint16_t>(mask_i & masks[final_j]);
                p.G2 = static_cast<uint16_t>(mask_i | masks[final_j]);
                p.nnz = static_cast<uint8_t>(temp_storage[i] + temp_storage[final_j]);
            } else {
                p.mask2 = 0;
                p.G1 = 0;
                p.G2 = mask_i;
                atomicAdd(&window_fine_fb[win_id], 1);
                p.nnz = static_cast<uint8_t>(temp_storage[i]);
            }
        }
    }
}

template <int kBlockThreads>
__global__ void init_pair_used_kernel(
    const int* row_ptr,
    int rows,
    int window_size,
    const int* window_ids,
    const int* window_num_pairs,
    uint8_t* workspace_pair_used)
{
    const int win_id = window_ids[blockIdx.x];
    const int row_start = win_id * window_size;
    if (row_start >= rows) {
        return;
    }
    const int workspace_offset = row_ptr[row_start];
    const int num_pairs = window_num_pairs[win_id];
    uint8_t* pair_used = workspace_pair_used + workspace_offset;
    for (int i = threadIdx.x; i < num_pairs; i += blockDim.x) {
        pair_used[i] = 0;
    }
}

template <int kBlockThreads>
__global__ void match_groups_multi_block_kernel(
    const int* row_ptr,
    int rows,
    int window_size,
    const int* window_ids,
    int blocks_per_window,
    const int* window_num_pairs,
    int* window_next_pair_i,
    int* window_num_groups,
    int* window_coarse_fb,
    int* window_fake_zeros,
    Pair1x2Device* workspace_pairs,
    uint8_t* workspace_pair_used)
{
    const int logical_block = blockIdx.x / blocks_per_window;
    const int win_id = window_ids[logical_block];
    const int row_start = win_id * window_size;
    if (row_start >= rows) {
        return;
    }

    const int row_end = min(row_start + window_size, rows);
    const int win_rows = row_end - row_start;
    const int workspace_offset = row_ptr[row_start];
    const int num_pairs = window_num_pairs[win_id];
    if (num_pairs <= 0) {
        return;
    }

    Pair1x2Device* pairs = workspace_pairs + workspace_offset;
    const int GROUP_SEARCH_WINDOW =
        (num_pairs > 8192) ? 128 : (num_pairs > 2048) ? 256 : kGroupSearchWindow;
    const int lane = threadIdx.x % kWarpSize;

    while (true) {
        int i = 0;
        if (lane == 0) {
            i = atomicAdd(&window_next_pair_i[win_id], 1);
        }
        i = __shfl_sync(0xffffffffu, i, 0);
        if (i >= num_pairs) {
            break;
        }

        const int gi = workspace_offset + i;
        if (workspace_pair_used[gi]) {
            continue;
        }

        const Pair1x2Device pa = pairs[i];
        int best_group_score = 1 << 30;
        int best_group_j = -1;
        int second_group_score = 1 << 30;
        int second_group_j = -1;
        const int j_end = min(num_pairs, i + 1 + GROUP_SEARCH_WINDOW);
        for (int j = i + 1 + lane; j < j_end; j += kWarpSize) {
            if (workspace_pair_used[workspace_offset + j]) {
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
            update_top2_pair(
                score, j, best_group_score, best_group_j, second_group_score, second_group_j);
        }
        warp_best_pair(best_group_score, best_group_j);
        best_group_j = __shfl_sync(0xffffffffu, best_group_j, 0);

        int retry_group_score = second_group_score;
        int retry_group_j = second_group_j;
        warp_best_pair(retry_group_score, retry_group_j);
        retry_group_j = __shfl_sync(0xffffffffu, retry_group_j, 0);

        if (lane == 0) {
            if (!try_claim_u8(workspace_pair_used, gi)) {
                continue;
            }

            int selected_j = -1;
            if (best_group_j != -1) {
                const int gj = workspace_offset + best_group_j;
                if (try_claim_u8(workspace_pair_used, gj)) {
                    selected_j = best_group_j;
                }
            }
            if (selected_j == -1 && retry_group_j != -1) {
                const int gj = workspace_offset + retry_group_j;
                if (try_claim_u8(workspace_pair_used, gj)) {
                    selected_j = retry_group_j;
                }
            }

            atomicAdd(&window_num_groups[win_id], 1);
            if (selected_j != -1) {
                const Pair1x2Device pb = pairs[selected_j];
                atomicAdd(&window_fake_zeros[win_id],
                          fake_zero_cost(pa.mask1, pa.mask2, pb.mask1, pb.mask2, win_rows));
            } else {
                atomicAdd(&window_coarse_fb[win_id], 2);
                atomicAdd(&window_fake_zeros[win_id], fake_zero_cost(pa.mask1, pa.mask2, 0, 0, win_rows));
            }
        }
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
//   3. 按窗口 nnz 将窗口分成轻/重批次，分别使用 64/128 线程 kernel
//   4. 同步并回收结果统计信息
//   5. 输出格式化统计表格，返回 Python 字典
//
// 工作区设计：
//   所有工作区（temp, masks, matched, pairs, pair_used, hash_keys, hash_masks）
//   在 host 端一次性分配，kernel 内通过 workspace_offset 切分使用。
//   当前实现使用 PyTorch CUDA tensor 作为底层存储，直接复用 caching allocator，
//   避免反复 cudaMalloc/cudaFree 带来的分配抖动和同步开销。
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

    cudaError_t err;
    const auto device = row_ptr.device();
    const int64_t workspace_elems = static_cast<int64_t>(nnz);
    const int64_t hash_elems = workspace_elems * 2;

    const auto int_options = torch::TensorOptions().device(device).dtype(torch::kInt32);
    const auto uint8_options = torch::TensorOptions().device(device).dtype(torch::kUInt8);
    const auto int16_options = torch::TensorOptions().device(device).dtype(torch::kInt16);

    // 使用 CUDA tensor 承载 workspace，底层由 PyTorch caching allocator 管理。
    torch::Tensor workspace_temp_tensor = torch::empty({workspace_elems}, int_options);
    torch::Tensor workspace_masks_tensor = torch::empty({workspace_elems}, int16_options);
    torch::Tensor workspace_matched_tensor = torch::empty({workspace_elems}, uint8_options);
    torch::Tensor workspace_pairs_tensor =
        torch::empty({workspace_elems * static_cast<int64_t>(sizeof(Pair1x2Device))}, uint8_options);
    torch::Tensor workspace_pair_used_tensor = torch::empty({workspace_elems}, uint8_options);
    torch::Tensor workspace_hash_keys_tensor = torch::empty({hash_elems}, int_options);
    torch::Tensor workspace_hash_masks_tensor = torch::empty({hash_elems}, int_options);
    int* workspace_temp = workspace_temp_tensor.data_ptr<int>();
    uint16_t* workspace_masks =
        reinterpret_cast<uint16_t*>(workspace_masks_tensor.data_ptr<int16_t>());
    uint8_t* workspace_matched = workspace_matched_tensor.data_ptr<uint8_t>();
    Pair1x2Device* workspace_pairs =
        reinterpret_cast<Pair1x2Device*>(workspace_pairs_tensor.data_ptr<uint8_t>());
    uint8_t* workspace_pair_used = workspace_pair_used_tensor.data_ptr<uint8_t>();
    int* workspace_hash_keys = workspace_hash_keys_tensor.data_ptr<int>();
    unsigned int* workspace_hash_masks =
        reinterpret_cast<unsigned int*>(workspace_hash_masks_tensor.data_ptr<int>());

    const int num_windows = host_stats.num_row_panels;
    constexpr int kLightWindowNnzThreshold = 512;
    constexpr int kGiantWindowNnzThreshold = 4096;
    constexpr int kGiantBlocksPerWindow = 4;

    std::vector<int> host_row_ptr(static_cast<size_t>(rows) + 1);
    err = cudaMemcpy(host_row_ptr.data(),
                     row_ptr.data_ptr<int>(),
                     static_cast<size_t>(rows + 1) * sizeof(int),
                     cudaMemcpyDeviceToHost);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpy row_ptr device->host 失败: ", cudaGetErrorString(err));

    std::vector<int> light_window_ids;
    std::vector<int> heavy_window_ids;
    std::vector<int> giant_window_ids;
    light_window_ids.reserve(num_windows);
    heavy_window_ids.reserve(num_windows);
    giant_window_ids.reserve(num_windows);

    for (int win_id = 0; win_id < num_windows; ++win_id) {
        const int row_start = win_id * window_size;
        const int row_end = std::min(row_start + window_size, rows);
        const int window_nnz = host_row_ptr[row_end] - host_row_ptr[row_start];
        if (window_nnz <= 0) {
            continue;
        }
        if (window_nnz <= kLightWindowNnzThreshold) {
            light_window_ids.push_back(win_id);
        } else if (window_nnz <= kGiantWindowNnzThreshold) {
            heavy_window_ids.push_back(win_id);
        } else {
            giant_window_ids.push_back(win_id);
        }
    }

    auto copy_window_ids_to_device = [&](const std::vector<int>& host_ids) {
        torch::Tensor device_ids =
            torch::empty({static_cast<int64_t>(host_ids.size())}, int_options);
        if (!host_ids.empty()) {
            cudaError_t copy_err = cudaMemcpy(device_ids.data_ptr<int>(),
                                              host_ids.data(),
                                              host_ids.size() * sizeof(int),
                                              cudaMemcpyHostToDevice);
            TORCH_CHECK(copy_err == cudaSuccess,
                        "cudaMemcpy window_ids host->device 失败: ",
                        cudaGetErrorString(copy_err));
        }
        return device_ids;
    };

    torch::Tensor light_window_ids_tensor;
    torch::Tensor heavy_window_ids_tensor;
    torch::Tensor giant_window_ids_tensor;

    torch::Tensor window_num_cols_tensor = torch::zeros({num_windows}, int_options);
    torch::Tensor window_next_i_tensor = torch::zeros({num_windows}, int_options);
    torch::Tensor window_num_pairs_tensor = torch::zeros({num_windows}, int_options);
    torch::Tensor window_fine_fb_tensor = torch::zeros({num_windows}, int_options);
    torch::Tensor window_next_pair_i_tensor = torch::zeros({num_windows}, int_options);
    torch::Tensor window_num_groups_tensor = torch::zeros({num_windows}, int_options);
    torch::Tensor window_coarse_fb_tensor = torch::zeros({num_windows}, int_options);
    torch::Tensor window_fake_zeros_tensor = torch::zeros({num_windows}, int_options);

    int* window_num_cols = window_num_cols_tensor.data_ptr<int>();
    int* window_next_i = window_next_i_tensor.data_ptr<int>();
    int* window_num_pairs = window_num_pairs_tensor.data_ptr<int>();
    int* window_fine_fb = window_fine_fb_tensor.data_ptr<int>();
    int* window_next_pair_i = window_next_pair_i_tensor.data_ptr<int>();
    int* window_num_groups = window_num_groups_tensor.data_ptr<int>();
    int* window_coarse_fb = window_coarse_fb_tensor.data_ptr<int>();
    int* window_fake_zeros = window_fake_zeros_tensor.data_ptr<int>();

    auto launch_build = [&](torch::Tensor& ids_tensor, int host_count, int threads) {
        if (host_count <= 0) {
            return;
        }
        if (threads == kMinBlockThreads) {
            build_window_data_kernel<kMinBlockThreads><<<host_count, kMinBlockThreads>>>(
                row_ptr.data_ptr<int>(),
                col_ind.data_ptr<int>(),
                rows,
                window_size,
                ids_tensor.data_ptr<int>(),
                workspace_temp,
                workspace_masks,
                workspace_matched,
                workspace_hash_keys,
                workspace_hash_masks,
                window_num_cols);
        } else {
            build_window_data_kernel<kMaxBlockThreads><<<host_count, kMaxBlockThreads>>>(
                row_ptr.data_ptr<int>(),
                col_ind.data_ptr<int>(),
                rows,
                window_size,
                ids_tensor.data_ptr<int>(),
                workspace_temp,
                workspace_masks,
                workspace_matched,
                workspace_hash_keys,
                workspace_hash_masks,
                window_num_cols);
        }
    };

    auto launch_pairs = [&](torch::Tensor& ids_tensor, int host_count, int threads, int blocks_per_window) {
        if (host_count <= 0) {
            return;
        }
        const int grid = host_count * blocks_per_window;
        if (threads == kMinBlockThreads) {
            match_pairs_multi_block_kernel<kMinBlockThreads><<<grid, kMinBlockThreads>>>(
                row_ptr.data_ptr<int>(),
                rows,
                window_size,
                ids_tensor.data_ptr<int>(),
                blocks_per_window,
                window_num_cols,
                window_next_i,
                window_num_pairs,
                window_fine_fb,
                workspace_temp,
                workspace_masks,
                workspace_matched,
                workspace_pairs);
        } else {
            match_pairs_multi_block_kernel<kMaxBlockThreads><<<grid, kMaxBlockThreads>>>(
                row_ptr.data_ptr<int>(),
                rows,
                window_size,
                ids_tensor.data_ptr<int>(),
                blocks_per_window,
                window_num_cols,
                window_next_i,
                window_num_pairs,
                window_fine_fb,
                workspace_temp,
                workspace_masks,
                workspace_matched,
                workspace_pairs);
        }
    };

    auto launch_init_pair_used = [&](torch::Tensor& ids_tensor, int host_count, int threads) {
        if (host_count <= 0) {
            return;
        }
        if (threads == kMinBlockThreads) {
            init_pair_used_kernel<kMinBlockThreads><<<host_count, kMinBlockThreads>>>(
                row_ptr.data_ptr<int>(),
                rows,
                window_size,
                ids_tensor.data_ptr<int>(),
                window_num_pairs,
                workspace_pair_used);
        } else {
            init_pair_used_kernel<kMaxBlockThreads><<<host_count, kMaxBlockThreads>>>(
                row_ptr.data_ptr<int>(),
                rows,
                window_size,
                ids_tensor.data_ptr<int>(),
                window_num_pairs,
                workspace_pair_used);
        }
    };

    auto launch_groups = [&](torch::Tensor& ids_tensor, int host_count, int threads, int blocks_per_window) {
        if (host_count <= 0) {
            return;
        }
        const int grid = host_count * blocks_per_window;
        if (threads == kMinBlockThreads) {
            match_groups_multi_block_kernel<kMinBlockThreads><<<grid, kMinBlockThreads>>>(
                row_ptr.data_ptr<int>(),
                rows,
                window_size,
                ids_tensor.data_ptr<int>(),
                blocks_per_window,
                window_num_pairs,
                window_next_pair_i,
                window_num_groups,
                window_coarse_fb,
                window_fake_zeros,
                workspace_pairs,
                workspace_pair_used);
        } else {
            match_groups_multi_block_kernel<kMaxBlockThreads><<<grid, kMaxBlockThreads>>>(
                row_ptr.data_ptr<int>(),
                rows,
                window_size,
                ids_tensor.data_ptr<int>(),
                blocks_per_window,
                window_num_pairs,
                window_next_pair_i,
                window_num_groups,
                window_coarse_fb,
                window_fake_zeros,
                workspace_pairs,
                workspace_pair_used);
        }
    };

    if (!light_window_ids.empty()) {
        light_window_ids_tensor = copy_window_ids_to_device(light_window_ids);
        launch_build(light_window_ids_tensor, static_cast<int>(light_window_ids.size()), kMinBlockThreads);
    }
    if (!heavy_window_ids.empty()) {
        heavy_window_ids_tensor = copy_window_ids_to_device(heavy_window_ids);
        launch_build(heavy_window_ids_tensor, static_cast<int>(heavy_window_ids.size()), kMaxBlockThreads);
    }
    if (!giant_window_ids.empty()) {
        giant_window_ids_tensor = copy_window_ids_to_device(giant_window_ids);
        launch_build(giant_window_ids_tensor, static_cast<int>(giant_window_ids.size()), kMaxBlockThreads);
    }

    launch_pairs(light_window_ids_tensor, static_cast<int>(light_window_ids.size()), kMinBlockThreads, 1);
    launch_pairs(heavy_window_ids_tensor, static_cast<int>(heavy_window_ids.size()), kMaxBlockThreads, 1);
    launch_pairs(giant_window_ids_tensor, static_cast<int>(giant_window_ids.size()), kMaxBlockThreads,
                 kGiantBlocksPerWindow);

    launch_init_pair_used(light_window_ids_tensor, static_cast<int>(light_window_ids.size()), kMinBlockThreads);
    launch_init_pair_used(heavy_window_ids_tensor, static_cast<int>(heavy_window_ids.size()), kMaxBlockThreads);
    launch_init_pair_used(giant_window_ids_tensor, static_cast<int>(giant_window_ids.size()), kMaxBlockThreads);

    launch_groups(light_window_ids_tensor, static_cast<int>(light_window_ids.size()), kMinBlockThreads, 1);
    launch_groups(heavy_window_ids_tensor, static_cast<int>(heavy_window_ids.size()), kMaxBlockThreads, 1);
    launch_groups(giant_window_ids_tensor, static_cast<int>(giant_window_ids.size()), kMaxBlockThreads,
                  kGiantBlocksPerWindow);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "多阶段 match kernel 启动失败: ", cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "多阶段 match kernel 执行失败: ", cudaGetErrorString(err));
    }

    std::vector<int> host_num_groups(num_windows, 0);
    std::vector<int> host_fine_fb(num_windows, 0);
    std::vector<int> host_coarse_fb(num_windows, 0);
    std::vector<int> host_fake_zeros(num_windows, 0);
    err = cudaMemcpy(host_num_groups.data(),
                     window_num_groups,
                     static_cast<size_t>(num_windows) * sizeof(int),
                     cudaMemcpyDeviceToHost);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpy num_groups device->host 失败: ", cudaGetErrorString(err));
    err = cudaMemcpy(host_fine_fb.data(),
                     window_fine_fb,
                     static_cast<size_t>(num_windows) * sizeof(int),
                     cudaMemcpyDeviceToHost);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpy fine_fb device->host 失败: ", cudaGetErrorString(err));
    err = cudaMemcpy(host_coarse_fb.data(),
                     window_coarse_fb,
                     static_cast<size_t>(num_windows) * sizeof(int),
                     cudaMemcpyDeviceToHost);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpy coarse_fb device->host 失败: ", cudaGetErrorString(err));
    err = cudaMemcpy(host_fake_zeros.data(),
                     window_fake_zeros,
                     static_cast<size_t>(num_windows) * sizeof(int),
                     cudaMemcpyDeviceToHost);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpy fake_zeros device->host 失败: ", cudaGetErrorString(err));

    host_stats.num_groups = 0;
    host_stats.num_16x16_blocks = 0;
    host_stats.block_padding_groups = 0;
    host_stats.fine_fb = 0;
    host_stats.coarse_fb = 0;
    host_stats.fake_zeros = 0;
    for (int win_id = 0; win_id < num_windows; ++win_id) {
        const int local_groups = host_num_groups[win_id];
        const int local_blocks = (local_groups + 3) / 4;
        host_stats.num_groups += local_groups;
        host_stats.num_16x16_blocks += local_blocks;
        host_stats.block_padding_groups += local_blocks * 4 - local_groups;
        host_stats.fine_fb += host_fine_fb[win_id];
        host_stats.coarse_fb += host_coarse_fb[win_id];
        host_stats.fake_zeros += host_fake_zeros[win_id];
    }

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
