/**
 * block_matching_online.cu  (v2 — 修复版)
 *   基于 CUDA 的在线双路路由匹配算法 (Online Dual-Routing TC+SPTC Matching)
 *
 * v2 修复:
 *   1. uint16_t → uint32_t s_cols_mask, 消除未对齐 atomicOr
 *   2. O(1) fake_zeros 计算: 在 Phase 2/3 构建 Group 时直接累加
 *   3. MAX_COLS 512 → 2048 + 哈希防死循环保险丝
 *   4. 分离 fine_fb_count / coarse_fb_count; pair_used → Shared Memory
 *
 * 线程映射:
 *   1 个 Thread Block 处理 1 个 16 行的 Window.
 *   线程并行提取 CSR → Shared Memory, 匹配由 Thread 0 串行执行.
 *
 * 接口:
 *   matching_utils_online.match_2to4_online(
 *       row_ptr, col_ind, values, window_size=16,
 *       dense_threshold=8, t_max=8
 *   ) -> dict
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <vector>

namespace py = pybind11;

// ============================================================================
// 常量
// ============================================================================
#define WINDOW_SIZE         16
#define MAX_COLS            2048   // 16行极稠密场景, 从 512 扩充
#define MAX_PAIRS           (MAX_COLS / 2)
#define SEARCH_WINDOW       32     // 细粒度滑动视窗
#define GROUP_SEARCH_WINDOW 1024   // 粗粒度滑动视窗 (Warp 以 32 步长迭代)
#define DENSE_THRESHOLD     8      // TC 路由激活阈值 (16×8 块)
#define SMEM_STATS_SIZE     7      // out_stats 每窗口字段数

// ============================================================================
// 统计信息 (Host 端使用)
// ============================================================================
struct OnlineMatchStats {
    int num_sptc_groups    = 0;
    int num_tc_blocks      = 0;
    int fine_fb            = 0;   // Phase1 孤儿列数
    int coarse_fb_cols     = 0;   // Phase2 拆散列数
    int dense_to_tc_cols   = 0;   // Dense Pool → TC 路由的列数
    int dense_to_sptc_cols = 0;   // Dense Pool → SPTC Fallback 的列数
    int total_fake_zeros   = 0;
};


// ============================================================================
// CUDA 设备函数: popc 封装
// ============================================================================
__device__ inline int dev_popc(unsigned int x) { return __popc(x); }


// ============================================================================
// 主 Kernel: 单窗口双路路由匹配
//
// Shared Memory 布局 (静态, 约 30KB):
//   s_cols_mask[n]     uint32_t  每列的 32-bit nnz 掩码 (低16位有效)
//   s_col_ids[n]       int       每列的原始列 ID
//   s_col_nnz[n]       int       每列非零元总数 (= popc(mask))
//   s_matched[n]       bool      列匹配标记
//
//   s_pair_G1[p]       uint16_t  Pair 的 G1
//   s_pair_G2[p]       uint16_t  Pair 的 G2
//   s_pair_c1[p]       int       Pair 列1索引
//   s_pair_c2[p]       int       Pair 列2索引 (-1=虚拟0列)
//   s_pair_used[p]     bool      Pair 匹配标记
//
//   s_dense_pool[d]    int       Dense Pool 列索引
//
//   s_sptc[g*4+c]      int       SPTC Group 的列 ID
//   s_tc[b*8+c]        int       TC Block 的列 ID
// ============================================================================
__global__ void block_matching_online_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    int        num_rows,

    int*       __restrict__ out_sptc_groups,
    int*       __restrict__ out_num_sptc_groups,

    int*       __restrict__ out_tc_blocks,
    int*       __restrict__ out_num_tc_blocks,

    int*       __restrict__ out_stats,        // [num_windows * 7]

    int        dense_threshold,
    int        t_max)
{
    // ================================================================
    // Shared Memory 声明
    //
    // 静态共享内存 (~39KB, 小于 RTX 3090 的 48KB 上限):
    //   s_cols_mask/s_col_ids 等核心匹配数组
    //
    // 动态共享内存 (~24KB, 由 kernel launch 时传入):
    //   s_sptc[4*MAX_PAIRS] + s_tc[8*(MAX_COLS/8)]
    //
    // s_col_nnz 已移除: Phase2/3 直接 popc(s_cols_mask[idx]), O(1) 无额外开销.
    // ================================================================
    __shared__ uint32_t s_cols_mask[MAX_COLS];
    __shared__ int      s_col_ids[MAX_COLS];
    __shared__ int      s_num_cols;

    __shared__ uint8_t  s_matched[MAX_COLS];   // uint8_t 省空间 (bool 对齐后也占1B)

    __shared__ uint16_t s_pair_G1[MAX_PAIRS];
    __shared__ uint16_t s_pair_G2[MAX_PAIRS];
    __shared__ int      s_pair_c1[MAX_PAIRS];
    __shared__ int      s_pair_c2[MAX_PAIRS];
    __shared__ uint8_t  s_pair_used[MAX_PAIRS];
    __shared__ int      s_num_pairs;

    __shared__ int      s_dense_pool[MAX_COLS];
    __shared__ int      s_num_dense;

    __shared__ int      s_num_sptc;   // 标量, 保留在静态
    __shared__ int      s_num_tc;

    // 大数组 → 动态 Shared Memory (避免超出 48KB 静态上限)
    extern __shared__ int s_ext[];
    int* s_sptc = s_ext;                                    // [4 * MAX_PAIRS]
    int* s_tc   = s_ext + 4 * MAX_PAIRS;                    // [MAX_COLS]

    // ================================================================
    // Phase 0: 并行 CSR 提取 → 构建列掩码表
    //
    // 策略: 将窗口的 16 行均匀分配给所有线程.
    // 每个线程直接按行遍历 CSR (无需二分查找行号).
    // ================================================================
    int win     = blockIdx.x;
    int row_beg = win * WINDOW_SIZE;
    int row_end = min(row_beg + WINDOW_SIZE, num_rows);
    int win_rows = row_end - row_beg;

    // Thread 0 初始化哈希表
    if (threadIdx.x == 0) {
        s_num_cols = 0;
        for (int i = 0; i < MAX_COLS; ++i) {
            s_cols_mask[i] = 0;
            s_col_ids[i]   = -1;
        }
    }
    __syncthreads();

    // 按行分配: 每个线程处理一批窗口行
    for (int local_r = threadIdx.x; local_r < win_rows; local_r += blockDim.x) {
        int r = row_beg + local_r;
        for (int e = row_ptr[r]; e < row_ptr[r + 1]; ++e) {
            int col = col_ind[e];

            // 线性探测哈希插入 (Fix3: 保险丝)
            int slot = col % MAX_COLS;
            int probe_count = 0;
            while (probe_count < MAX_COLS) {
                int existing = atomicCAS(&s_col_ids[slot], -1, col);
                if (existing == -1 || existing == col) {
                    if (existing == -1) atomicAdd(&s_num_cols, 1);
                    s_col_ids[slot] = col;
                    // Fix1: 32-bit 原生 atomicOr, 无未对齐风险
                    atomicOr(&s_cols_mask[slot], 1u << local_r);
                    break;
                }
                slot = (slot + 1) % MAX_COLS;
                probe_count++;
            }
        }
    }
    __syncthreads();

    // 压实: 稀疏哈希表 → 稠密数组 (Thread 0)
    int n = 0;
    if (threadIdx.x == 0) {
        for (int slot = 0; slot < MAX_COLS; ++slot) {
            if (s_col_ids[slot] != -1) {
                int dst = n++;
                s_col_ids[dst]   = s_col_ids[slot];
                s_cols_mask[dst] = s_cols_mask[slot];
            }
        }
        for (int i = n; i < MAX_COLS; ++i) {
            s_col_ids[i]   = -1;
            s_cols_mask[i] = 0;
        }
        s_num_cols = n;
    }
    __syncthreads();

    // ================================================================
    // 匹配逻辑: Warp 0 (Thread 0~31) 并行执行
    //
    // Phase 1: 每个 lane 覆盖 SEARCH_WINDOW 中的一列, __ballot_sync 并行寻优.
    // Phase 2: Warp 以 32 为步长滚动搜索 GROUP_SEARCH_WINDOW × 1024.
    // Phase 3: Lane 0 串行路由决策.
    // ================================================================
    if (threadIdx.x < 32 && s_num_cols > 0) {
        int lane_id = threadIdx.x;   // 0..31
        n = s_num_cols;

        // ---- 初始化: 所有 32 个 lane 协同 ----
        for (int i = lane_id; i < n; i += 32) {
            s_matched[i] = 0;
        }
        if (lane_id == 0) {
            s_num_pairs  = 0;
            s_num_dense  = 0;
            s_num_sptc   = 0;
            s_num_tc     = 0;
        }
        int total_fz       = 0;  // Fix2: lane 0 写, 其余 lane 携带但不使用
        int fine_fb_count  = 0;
        int coarse_fb_count = 0;
        __syncwarp();

        // ============================================================
        // Phase 1: Warp-Level 细粒度匹配 (1:2 Pairs)
        //
        // SEARCH_WINDOW = 32 = 1 个 Warp 宽度.
        // 每 lane 计算一列 cost, __ballot_sync 找最小 cost 的 lane.
        // cost==0 最高优先, 其次 cost<=t_max 中最小.
        // ============================================================
        for (int i = 0; i < n; ++i) {
            // 所有 lane 必须收敛, 不可在循环体内 divergent continue
            int active = (s_matched[i] == 0) ? 1 : 0;
            unsigned active_mask = __ballot_sync(0xffffffff, active);
            if (active_mask == 0) continue;  // 所有 lane 一致跳过

            uint32_t mask_i = s_cols_mask[i];

            int j     = i + 1 + lane_id;
            bool valid = (j < n) && (s_matched[j] == 0) && (lane_id < SEARCH_WINDOW);
            int my_cost = valid ? dev_popc(mask_i & s_cols_mask[j]) : 999;

            // 贪心找最小 cost (cost==0 最先命中)
            int best_j = -1;
            for (int target_cost = 0; target_cost <= t_max; ++target_cost) {
                unsigned match_mask = __ballot_sync(0xffffffff, my_cost == target_cost);
                if (match_mask > 0) {
                    int winner_lane = __ffs(match_mask) - 1;
                    best_j = i + 1 + winner_lane;
                    break;
                }
            }

            // Lane 0 负责写入共享状态
            if (lane_id == 0) {
                if (best_j != -1) {
                    int p = s_num_pairs++;
                    s_pair_G1[p] = (uint16_t)(mask_i & s_cols_mask[best_j]);
                    s_pair_G2[p] = (uint16_t)(mask_i | s_cols_mask[best_j]);
                    s_pair_c1[p] = i;
                    s_pair_c2[p] = best_j;
                    s_matched[i]      = 1;
                    s_matched[best_j] = 1;
                } else {
                    s_dense_pool[s_num_dense++] = i;
                    s_matched[i] = 1;
                    fine_fb_count++;
                }
            }
            __syncwarp();  // 让 lane 0 的写入对所有 lane 可见
        }

        // ============================================================
        // Phase 2: Warp 步进粗粒度匹配 (2:4 Groups)
        //
        // GROUP_SEARCH_WINDOW = 1024, Warp 以 32 为步长迭代.
        // 每 lane 计算一个候选 j 的 Z, __ballot_sync 找 Z==0 的 lane.
        // ============================================================
        int np = s_num_pairs;   // 所有 lane 读取 (__syncwarp 后可见)
        for (int p = lane_id; p < np; p += 32) {
            s_pair_used[p] = 0;
        }
        __syncwarp();

        for (int i = 0; i < np; ++i) {
            int active2 = (s_pair_used[i] == 0) ? 1 : 0;
            unsigned active_mask2 = __ballot_sync(0xffffffff, active2);
            if (active_mask2 == 0) continue;

            uint16_t pa_G1 = s_pair_G1[i];
            uint16_t pa_G2 = s_pair_G2[i];

            int best_j = -1;

            // Warp 以 32 为步长滚动搜索
            for (int step = 0; step < GROUP_SEARCH_WINDOW; step += 32) {
                int j     = i + 1 + step + lane_id;
                bool valid = (j < np) && (s_pair_used[j] == 0);

                uint16_t Z = 0xFFFF;  // 非零值代表"无效/冲突"
                if (valid) {
                    uint16_t pb_G1 = s_pair_G1[j];
                    uint16_t pb_G2 = s_pair_G2[j];
                    Z = (pa_G1 & pb_G2) | (pa_G2 & pb_G1);
                }

                unsigned match_mask = __ballot_sync(0xffffffff, Z == 0);
                if (match_mask > 0) {
                    int winner_lane = __ffs(match_mask) - 1;
                    best_j = i + 1 + step + winner_lane;
                    break;
                }
            }

            // Lane 0 写入结果
            if (lane_id == 0) {
                if (best_j != -1) {
                    int g = s_num_sptc++;
                    s_sptc[g * 4 + 0] = s_col_ids[s_pair_c1[i]];
                    s_sptc[g * 4 + 1] = s_col_ids[s_pair_c2[i]];
                    s_sptc[g * 4 + 2] = s_col_ids[s_pair_c1[best_j]];
                    s_sptc[g * 4 + 3] = s_col_ids[s_pair_c2[best_j]];

                    int c1_i = s_pair_c1[i], c2_i = s_pair_c2[i];
                    int c1_j = s_pair_c1[best_j], c2_j = s_pair_c2[best_j];
                    int gn = dev_popc(s_cols_mask[c1_i]);
                    gn += (c2_i >= 0) ? dev_popc(s_cols_mask[c2_i]) : 0;
                    gn += dev_popc(s_cols_mask[c1_j]);
                    gn += (c2_j >= 0) ? dev_popc(s_cols_mask[c2_j]) : 0;
                    total_fz += (32 - gn);

                    s_pair_used[i]      = 1;
                    s_pair_used[best_j] = 1;
                } else {
                    int c1 = s_pair_c1[i];
                    int c2 = s_pair_c2[i];
                    if (c1 >= 0) s_dense_pool[s_num_dense++] = c1;
                    if (c2 >= 0) s_dense_pool[s_num_dense++] = c2;
                    s_pair_used[i] = 1;
                    coarse_fb_count += 2;
                }
            }
            __syncwarp();
        }

        // ============================================================
        // Phase 3: 双路路由决策 (Lane 0 串行)
        // ============================================================
        if (lane_id == 0) {
            int nd = s_num_dense;
            int processed = 0;

            // 路径 A: TC 路由 — 每 8 列打包为 16×8 稠密块
            while (nd - processed >= dense_threshold) {
                int b = s_num_tc++;
                for (int k = 0; k < dense_threshold; ++k) {
                    int col_idx = s_dense_pool[processed + k];
                    s_tc[b * dense_threshold + k] = s_col_ids[col_idx];
                }
                processed += dense_threshold;
            }

            // 路径 B: SPTC Fallback — 每组最多塞 2 个残余列
            // 数学约束: 任意 2 列同行 ≤ 2 nnz (2:4 天然满足).
            // 若塞 3-4 列, 同行可能 > 2 nnz, 硬件 mma.sp 指令直接算错!
            int residue = nd - processed;
            if (residue > 0) {
                int num_groups = (residue + 1) / 2;  // 每组最多 2 列
                int idx = processed;
                for (int g = 0; g < num_groups; ++g) {
                    int gp = s_num_sptc++;
                    int fallback_nnz = 0;
                    for (int k = 0; k < 4; ++k) {
                        if (k < 2 && idx < nd) {  // 前两列放真实数据
                            int col_idx = s_dense_pool[idx++];
                            s_sptc[gp * 4 + k] = s_col_ids[col_idx];
                            fallback_nnz += dev_popc(s_cols_mask[col_idx]);
                        } else {                   // 后两列强制补 0
                            s_sptc[gp * 4 + k] = -1;
                        }
                    }
                    total_fz += (32 - fallback_nnz);
                }
            }

            // ---- 写回统计 ----
            out_num_sptc_groups[win] = s_num_sptc;
            out_num_tc_blocks[win]   = s_num_tc;

            int base = win * SMEM_STATS_SIZE;
            out_stats[base + 0] = s_num_sptc;
            out_stats[base + 1] = s_num_tc;
            out_stats[base + 2] = processed;
            out_stats[base + 3] = residue;
            out_stats[base + 4] = fine_fb_count;
            out_stats[base + 5] = coarse_fb_count;
            out_stats[base + 6] = total_fz;
        }
    } // end warp 0

    // ================================================================
    // 结果写回 Global Memory (所有线程协作)
    // ================================================================
    __syncthreads();

    int max_groups = MAX_COLS / 2;

    // SPTC Groups — 所有线程并行写回
    {
        int ng  = s_num_sptc;
        int off = win * max_groups;
        for (int g = threadIdx.x; g < ng; g += blockDim.x) {
            int base = (off + g) * 4;
            for (int k = 0; k < 4; ++k)
                out_sptc_groups[base + k] = s_sptc[g * 4 + k];
        }
    }

    // TC Blocks — 以 MAX_COLS 为窗口间绝对安全步长, 消除 dense_threshold 参数耦合
    {
        int nt  = s_num_tc;
        int off = win * MAX_COLS;
        for (int b = threadIdx.x; b < nt; b += blockDim.x) {
            int base = off + b * dense_threshold;
            for (int k = 0; k < dense_threshold; ++k)
                out_tc_blocks[base + k] = s_tc[b * dense_threshold + k];
        }
    }
}


// ============================================================================
// Host 端封装
// ============================================================================
static void launch_block_matching_online(
    const int*   row_ptr,
    const int*   col_ind,
    int          rows,
    int          /*cols*/,
    int          window_size,
    int          dense_threshold,
    int          t_max,
    std::vector<int>& out_sptc_groups,
    std::vector<int>& out_num_sptc_per_win,
    std::vector<int>& out_tc_blocks,
    std::vector<int>& out_num_tc_per_win,
    std::vector<int>& out_stats)
{
    int num_windows = (rows + window_size - 1) / window_size;

    int max_groups_per_win    = MAX_COLS / 2;
    int total_max_groups      = num_windows * max_groups_per_win;
    int total_max_tc_capacity = num_windows * MAX_COLS;  // 以 MAX_COLS 为窗口间硬隔离步长

    out_sptc_groups.resize(total_max_groups * 4, -1);
    out_num_sptc_per_win.resize(num_windows, 0);
    out_tc_blocks.resize(total_max_tc_capacity, -1);
    out_num_tc_per_win.resize(num_windows, 0);
    out_stats.resize(num_windows * SMEM_STATS_SIZE, 0);

    int nnz = row_ptr[rows];
    int *d_row_ptr, *d_col_ind, *d_sptc, *d_num_sptc, *d_tc, *d_num_tc, *d_stats;

    cudaMalloc(&d_row_ptr,  (rows + 1) * sizeof(int));
    cudaMalloc(&d_col_ind,  nnz * sizeof(int));
    cudaMalloc(&d_sptc,     total_max_groups * 4 * sizeof(int));
    cudaMalloc(&d_num_sptc, num_windows * sizeof(int));
    cudaMalloc(&d_tc,       total_max_tc_capacity * sizeof(int));
    cudaMalloc(&d_num_tc,   num_windows * sizeof(int));
    cudaMalloc(&d_stats,    num_windows * SMEM_STATS_SIZE * sizeof(int));

    // ---- CUDA Event 计时 ----
    cudaEvent_t ev_h2d_start, ev_h2d_stop, ev_kern_start, ev_kern_stop, ev_d2h_start, ev_d2h_stop;
    cudaEventCreate(&ev_h2d_start); cudaEventCreate(&ev_h2d_stop);
    cudaEventCreate(&ev_kern_start); cudaEventCreate(&ev_kern_stop);
    cudaEventCreate(&ev_d2h_start); cudaEventCreate(&ev_d2h_stop);

    // === Phase 1: H2D 传输 ===
    cudaEventRecord(ev_h2d_start);
    cudaMemcpy(d_row_ptr, row_ptr, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, col_ind, nnz * sizeof(int),      cudaMemcpyHostToDevice);
    cudaMemset(d_num_sptc, 0, num_windows * sizeof(int));
    cudaMemset(d_num_tc,   0, num_windows * sizeof(int));
    cudaEventRecord(ev_h2d_stop);

    // === Phase 2: Kernel 计算 ===
    dim3 grid(num_windows);
    dim3 block(256);

    size_t dyn_smem = (4 * MAX_PAIRS + MAX_COLS) * sizeof(int);
    cudaFuncSetAttribute(block_matching_online_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem);

    cudaEventRecord(ev_kern_start);
    block_matching_online_kernel<<<grid, block, dyn_smem>>>(
        d_row_ptr, d_col_ind, rows,
        d_sptc, d_num_sptc, d_tc, d_num_tc, d_stats,
        dense_threshold, t_max);
    cudaEventRecord(ev_kern_stop);

    cudaDeviceSynchronize();

    // === Phase 3: D2H 传输 ===
    cudaEventRecord(ev_d2h_start);
    cudaMemcpy(out_sptc_groups.data(),      d_sptc,
               total_max_groups * 4 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_num_sptc_per_win.data(), d_num_sptc,
               num_windows * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_tc_blocks.data(),        d_tc,
               total_max_tc_capacity * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_num_tc_per_win.data(),   d_num_tc,
               num_windows * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_stats.data(),            d_stats,
               num_windows * SMEM_STATS_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(ev_d2h_stop);
    cudaDeviceSynchronize();

    // ---- 打印耗时分解 ----
    float ms_h2d, ms_kern, ms_d2h;
    cudaEventElapsedTime(&ms_h2d,  ev_h2d_start,  ev_h2d_stop);
    cudaEventElapsedTime(&ms_kern, ev_kern_start, ev_kern_stop);
    cudaEventElapsedTime(&ms_d2h,  ev_d2h_start,  ev_d2h_stop);

    printf("\n");
    printf("╔══════════════════════════════════════════╗\n");
    printf("║  CUDA Online 耗时分解 (ms)               ║\n");
    printf("╠══════════════════════════════════════════╣\n");
    printf("║  H2D 数据传输:     %8.3f ms          ║\n", ms_h2d);
    printf("║  Kernel 计算:      %8.3f ms          ║\n", ms_kern);
    printf("║  D2H 数据传输:     %8.3f ms          ║\n", ms_d2h);
    printf("║  ─────────────────────────────────── ║\n");
    printf("║  CUDA 总计:        %8.3f ms          ║\n", ms_h2d + ms_kern + ms_d2h);
    printf("╚══════════════════════════════════════════╝\n");

    cudaEventDestroy(ev_h2d_start); cudaEventDestroy(ev_h2d_stop);
    cudaEventDestroy(ev_kern_start); cudaEventDestroy(ev_kern_stop);
    cudaEventDestroy(ev_d2h_start); cudaEventDestroy(ev_d2h_stop);

    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);
    cudaFree(d_sptc);
    cudaFree(d_num_sptc);
    cudaFree(d_tc);
    cudaFree(d_num_tc);
    cudaFree(d_stats);
}


// ============================================================================
// pybind11 接口
// ============================================================================
py::dict match_2to4_online_py(
    py::array_t<int,   py::array::c_style | py::array::forcecast> row_ptr_arr,
    py::array_t<int,   py::array::c_style | py::array::forcecast> col_ind_arr,
    py::array_t<float, py::array::c_style | py::array::forcecast> /*values_arr*/,
    int window_size     = 16,
    int dense_threshold = 8,
    int t_max           = 8)
{
    py::buffer_info row_buf = row_ptr_arr.request();
    py::buffer_info col_buf = col_ind_arr.request();

    const int* row_ptr = static_cast<const int*>(row_buf.ptr);
    const int* col_ind = static_cast<const int*>(col_buf.ptr);

    int rows = (int)row_buf.shape[0] - 1;
    int nnz  = (int)col_buf.shape[0];

    int cols = 0;
    for (int i = 0; i < nnz; ++i)
        if (col_ind[i] >= cols) cols = col_ind[i] + 1;

    // 调用 CUDA Kernel
    std::vector<int> sptc_groups, num_sptc_per_win;
    std::vector<int> tc_blocks, num_tc_per_win;
    std::vector<int> stats;

    launch_block_matching_online(
        row_ptr, col_ind, rows, cols, window_size,
        dense_threshold, t_max,
        sptc_groups, num_sptc_per_win,
        tc_blocks, num_tc_per_win, stats);

    // 聚合统计
    int num_windows = (rows + window_size - 1) / window_size;
    OnlineMatchStats total;
    for (int w = 0; w < num_windows; ++w) {
        int base = w * SMEM_STATS_SIZE;
        total.num_sptc_groups    += stats[base + 0];
        total.num_tc_blocks      += stats[base + 1];
        total.dense_to_tc_cols   += stats[base + 2];
        total.dense_to_sptc_cols += stats[base + 3];
        total.fine_fb            += stats[base + 4];
        total.coarse_fb_cols     += stats[base + 5];
        total.total_fake_zeros   += stats[base + 6];
    }

    int total_16x16_blocks = 0, block_padding = 0;
    for (int w = 0; w < num_windows; ++w) {
        int ng = num_sptc_per_win[w];
        int blk = (ng + 3) / 4;
        total_16x16_blocks += blk;
        block_padding += (blk * 4 - ng);
    }

    // ---- C++ 层输出 ----
    printf("\n");
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║  Online Dual-Routing TC+SPTC 匹配统计 (v2)  ║\n");
    printf("╠══════════════════════════════════════════════╣\n");
    printf("║  矩阵形状:          %6d × %-6d          ║\n", rows, cols);
    printf("║  总非零元:          %-12d            ║\n", nnz);
    printf("║  稀疏度:            %8.4f%%             ║\n",
           100.0 * (1.0 - (double)nnz / ((double)rows * cols)));
    printf("╠══════════════════════════════════════════════╣\n");
    printf("║  [路由决策]                                  ║\n");
    printf("║  SPTC 2:4 Groups:   %-8d              ║\n", total.num_sptc_groups);
    printf("║  TC 16×8 Blocks:    %-8d              ║\n", total.num_tc_blocks);
    printf("║  → TC 路由列数:     %-8d              ║\n", total.dense_to_tc_cols);
    printf("║  → SPTC Fallback:   %-8d              ║\n", total.dense_to_sptc_cols);
    printf("╠══════════════════════════════════════════════╣\n");
    printf("║  [Ablation 分离统计]                         ║\n");
    printf("║  细粒度孤儿列:      %-8d (Phase1 → Pool) ║\n", total.fine_fb);
    printf("║  粗粒度拆散列:      %-8d (Phase2 → Pool) ║\n", total.coarse_fb_cols);
    printf("╠══════════════════════════════════════════════╣\n");
    printf("║  [SPTC 硬件层]                               ║\n");
    printf("║  16×16 Block 数:    %-8d              ║\n", total_16x16_blocks);
    printf("║  Block 填充 Group:  %-8d              ║\n", block_padding);
    printf("║  假0填充 (O(1)):    %-8d              ║\n", total.total_fake_zeros);
    printf("╚══════════════════════════════════════════════╝\n\n");

    py::dict result;
    result["total_rows"]           = rows;
    result["total_cols"]           = cols;
    result["total_nnz"]            = (long long)nnz;
    result["num_sptc_groups"]      = total.num_sptc_groups;
    result["num_tc_blocks"]        = total.num_tc_blocks;
    result["dense_to_tc_cols"]     = total.dense_to_tc_cols;
    result["dense_to_sptc_cols"]   = total.dense_to_sptc_cols;
    result["fine_fb"]              = total.fine_fb;
    result["coarse_fb_cols"]       = total.coarse_fb_cols;
    result["num_16x16_blocks"]     = total_16x16_blocks;
    result["block_padding_groups"] = block_padding;
    result["fake_zeros"]           = total.total_fake_zeros;

    return result;
}


// ============================================================================
// 模块定义
// ============================================================================
PYBIND11_MODULE(matching_utils_online, m) {
    m.doc() = R"pbdoc(
        Online Dual-Routing TC+SPTC 匹配模块 (CUDA Kernel, v2 修复版)

        v2 修复:
          1. uint32_t s_cols_mask → 消除未对齐 atomicOr
          2. O(1) fake_zeros → 在 Phase 2/3 构建时直接累加
          3. MAX_COLS=2048 + probe_count 保险丝
          4. fine_fb / coarse_fb 分离统计

        接口:
          match_2to4_online(row_ptr, col_ind, values,
                            window_size=16, dense_threshold=8, t_max=8) -> dict
    )pbdoc";

    m.def("match_2to4_online", &match_2to4_online_py,
          py::arg("row_ptr"),
          py::arg("col_ind"),
          py::arg("values"),
          py::arg("window_size")     = 16,
          py::arg("dense_threshold") = 8,
          py::arg("t_max")           = 8,
          R"pbdoc(
对 CSR 格式稀疏矩阵执行 Online Dual-Routing TC+SPTC 匹配 (CUDA).

参数:
    row_ptr (np.ndarray):    CSR 行偏移, dtype=int32
    col_ind (np.ndarray):    CSR 列索引, dtype=int32
    values (np.ndarray):     CSR 数值, dtype=float32
    window_size (int):       窗口行数, 默认 16
    dense_threshold (int):   TC 路由阈值, 默认 8
    t_max (int):             细粒度匹配容忍阈值, 默认 8

返回:
    dict: {
        num_sptc_groups, num_tc_blocks,
        dense_to_tc_cols, dense_to_sptc_cols,
        fine_fb, coarse_fb_cols, fake_zeros,
        num_16x16_blocks, block_padding_groups
    }
          )pbdoc");
}
