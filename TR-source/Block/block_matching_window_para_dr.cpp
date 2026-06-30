/**
 * block_matching.cpp — 双路路由 2:4 结构化稀疏匹配 (Dual-Routing TC+SPTC)
 *
 * 算法原理:
 *   将稀疏矩阵按 16 行划分为窗口, 通过位运算进行 1:2/2:4 结构化匹配.
 *   匹配失败的"孤儿列"不再无条件补 0, 而是进入 Dense Pool 进行双路路由:
 *     - 凑满 dense_threshold 列 → 打包为 TC 16×8 稠密块 (TC 路由)
 *     - 不足 dense_threshold → SPTC Fallback 补 0 (每组最多 2 列)
 *
 * 线程安全: 每窗口独立 dense_pool, 结果存入预分配 per-window 数组,
 *   OpenMP 循环后主线程聚合, 零锁开销.
 *
 * 暴露接口:
 *   matching_utils.match_2to4(row_ptr, col_ind, values,
 *                              window_size=16, dense_threshold=8, t_max=8)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <omp.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

// ============================================================================
// 数据结构
// ============================================================================

struct WinColumn {
    int      col_id;
    uint16_t mask;
    int      nnz;
    // 重载小于号，让 lower_bound 自动识别，不需要再写复杂的 Lambda
    bool operator<(const WinColumn& other) const {
        return col_id < other.col_id;
    }
};

struct Pair1x2 {
    int      col1;
    int      col2;
    uint16_t G1;
    uint16_t G2;
    int      nnz = 0;
};

/**
 * 双路路由匹配统计 (C++ 输出 + Python 返回)
 */
struct MatchStats {
    int       total_rows          = 0;
    int       total_cols          = 0;
    long long total_nnz           = 0;
    double    sparsity            = 0.0;
    int       num_row_panels      = 0;   // 16 行 Row Panel 条带数
    int       num_sptc_groups     = 0;   // 最终 2:4 SPTC Group 数
    int       num_tc_blocks       = 0;   // 最终 16×8 Dense TC Block 数
    int       num_16x16_blocks    = 0;   // 16×16 SPTC 稀疏块 (4 Groups → 1 Block)
    int       block_padding_groups = 0;  // 为对齐 16×16 Block 填充的空白 Group 数
    int       fine_fb             = 0;   // 细粒度孤儿列进池数
    int       coarse_fb           = 0;   // 粗粒度不兼容 Pair 进池列数
    int       fake_zeros          = 0;   // 元素级假0填充总数

    // Dense TC 块密度统计
    long long tc_total_nnz        = 0;      // TC 块累计非零元
    int       tc_block_count      = 0;      // TC 块总个数
    int       tc_block_max_nnz    = 0;      // 单 TC 块最大 nnz
    int       tc_block_min_nnz    = 999999; // 单 TC 块最小 nnz
};


// ============================================================================
// 核心算法: 单窗口双路路由匹配
//
// 参数:
//   win_rows        - 窗口实际行数
//   columns         - 窗口内候选列 (已按 col_id 排序)
//   dense_threshold - TC 路由阈值
//   t_max           - 细粒度容忍阈值
//
// 列 ID 为 -1 表示虚拟全0列.
// ============================================================================
static void match_window_2to4(
    int win_rows,
    std::vector<WinColumn>& columns,
    int dense_threshold,
    int t_max,
    // 出参:
    std::vector<int>& out_sptc_flat,   // [num_groups * 4]
    std::vector<int>& out_tc_flat,     // [num_blocks * dense_threshold]
    MatchStats&       win_stats)
{
    int n = (int)columns.size();
    if (n == 0) return;

    // ---- 步骤 1: 按原始列索引排序 ----
    std::sort(columns.begin(), columns.end(),
        [](const WinColumn& a, const WinColumn& b) {
            return a.col_id < b.col_id;
        });

    // ================================================================
    // 步骤 2: 细粒度滑动视窗匹配 (1:2 Pairs)
    //
    // 找不到匹配 → 列 ID 压入 local_dense_pool, fine_fb++
    // ================================================================
    const int SEARCH_WINDOW = 32;

    std::vector<bool> matched(n, false);
    std::vector<Pair1x2> pairs;
    std::vector<int> local_dense_pool;  // 窗口局部 Dense Pool (列 ID)

    for (int i = 0; i < n; ++i) {
        if (matched[i]) continue;

        uint16_t mask_i = columns[i].mask;
        int best_j = -1;
        int min_cost = t_max + 1;
        int j_end = std::min(i + 1 + SEARCH_WINDOW, n);

        for (int j = i + 1; j < j_end; ++j) {
            if (matched[j]) continue;
            int cost = __builtin_popcount(mask_i & columns[j].mask);
            if (cost == 0) { best_j = j; break; }
            if (cost <= t_max && cost < min_cost) { min_cost = cost; best_j = j; }
        }

        if (best_j != -1) {
            Pair1x2 p;
            p.col1 = columns[i].col_id;
            p.col2 = columns[best_j].col_id;
            p.G1   = mask_i & columns[best_j].mask;
            p.G2   = mask_i | columns[best_j].mask;
            p.nnz  = columns[i].nnz + columns[best_j].nnz;
            pairs.push_back(p);
            matched[i] = true;
            matched[best_j] = true;
        } else {
            local_dense_pool.push_back(columns[i].col_id);
            matched[i] = true;
            win_stats.fine_fb++;
        }
    }

    // ================================================================
    // 步骤 3: 粗粒度滑动视窗匹配 (2:4 Groups)
    //
    // Z == 0 → SPTC Group + O(1) fake_zeros.
    // !found → 拆散 Pair 的 2 列进 local_dense_pool, coarse_fb += 2.
    // ================================================================
    const int GROUP_SEARCH_WINDOW = 1024;

    int np = (int)pairs.size();
    std::vector<bool> pair_used(np, false);

    for (int i = 0; i < np; ++i) {
        if (pair_used[i]) continue;

        const Pair1x2& pa = pairs[i];
        bool found = false;
        int j_end = std::min(i + 1 + GROUP_SEARCH_WINDOW, np);

        for (int j = i + 1; j < j_end; ++j) {
            if (pair_used[j]) continue;

            const Pair1x2& pb = pairs[j];
            uint16_t Z = (pa.G1 & pb.G2) | (pa.G2 & pb.G1);

            if (Z == 0) {
                out_sptc_flat.push_back(pa.col1);
                out_sptc_flat.push_back(pa.col2);
                out_sptc_flat.push_back(pb.col1);
                out_sptc_flat.push_back(pb.col2);

                // O(1) fake_zeros
                win_stats.fake_zeros += (win_rows * 2) - (pa.nnz + pb.nnz);

                pair_used[i] = true;
                pair_used[j] = true;
                found = true;
                break;
            }
        }

        if (!found) {
            if (pa.col1 >= 0) local_dense_pool.push_back(pa.col1);
            if (pa.col2 >= 0) local_dense_pool.push_back(pa.col2);
            pair_used[i] = true;
            win_stats.coarse_fb += 2;
        }
    }

    // ================================================================
    // 步骤 4: 局部双路路由决策
    //
    // ≥ dense_threshold 列 → TC 16×8 稠密块
    // <  dense_threshold 列 → SPTC Fallback (每组最多 2 列 + 补0)
    // ================================================================
    int nd = (int)local_dense_pool.size();
    int processed = 0;

    // 路径 A: TC 路由
    while (nd - processed >= dense_threshold) {
        int block_nnz = 0;
        for (int k = 0; k < dense_threshold; ++k) {
            int col_id = local_dense_pool[processed + k];
            out_tc_flat.push_back(col_id);

            // O(log N) 二分查列在当前窗口的 nnz
            WinColumn target;
            target.col_id = col_id;
            auto it = std::lower_bound(columns.begin(), columns.end(), target);
            if (it != columns.end() && it->col_id == col_id)
                block_nnz += it->nnz;
        }
        win_stats.tc_total_nnz += block_nnz;
        win_stats.tc_block_count++;
        win_stats.tc_block_max_nnz = std::max(win_stats.tc_block_max_nnz, block_nnz);
        win_stats.tc_block_min_nnz = std::min(win_stats.tc_block_min_nnz, block_nnz);
        processed += dense_threshold;
    }

    // 路径 B: SPTC Fallback — 尾部残余, 每组最多塞 2 个真实列
    int residue = nd - processed;
    if (residue > 0) {
        int num_groups = (residue + 1) / 2;  // ceil(residue/2)
        int idx = processed;
        for (int g = 0; g < num_groups; ++g) {
            int fallback_nnz = 0;
            for (int k = 0; k < 4; ++k) {
                if (k < 2 && idx < nd) {
                    int col_id = local_dense_pool[idx++];
                    out_sptc_flat.push_back(col_id);

                    // O(log N) 二分查找: 步骤1 已按 col_id 升序排列
                    WinColumn target;
                    target.col_id = col_id;
                    // auto it = std::lower_bound(columns.begin(), columns.end(), target,
                    //     [](const WinColumn& a, const WinColumn& b) {
                    //         return a.col_id < b.col_id;
                    //     });
                    auto it = std::lower_bound(columns.begin(), columns.end(), target);
                    if (it != columns.end() && it->col_id == col_id) {
                        fallback_nnz += it->nnz;
                    }
                } else {
                    out_sptc_flat.push_back(-1);
                }
            }
            win_stats.fake_zeros += (win_rows * 2) - fallback_nnz;
        }
    }

    win_stats.num_sptc_groups = (int)out_sptc_flat.size() / 4;
    win_stats.num_tc_blocks   = (int)out_tc_flat.size() / dense_threshold;

    // ---- 16×4 Group → 16×16 SPTC 稀疏块硬件映射 ----
    // 每 4 个同窗口 Group 横向拼接为一个 16×16 的 m16n8k16 指令输入块.
    // 每个窗口独立打包 (ceil), 不足 4 个的部分用全 0 Group 补齐.
    int num_sptc   = win_stats.num_sptc_groups;
    int blocks     = (num_sptc + 3) / 4;            // ceil(num_sptc / 4)
    int pad_groups = blocks * 4 - num_sptc;
    win_stats.num_16x16_blocks    = blocks;
    win_stats.block_padding_groups = pad_groups;
}


// ============================================================================
// 顶层入口: 对整个稀疏矩阵执行双路路由匹配
//
// 线程安全: 预分配 per-window 数组, OpenMP 各线程写入 win 索引 (无竞争).
//   循环结束后主线程串行聚合.
// ============================================================================
static MatchStats run_2to4_matching(
    const int*    row_ptr,
    const int*    col_ind,
    const float*  /*values*/,
    int           rows,
    int           cols,
    int           window_size,
    int           dense_threshold,
    int           t_max)
{
    int num_windows = (rows + window_size - 1) / window_size;

    MatchStats total_stats;
    total_stats.total_rows    = rows;
    total_stats.total_cols    = cols;
    total_stats.total_nnz     = row_ptr[rows];
    total_stats.sparsity      = 1.0 - (double)total_stats.total_nnz / ((double)rows * cols);
    total_stats.num_row_panels = num_windows;

    // 预分配 per-window 容器 (不同 win 索引无竞争)
    std::vector<std::vector<int>> all_sptc(num_windows);
    std::vector<std::vector<int>> all_tc(num_windows);
    std::vector<MatchStats>       all_win_stats(num_windows);

    #pragma omp parallel for schedule(dynamic, 16)
    for (int win = 0; win < num_windows; ++win) {
        int row_start = win * window_size;
        int row_end   = std::min(row_start + window_size, rows);
        int win_rows  = row_end - row_start;

        // ---- 步骤 0: 窗口划分与编码 ----
        std::unordered_map<int, uint16_t> col_mask_of;
        col_mask_of.reserve(256);

        for (int r = row_start; r < row_end; ++r) {
            int local_row = r - row_start;
            uint16_t bit  = (uint16_t)1 << local_row;
            for (int e = row_ptr[r]; e < row_ptr[r + 1]; ++e)
                col_mask_of[col_ind[e]] |= bit;
        }

        std::vector<WinColumn> columns;
        columns.reserve(col_mask_of.size());
        for (const auto& kv : col_mask_of) {
            WinColumn wc;
            wc.col_id = kv.first;
            wc.mask   = kv.second;
            wc.nnz    = __builtin_popcount(kv.second);
            columns.push_back(wc);
        }

        if (columns.empty()) continue;

        // ---- 单窗口双路路由匹配 ----
        std::vector<int> sptc_flat, tc_flat;
        MatchStats win_stats;

        match_window_2to4(win_rows, columns, dense_threshold, t_max,
                          sptc_flat, tc_flat, win_stats);

        // 无锁写入 per-window 数组
        all_sptc[win] = std::move(sptc_flat);
        all_tc[win]   = std::move(tc_flat);
        all_win_stats[win] = win_stats;
    }

    // ---- 主线程串行聚合 (零锁开销) ----
    for (int win = 0; win < num_windows; ++win) {
        const MatchStats& ws = all_win_stats[win];
        total_stats.num_sptc_groups  += ws.num_sptc_groups;
        total_stats.num_tc_blocks    += ws.num_tc_blocks;
        total_stats.num_16x16_blocks += ws.num_16x16_blocks;
        total_stats.block_padding_groups += ws.block_padding_groups;
        total_stats.fine_fb          += ws.fine_fb;
        total_stats.coarse_fb        += ws.coarse_fb;
        total_stats.fake_zeros       += ws.fake_zeros;

        // Dense TC 块密度统计
        total_stats.tc_total_nnz     += ws.tc_total_nnz;
        total_stats.tc_block_count   += ws.tc_block_count;
        total_stats.tc_block_max_nnz  = std::max(total_stats.tc_block_max_nnz, ws.tc_block_max_nnz);
        total_stats.tc_block_min_nnz  = std::min(total_stats.tc_block_min_nnz, ws.tc_block_min_nnz);
    }

    return total_stats;
}


// ============================================================================
// pybind11 接口
// ============================================================================
py::dict match_2to4_py(
    py::array_t<int,   py::array::c_style | py::array::forcecast> row_ptr_arr,
    py::array_t<int,   py::array::c_style | py::array::forcecast> col_ind_arr,
    py::array_t<float, py::array::c_style | py::array::forcecast> values_arr,
    int window_size     = 16,
    int dense_threshold = 8,
    int t_max           = 8)
{
    py::buffer_info row_buf = row_ptr_arr.request();
    py::buffer_info col_buf = col_ind_arr.request();
    py::buffer_info val_buf = values_arr.request();

    const int*   row_ptr = static_cast<const int*>(row_buf.ptr);
    const int*   col_ind = static_cast<const int*>(col_buf.ptr);
    const float* values  = static_cast<const float*>(val_buf.ptr);

    int rows = (int)row_buf.shape[0] - 1;
    int nnz  = (int)col_buf.shape[0];

    int cols = 0;
    for (int i = 0; i < nnz; ++i)
        if (col_ind[i] >= cols) cols = col_ind[i] + 1;

    if (window_size < 1 || window_size > 64)
        throw std::runtime_error("window_size 必须在 [1, 64] 范围内");
    if (rows <= 0 || cols <= 0)
        throw std::runtime_error("无效的矩阵维度");

    MatchStats stats = run_2to4_matching(row_ptr, col_ind, values,
                                         rows, cols, window_size,
                                         dense_threshold, t_max);

    // ---- C++ 层输出 ----
    printf("\n");
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║   Dual-Routing TC+SPTC 匹配统计             ║\n");
    printf("╠══════════════════════════════════════════════╣\n");
    printf("║  矩阵形状:          %6d × %-6d          ║\n", stats.total_rows, stats.total_cols);
    printf("║  总非零元数:        %-14lld            ║\n", (long long)stats.total_nnz);
    printf("║  稀疏度 (Sparsity):   %8.4f%%             ║\n", stats.sparsity * 100.0);
    printf("╠══════════════════════════════════════════════╣\n");
    printf("║  [逻辑层]                                    ║\n");
    printf("║  16行 Row Panels:  %-8d                 ║\n", stats.num_row_panels);
    printf("╠══════════════════════════════════════════════╣\n");
    printf("║  [路由决策]                                  ║\n");
    printf("║  SPTC 2:4 Groups:  %-8d                ║\n", stats.num_sptc_groups);
    printf("║  TC 16×8 Blocks:   %-8d                ║\n", stats.num_tc_blocks);
    if (stats.tc_block_count > 0) {
        int     slots_per_block = 16 * 8;
        double  avg_nnz = (double)stats.tc_total_nnz / stats.tc_block_count;
        double  avg_den = avg_nnz / slots_per_block;
        double  min_den = (double)stats.tc_block_min_nnz / slots_per_block;
        double  max_den = (double)stats.tc_block_max_nnz / slots_per_block;
        printf("║  TC 块的 nnz/块:    avg=%-7.1f  min=%-5d  max=%-5d║\n",
               avg_nnz, stats.tc_block_min_nnz, stats.tc_block_max_nnz);
        printf("║  Dense TC 密度:     avg=%-7.4f  min=%-7.4f  max=%-7.4f║\n",
               avg_den, min_den, max_den);
    }
    printf("╠══════════════════════════════════════════════╣\n");
    printf("║  [SPTC 硬件层 — m16n8k16 指令映射]          ║\n");
    printf("║  16×16 Block 数:   %-8d                ║\n", stats.num_16x16_blocks);
    printf("║  Block 填充 Group: %-8d                ║\n", stats.block_padding_groups);
    printf("╠══════════════════════════════════════════════╣\n");
    printf("║  [Ablation 分离统计]                         ║\n");
    printf("║  细粒度孤儿列:     %-8d (Phase1→Pool)    ║\n", stats.fine_fb);
    printf("║  粗粒度拆散列:     %-8d (Phase2→Pool)    ║\n", stats.coarse_fb);
    printf("║  元素级假0填充:    %-8d                ║\n", stats.fake_zeros);
    printf("╚══════════════════════════════════════════════╝\n\n");

    py::dict result;
    result["total_rows"]        = stats.total_rows;
    result["total_cols"]        = stats.total_cols;
    result["total_nnz"]         = (long long)stats.total_nnz;
    result["sparsity"]          = stats.sparsity;
    result["num_row_panels"]    = stats.num_row_panels;
    result["num_sptc_groups"]     = stats.num_sptc_groups;
    result["num_tc_blocks"]       = stats.num_tc_blocks;
    result["num_16x16_blocks"]    = stats.num_16x16_blocks;
    result["block_padding_groups"] = stats.block_padding_groups;
    result["fine_fallback"]       = stats.fine_fb;
    result["coarse_fallback"]     = stats.coarse_fb;
    result["fake_zeros"]          = stats.fake_zeros;
    result["tc_total_nnz"]        = (long long)stats.tc_total_nnz;
    result["tc_block_count"]      = stats.tc_block_count;
    result["tc_block_max_nnz"]    = stats.tc_block_max_nnz;
    result["tc_block_min_nnz"]    = stats.tc_block_min_nnz;

    return result;
}


// ============================================================================
// 模块定义
// ============================================================================
PYBIND11_MODULE(matching_utils, m) {
    m.doc() = R"pbdoc(
        Dual-Routing TC+SPTC 结构化稀疏匹配模块

        核心算法:
          1. 16行窗口划分 → uint16_t 编码
          2. 滑动视窗细粒度匹配 (1:2 Pair)
          3. Z_conflict==0 粗粒度匹配 (2:4 Group)
          4. 匹配失败列 → Dense Pool → 双路路由:
               ≥dense_threshold → TC 16×8 块
               <dense_threshold  → SPTC Fallback 补0
          5. O(1) fake_zeros 计算

        接口:
          match_2to4(row_ptr, col_ind, values,
                     window_size=16, dense_threshold=8, t_max=8) -> dict
    )pbdoc";

    m.def("match_2to4", &match_2to4_py,
          py::arg("row_ptr"),
          py::arg("col_ind"),
          py::arg("values"),
          py::arg("window_size")     = 16,
          py::arg("dense_threshold") = 8,
          py::arg("t_max")           = 8,
          R"pbdoc(
对 CSR 格式稀疏矩阵执行 Dual-Routing TC+SPTC 匹配.

参数:
    row_ptr (np.ndarray):    CSR 行偏移, dtype=int32
    col_ind (np.ndarray):    CSR 列索引, dtype=int32
    values (np.ndarray):     CSR 数值, dtype=float32
    window_size (int):       窗口行数, 默认 16
    dense_threshold (int):   TC 路由阈值, 默认 8
    t_max (int):             细粒度容忍阈值, 默认 8

返回:
    dict: {
        total_rows, total_cols, total_nnz, sparsity,
        num_row_panels, num_sptc_groups, num_tc_blocks,
        fine_fallback, coarse_fallback, fake_zeros
    }
          )pbdoc");
}
