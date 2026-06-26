/**
 * block_matching_bucket.cpp
 *   基于高低位特征分桶的物理隔离匹配 (High-Low Bucketing Matching, Ablation Study)
 *
 * Step 2 细粒度匹配:
 *   Phase 0: 按 nnz 在低 8 位/高 8 位的分布, 将列分为 top_heavy / bottom_heavy / balanced
 *   Phase 1: top_heavy × bottom_heavy 跨极区匹配 (天然互补)
 *   Phase 2: 剩余极区列 × balanced 匹配
 *   Phase 3: balanced × balanced 内部消化
 *   Phase 4: 全量兜底 Fallback
 *
 * Step 4 Padding: O(1) 计算, fake_zeros = 32 - Σ(col_nnz) per group
 *
 * 暴露接口:
 *   matching_utils_bucket.match_2to4_bucket(row_ptr, col_ind, values, window_size=16)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

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
    int      nnz;       // popcnt(mask), 窗口内非零元总数
};

struct Pair1x2 {
    int      col1;
    int      col2;
    uint16_t G1;        // col1 & col2
    uint16_t G2;        // col1 | col2
    bool     is_fallback;
};

struct Group2x4 {
    int cols[4];
    int zero_cols = 0;
};

struct MatchStats {
    int       total_rows          = 0;
    int       total_cols          = 0;
    long long total_nnz           = 0;
    double    sparsity            = 0.0;
    int       num_row_panels      = 0;
    int       num_groups          = 0;
    int       num_16x16_blocks    = 0;
    int       block_padding_groups = 0;
    int       fine_fb             = 0;
    int       coarse_fb           = 0;
    int       fake_zeros          = 0;
};


// ============================================================================
// 辅助: 分桶
// ============================================================================
static void bucket_columns(
    const std::vector<WinColumn>& columns,
    std::vector<int>& top_heavy,
    std::vector<int>& bottom_heavy,
    std::vector<int>& balanced)
{
    int n = (int)columns.size();
    for (int i = 0; i < n; ++i) {
        uint16_t mask = columns[i].mask;
        int top_nnz    = __builtin_popcount(mask & 0x00FF);  // 行 0-7
        int bottom_nnz = __builtin_popcount(mask & 0xFF00);  // 行 8-15

        if (top_nnz > bottom_nnz)
            top_heavy.push_back(i);
        else if (bottom_nnz > top_nnz)
            bottom_heavy.push_back(i);
        else
            balanced.push_back(i);
    }
}

// ============================================================================
// 辅助: 在候选桶中寻找最优匹配 (带滑动视窗截断)
//
// 在 candidates 中找 cost <= T_MAX 中 cost 最小的列.
// cost==0 立即截断. 最多查看 SEARCH_WINDOW 个未匹配候选.
// ============================================================================
static int find_best_in_bucket(
    const std::vector<WinColumn>& columns,
    uint16_t mask_i,
    const std::vector<int>& candidates,
    const std::vector<bool>& matched,
    int T_MAX,
    int SEARCH_WINDOW)
{
    int best_j = -1;
    int min_cost = T_MAX + 1;
    int seen = 0;

    for (int idx : candidates) {
        if (matched[idx]) continue;

        int cost = __builtin_popcount(mask_i & columns[idx].mask);

        if (cost == 0) return idx;               // 完美匹配, 立即截断
        if (cost <= T_MAX && cost < min_cost) {
            min_cost = cost;
            best_j = idx;
        }
        if (++seen >= SEARCH_WINDOW) break;      // 视窗截断
    }
    return best_j;
}


// ============================================================================
// 核心算法: 单窗口 2:4 结构化匹配 (Bucket-based Step 2)
// ============================================================================
static std::vector<Group2x4> match_window_2to4_bucket(
    int win_rows,
    std::vector<WinColumn>& columns,
    const std::unordered_map<int, uint16_t>& col_mask_of,
    MatchStats& stats)
{
    std::vector<Group2x4> groups;

    if (columns.empty()) return groups;

    int n = (int)columns.size();

    // ---- 构建 col_id → nnz 映射, 供 Step 4 O(1) 计算 fake_zeros ----
    std::unordered_map<int, int> col_nnz;
    col_nnz.reserve(n);
    for (int i = 0; i < n; ++i) {
        col_nnz[columns[i].col_id] = columns[i].nnz;
    }

    // ================================================================
    // 步骤 1: 按原始列索引排序 (保持空间局部性)
    // ================================================================
    std::sort(columns.begin(), columns.end(),
        [](const WinColumn& a, const WinColumn& b) {
            return a.col_id < b.col_id;
        });

    // ================================================================
    // 步骤 2: 高低位分桶 + 分层漏斗匹配
    // ================================================================
    const int SEARCH_WINDOW = 16;   // 桶内搜索视窗
    const int T_MAX = 8;            // 允许的最大冲突行数

    std::vector<bool> matched(n, false);
    std::vector<Pair1x2> pairs;

    // ---- Phase 0: 特征提取与分桶 ----
    std::vector<int> top_heavy, bottom_heavy, balanced;
    bucket_columns(columns, top_heavy, bottom_heavy, balanced);

    // ---- Phase 1: 跨极区高效匹配 (Top + Bottom) ----
    // 偏上列和偏下列天然在不同半区, 冲突概率低, 优先配对.
    for (int top_idx : top_heavy) {
        if (matched[top_idx]) continue;

        int best_j = find_best_in_bucket(columns, columns[top_idx].mask,
                                         bottom_heavy, matched,
                                         T_MAX, SEARCH_WINDOW);

        if (best_j != -1) {
            uint16_t and_val = columns[top_idx].mask & columns[best_j].mask;
            Pair1x2 p;
            p.col1 = columns[top_idx].col_id;
            p.col2 = columns[best_j].col_id;
            p.G1   = and_val;
            p.G2   = columns[top_idx].mask | columns[best_j].mask;
            p.is_fallback = false;
            pairs.push_back(p);

            matched[top_idx] = true;
            matched[best_j] = true;
        }
    }

    // ---- Phase 2: 剩余极区列与均匀列匹配 (Leftovers + Balanced) ----
    for (int idx : top_heavy) {
        if (matched[idx]) continue;

        int best_j = find_best_in_bucket(columns, columns[idx].mask,
                                         balanced, matched,
                                         T_MAX, SEARCH_WINDOW);

        if (best_j != -1) {
            uint16_t and_val = columns[idx].mask & columns[best_j].mask;
            Pair1x2 p;
            p.col1 = columns[idx].col_id;
            p.col2 = columns[best_j].col_id;
            p.G1   = and_val;
            p.G2   = columns[idx].mask | columns[best_j].mask;
            p.is_fallback = false;
            pairs.push_back(p);

            matched[idx] = true;
            matched[best_j] = true;
        }
    }
    for (int idx : bottom_heavy) {
        if (matched[idx]) continue;

        int best_j = find_best_in_bucket(columns, columns[idx].mask,
                                         balanced, matched,
                                         T_MAX, SEARCH_WINDOW);

        if (best_j != -1) {
            uint16_t and_val = columns[idx].mask & columns[best_j].mask;
            Pair1x2 p;
            p.col1 = columns[idx].col_id;
            p.col2 = columns[best_j].col_id;
            p.G1   = and_val;
            p.G2   = columns[idx].mask | columns[best_j].mask;
            p.is_fallback = false;
            pairs.push_back(p);

            matched[idx] = true;
            matched[best_j] = true;
        }
    }

    // ---- Phase 3: 均匀列内部消化 (Balanced + Balanced) ----
    for (int bi : balanced) {
        if (matched[bi]) continue;

        int best_j = -1;
        int min_cost = T_MAX + 1;
        int seen = 0;

        for (int bj : balanced) {
            if (bj == bi || matched[bj]) continue;

            int cost = __builtin_popcount(columns[bi].mask & columns[bj].mask);

            if (cost == 0) { best_j = bj; break; }
            if (cost <= T_MAX && cost < min_cost) {
                min_cost = cost;
                best_j = bj;
            }
            if (++seen >= SEARCH_WINDOW) break;
        }

        if (best_j != -1) {
            uint16_t and_val = columns[bi].mask & columns[best_j].mask;
            Pair1x2 p;
            p.col1 = columns[bi].col_id;
            p.col2 = columns[best_j].col_id;
            p.G1   = and_val;
            p.G2   = columns[bi].mask | columns[best_j].mask;
            p.is_fallback = false;
            pairs.push_back(p);

            matched[bi] = true;
            matched[best_j] = true;
        }
    }

    // ---- Phase 4: 兜底处理 (Fallback) ----
    for (int i = 0; i < n; ++i) {
        if (matched[i]) continue;

        Pair1x2 p;
        p.col1 = columns[i].col_id;
        p.col2 = -1;
        p.G1   = 0;
        p.G2   = columns[i].mask;
        p.is_fallback = true;
        pairs.push_back(p);

        matched[i] = true;
        stats.fine_fb++;
    }

    // ================================================================
    // 步骤 3: 粗粒度局部滑动视窗匹配 (2:4 结构)
    // ================================================================
    const int GROUP_SEARCH_WINDOW = 16;

    int np = (int)pairs.size();
    std::vector<bool> pair_used(np, false);

    for (int i = 0; i < np; ++i) {
        if (pair_used[i]) continue;

        const Pair1x2& pa = pairs[i];
        bool found = false;
        int j_end = std::min(i + GROUP_SEARCH_WINDOW, np);

        for (int j = i + 1; j < j_end; ++j) {
            if (pair_used[j]) continue;

            const Pair1x2& pb = pairs[j];
            uint16_t Z = (pa.G1 & pb.G2) | (pa.G2 & pb.G1);

            if (Z == 0) {
                Group2x4 g;
                g.cols[0] = pa.col1;
                g.cols[1] = pa.col2;
                g.cols[2] = pb.col1;
                g.cols[3] = pb.col2;
                groups.push_back(g);

                pair_used[i] = true;
                pair_used[j] = true;
                found = true;
                break;
            }
        }

        if (!found) {
            Group2x4 g;
            g.cols[0] = pa.col1;
            g.cols[1] = pa.col2;
            g.cols[2] = -1;
            g.cols[3] = -1;
            g.zero_cols = 2;
            groups.push_back(g);

            pair_used[i] = true;
            stats.coarse_fb += 2;
        }
    }

    stats.num_groups += (int)groups.size();

    // ================================================================
    // 步骤 4: 元素级填充对齐 (Padding) — O(1) 计算
    //
    // 每个 2:4 Group 有 16 行 × 2 位置 = 32 个 slot.
    // fake_zeros = 32 - Σ(有效列在该窗口内的 nnz)
    // ================================================================
    for (const Group2x4& g : groups) {
        int group_nnz = 0;
        for (int c = 0; c < 4; ++c) {
            int col_id = g.cols[c];
            if (col_id < 0) continue;
            auto it = col_nnz.find(col_id);
            if (it != col_nnz.end()) group_nnz += it->second;
        }
        stats.fake_zeros += (32 - group_nnz);
    }

    return groups;
}


// ============================================================================
// 顶层入口: 对整个稀疏矩阵执行 2:4 结构化匹配
// ============================================================================
static MatchStats run_2to4_matching_bucket(
    const int*    row_ptr,
    const int*    col_ind,
    const float*  /*values*/,
    int           rows,
    int           cols,
    int           window_size)
{
    MatchStats stats;
    stats.total_rows = rows;
    stats.total_cols = cols;
    stats.total_nnz  = row_ptr[rows];
    stats.sparsity   = 1.0 - (double)stats.total_nnz / ((double)rows * cols);

    int num_windows = (rows + window_size - 1) / window_size;
    stats.num_row_panels = num_windows;

    for (int win = 0; win < num_windows; ++win) {
        int row_start = win * window_size;
        int row_end   = std::min(row_start + window_size, rows);
        int win_rows  = row_end - row_start;

        std::unordered_map<int, uint16_t> col_mask_of;
        col_mask_of.reserve(256);

        for (int r = row_start; r < row_end; ++r) {
            int local_row = r - row_start;
            uint16_t bit  = (uint16_t)1 << local_row;

            for (int e = row_ptr[r]; e < row_ptr[r + 1]; ++e) {
                int c = col_ind[e];
                col_mask_of[c] |= bit;
            }
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

        std::vector<Group2x4> win_groups =
            match_window_2to4_bucket(win_rows, columns, col_mask_of, stats);

        int N = (int)win_groups.size();
        int blocks = (N + 3) / 4;
        int pad_groups = blocks * 4 - N;
        stats.num_16x16_blocks += blocks;
        stats.block_padding_groups += pad_groups;
    }

    return stats;
}


// ============================================================================
// pybind11 接口
// ============================================================================
py::dict match_2to4_bucket_py(
    py::array_t<int,   py::array::c_style | py::array::forcecast> row_ptr_arr,
    py::array_t<int,   py::array::c_style | py::array::forcecast> col_ind_arr,
    py::array_t<float, py::array::c_style | py::array::forcecast> values_arr,
    int window_size = 16)
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
    for (int i = 0; i < nnz; ++i) {
        if (col_ind[i] >= cols) cols = col_ind[i] + 1;
    }

    if (window_size < 1 || window_size > 64) {
        throw std::runtime_error("window_size 必须在 [1, 64] 范围内");
    }
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("无效的矩阵维度");
    }

    MatchStats stats = run_2to4_matching_bucket(row_ptr, col_ind, values, rows, cols, window_size);

    // ---- C++ 层统计输出 ----
    printf("\n");
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║  2:4 结构化稀疏匹配统计 (Bucket 分桶)       ║\n");
    printf("╠══════════════════════════════════════════════╣\n");
    printf("║  矩阵形状:          %6d × %-6d          ║\n", stats.total_rows, stats.total_cols);
    printf("║  总非零元数:        %-14lld            ║\n", (long long)stats.total_nnz);
    printf("║  稀疏度 (Sparsity):   %8.4f%%             ║\n", stats.sparsity * 100.0);
    printf("╠══════════════════════════════════════════════╣\n");
    printf("║  [逻辑层]                                    ║\n");
    printf("║  16行 Row Panels:  %-8d                 ║\n", stats.num_row_panels);
    printf("║  16×4 基础 Group:  %-8d                 ║\n", stats.num_groups);
    printf("╠══════════════════════════════════════════════╣\n");
    printf("║  [硬件层 — m16n8k16 指令映射]               ║\n");
    printf("║  16×16 稀疏块 (Block): %-8d             ║\n", stats.num_16x16_blocks);
    printf("║  Block 对齐填充 Group:  %-8d             ║\n", stats.block_padding_groups);
    printf("╠══════════════════════════════════════════════╣\n");
    printf("║  [填充统计]                                  ║\n");
    printf("║  细粒度 Fallback:   %-8d (单列0)         ║\n", stats.fine_fb);
    printf("║  粗粒度 Fallback:   %-8d (整列0)         ║\n", stats.coarse_fb);
    printf("║  元素级假0填充:     %-8d                ║\n", stats.fake_zeros);
    printf("╚══════════════════════════════════════════════╝\n\n");

    py::dict result;
    result["total_rows"]          = stats.total_rows;
    result["total_cols"]          = stats.total_cols;
    result["total_nnz"]           = (long long)stats.total_nnz;
    result["sparsity"]            = stats.sparsity;
    result["num_row_panels"]      = stats.num_row_panels;
    result["num_groups"]          = stats.num_groups;
    result["num_16x16_blocks"]    = stats.num_16x16_blocks;
    result["block_padding_groups"] = stats.block_padding_groups;
    result["fine_fallback"]       = stats.fine_fb;
    result["coarse_fallback"]     = stats.coarse_fb;
    result["fake_zeros"]          = stats.fake_zeros;

    return result;
}


// ============================================================================
// 模块定义
// ============================================================================
PYBIND11_MODULE(matching_utils_bucket, m) {
    m.doc() = R"pbdoc(
        2:4 结构化稀疏匹配模块 (High-Low Bucketing Matching)

        核心算法:
          1. 16行窗口划分 → uint16_t 编码 nnz 分布
          2. Phase 0: 按高低 8 位 nnz 分布分桶 (top_heavy/bottom_heavy/balanced)
          3. Phase 1: 跨极区匹配 (top × bottom)
          4. Phase 2: 剩余极区 × balanced
          5. Phase 3: balanced 内部消化
          6. Phase 4: 全量兜底 Fallback
          7. 粗粒度滑动视窗配对 (2:4 Group)
          8. O(1) fake_zeros = 32 - group_nnz

        接口:
          match_2to4_bucket(row_ptr, col_ind, values, window_size=16) -> dict
    )pbdoc";

    m.def("match_2to4_bucket", &match_2to4_bucket_py,
          py::arg("row_ptr"),
          py::arg("col_ind"),
          py::arg("values"),
          py::arg("window_size") = 16,
          R"pbdoc(
对 CSR 格式稀疏矩阵执行 2:4 结构化稀疏匹配 (Bucket 分桶策略).

参数:
    row_ptr (np.ndarray):    CSR 行偏移, dtype=int32, 长度 rows+1
    col_ind (np.ndarray):    CSR 列索引, dtype=int32, 长度 nnz
    values (np.ndarray):     CSR 数值, dtype=float32, 长度 nnz
    window_size (int):       窗口行数, 默认 16

返回:
    dict: 统计字典 (字段与 match_2to4 相同)
          )pbdoc");
}
