/**
 * block_matching_hash.cpp
 *   基于 Fast-Path 反码哈希与 Slow-Path 局部扫描的混合匹配算法 (Ablation Study)
 *
 * 与 block_matching.cpp 的区别仅在 Step 2:
 *   - Phase 0: 构建 mask → column_index 的哈希链表 (头插法)
 *   - Phase 1 (Fast-Path): 对每列计算 ideal = ~mask & 0xFFFF, O(1) 哈希查找
 *     完美 0 冲突匹配, 找到即锁定
 *   - Phase 2 (Slow-Path): 未能完美匹配的剩余列, 降级为 32 列滑动视窗扫描
 *
 * 哈希表重置: 使用版本号数组 (version array) 技术, 每个 window 仅 current_version++
 *   无需 memset 65536 个条目, 实现 O(1) 清空
 *
 * 暴露接口:
 *   matching_utils_hash.match_2to4_hash(row_ptr, col_ind, values, window_size=16)
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
// 数据结构 (与 block_matching.cpp 完全一致)
// ============================================================================

struct WinColumn {
    int      col_id;
    uint16_t mask;
    int      nnz;
};

struct Pair1x2 {
    int      col1;
    int      col2;
    uint16_t G1;
    uint16_t G2;
    bool     is_fallback;
    int      nnz = 0;      // Pair 包含的实际非零元总数 (供 O(1) fake_zeros)
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
// O(1) 哈希表重置: 版本号数组技术
//
// head[mask] 存储该 mask 对应链表的第一个节点索引.
// head_version[mask] 存储 head[mask] 的有效版本.
// 每个新 window 只需 current_version++, 无需 memset 65536 个条目.
// ============================================================================
// thread_local: 每个 OpenMP 线程独立哈希表, 避免多线程竞争
static thread_local int  s_head[65536];
static thread_local int  s_head_version[65536] = {0};
static thread_local int  s_current_version = 0;

inline int hash_get_head(uint16_t mask) {
    if (s_head_version[mask] != s_current_version) {
        s_head_version[mask] = s_current_version;
        s_head[mask] = -1;
    }
    return s_head[mask];
}

inline void hash_insert(uint16_t mask, int col_idx, std::vector<int>& next_node) {
    if (s_head_version[mask] != s_current_version) {
        s_head_version[mask] = s_current_version;
        s_head[mask] = -1;
    }
    next_node[col_idx] = s_head[mask];   // 头插法
    s_head[mask] = col_idx;
}


// ============================================================================
// 核心算法: 单窗口 2:4 结构化匹配 (Hash-based Step 2)
// ============================================================================
static std::vector<Group2x4> match_window_2to4_hash(
    int win_rows,
    std::vector<WinColumn>& columns,
    const std::unordered_map<int, uint16_t>& col_mask_of,
    MatchStats& stats)
{
    std::vector<Group2x4> groups;

    if (columns.empty()) return groups;

    int n = (int)columns.size();

    // ================================================================
    // 步骤 1: 按原始列索引排序 (保持空间局部性)
    // ================================================================
    std::sort(columns.begin(), columns.end(),
        [](const WinColumn& a, const WinColumn& b) {
            return a.col_id < b.col_id;
        });

    // ================================================================
    // 步骤 2: Fast-Path 反码哈希 + Slow-Path 局部扫描
    // ================================================================
    std::vector<bool> matched(n, false);
    std::vector<Pair1x2> pairs;

    // ---- O(1) 哈希表初始化: 版本号 +1 即清空 ----
    s_current_version++;
    std::vector<int> next_node(n, -1);  // 每列最多在一条链表中

    // ---- Phase 0: 构建哈希链表 (mask → column_index) ----
    for (int i = 0; i < n; ++i) {
        hash_insert(columns[i].mask, i, next_node);
    }

    // ---- Phase 1: Fast-Path 完美反码收割 ----
    // 对每列计算 ideal = ~mask & 0xFFFF, 在哈希表中 O(1) 查找.
    // 找到的列一定与其 0 冲突 (ideal & mask == 0), 直接锁定.
    for (int i = 0; i < n; ++i) {
        if (matched[i]) continue;

        uint16_t ideal = (~columns[i].mask) & 0xFFFF;

        for (int j = hash_get_head(ideal); j != -1; j = next_node[j]) {
            if (!matched[j] && j != i) {
                // 完美 0 冲突: ideal & mask == 0 恒成立
                Pair1x2 p;
                p.col1 = columns[i].col_id;
                p.col2 = columns[j].col_id;
                p.G1   = 0;                          // AND == 0 保证
                p.G2   = columns[i].mask | columns[j].mask;
                p.is_fallback = false;
                pairs.push_back(p);

                matched[i] = true;
                matched[j] = true;
                break;
            }
        }
    }

    // ---- Phase 2: Slow-Path 容错扫描 ----
    // 对 Phase 1 未匹配的剩余列, 降级为局部滑动视窗 (32 列).
    const int SEARCH_WINDOW = 32;
    const int T_MAX = 8;

    for (int i = 0; i < n; ++i) {
        if (matched[i]) continue;

        uint16_t mask_i = columns[i].mask;
        int best_j = -1;
        int min_cost = T_MAX + 1;
        int j_end = std::min(i + 1 + SEARCH_WINDOW, n);  // Fix: +1 修复 off-by-one

        for (int j = i + 1; j < j_end; ++j) {
            if (matched[j]) continue;

            uint16_t and_val = mask_i & columns[j].mask;
            int cost = __builtin_popcount(and_val);

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

        if (best_j != -1) {
            uint16_t and_val = mask_i & columns[best_j].mask;
            Pair1x2 p;
            p.col1 = columns[i].col_id;
            p.col2 = columns[best_j].col_id;
            p.G1   = and_val;
            p.G2   = mask_i | columns[best_j].mask;
            p.is_fallback = false;
            p.nnz   = columns[i].nnz + columns[best_j].nnz;
            pairs.push_back(p);

            matched[i] = true;
            matched[best_j] = true;
        } else {
            Pair1x2 p;
            p.col1 = columns[i].col_id;
            p.col2 = -1;
            p.G1   = 0;
            p.G2   = mask_i;
            p.is_fallback = true;
            p.nnz   = columns[i].nnz;
            pairs.push_back(p);

            matched[i] = true;
            stats.fine_fb++;
        }
    }

    // ================================================================
    // 步骤 3: 粗粒度局部滑动视窗匹配 (2:4 结构)
    // 与 block_matching.cpp 完全一致
    // ================================================================
    const int GROUP_SEARCH_WINDOW = 1024;

    int np = (int)pairs.size();
    std::vector<bool> pair_used(np, false);

    for (int i = 0; i < np; ++i) {
        if (pair_used[i]) continue;

        const Pair1x2& pa = pairs[i];
        bool found = false;
        int j_end = std::min(i + 1 + GROUP_SEARCH_WINDOW, np);  // Fix: +1 修复 off-by-one

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

                // O(1) fake_zeros
                stats.fake_zeros += (win_rows * 2) - (pa.nnz + pb.nnz);

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

            // O(1) fake_zeros: Fallback Group 仅有 pa 的 nnz
            stats.fake_zeros += (win_rows * 2) - pa.nnz;

            pair_used[i] = true;
            stats.coarse_fb += 2;
        }
    }

    stats.num_groups += (int)groups.size();

    return groups;
}


// ============================================================================
// 顶层入口: 对整个稀疏矩阵执行 2:4 结构化匹配
// 与 block_matching.cpp 完全一致
// ============================================================================
static MatchStats run_2to4_matching_hash(
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

    // ---- 逐窗口处理 (OpenMP 并行) ----
    // 哈希表已声明为 thread_local, 每个线程独立, 无竞争.
    #pragma omp parallel for schedule(dynamic, 16)
    for (int win = 0; win < num_windows; ++win) {
        MatchStats local_stats;

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

        if (columns.empty()) {
            #pragma omp critical
            { /* nothing to merge */ }
            continue;
        }

        std::vector<Group2x4> win_groups =
            match_window_2to4_hash(win_rows, columns, col_mask_of, local_stats);

        int N = (int)win_groups.size();
        int blocks = (N + 3) / 4;
        int pad_groups = blocks * 4 - N;
        local_stats.num_16x16_blocks += blocks;
        local_stats.block_padding_groups += pad_groups;

        #pragma omp critical
        {
            stats.num_groups          += local_stats.num_groups;
            stats.num_16x16_blocks    += local_stats.num_16x16_blocks;
            stats.block_padding_groups += local_stats.block_padding_groups;
            stats.fine_fb             += local_stats.fine_fb;
            stats.coarse_fb           += local_stats.coarse_fb;
            stats.fake_zeros          += local_stats.fake_zeros;
        }
    }

    return stats;
}


// ============================================================================
// pybind11 接口
// ============================================================================
py::dict match_2to4_hash_py(
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

    MatchStats stats = run_2to4_matching_hash(row_ptr, col_ind, values, rows, cols, window_size);

    // ---- C++ 层统计输出 ----
    printf("\n");
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║  2:4 结构化稀疏匹配统计 (Hash-based 混合)   ║\n");
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
PYBIND11_MODULE(matching_utils_hash, m) {
    m.doc() = R"pbdoc(
        2:4 结构化稀疏匹配模块 (Fast-Path 反码哈希 + Slow-Path 局部扫描)

        核心算法:
          1. 16行窗口划分 → uint16_t 编码 nnz 分布
          2. Phase 0: mask → column_index 哈希链表
          3. Phase 1: Fast-Path ideal = ~mask & 0xFFFF, O(1) 完美 0 冲突匹配
          4. Phase 2: Slow-Path 32 列滑动视窗, cost <= 2 容错匹配
          5. 粗粒度滑动视窗配对 (2:4 Group) + Z_conflict == 0 检测
          6. 元素级假0对齐填充

        接口:
          match_2to4_hash(row_ptr, col_ind, values, window_size=16) -> dict
    )pbdoc";

    m.def("match_2to4_hash", &match_2to4_hash_py,
          py::arg("row_ptr"),
          py::arg("col_ind"),
          py::arg("values"),
          py::arg("window_size") = 16,
          R"pbdoc(
对 CSR 格式稀疏矩阵执行 2:4 结构化稀疏匹配 (Hash-based 混合策略).

参数:
    row_ptr (np.ndarray):    CSR 行偏移, dtype=int32, 长度 rows+1
    col_ind (np.ndarray):    CSR 列索引, dtype=int32, 长度 nnz
    values (np.ndarray):     CSR 数值, dtype=float32, 长度 nnz
    window_size (int):       窗口行数, 默认 16

返回:
    dict: 统计字典 (字段与 match_2to4 相同)
          )pbdoc");
}
