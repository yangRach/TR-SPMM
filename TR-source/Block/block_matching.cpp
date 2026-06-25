/**  双指针
 * block_matching.cpp — 基于底层位运算的 2:4 结构化稀疏匹配算法
 *
 * 算法原理:
 *   将稀疏矩阵按 16 行划分为窗口 (16x16 大块), 通过位运算在窗口内对列进行
 *   1:2 (两列配对) 和 2:4 (四列成组) 的结构化匹配, 确保每组 4 列中每行最多
 *   2 个非零元, 符合 GPU 2:4 结构化稀疏的硬件格式要求。
 *
 * 暴露接口:
 *   matching_utils.match_2to4(row_ptr, col_ind, values, window_size=16)
 *
 * 依赖: pybind11 (Python 绑定), <cstdint> (uint16_t 位运算)
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

/**
 * 窗口内列信息 (16 行的 nnz 模式用 uint16_t 编码)
 *
 * 位编码规则: mask 的第 r 位为 1 表示该列在窗口内第 r 行有非零元。
 * nnz 通过 __builtin_popcount(mask) 在 O(1) 时间求得。
 */
struct WinColumn {
    int      col_id;   // 原始矩阵列索引
    uint16_t mask;     // 16-bit nnz 分布编码
    int      nnz;      // 窗口内非零元个数 (= popcount(mask))
};

/**
 * 步骤 2 输出: 1:2 配对 (两列组成一个 Pair)
 *
 * G1 = col1 & col2  (两列同时存在的行 → 冲突行)
 * G2 = col1 | col2  (两列的并集 → 有效行)
 */
struct Pair1x2 {
    int      col1;         // 第1列原始索引 (-1 表示虚拟全0列)
    int      col2;         // 第2列原始索引 (-1 表示虚拟全0列)
    uint16_t G1;           // 逻辑与: col1 & col2
    uint16_t G2;           // 逻辑或: col1 | col2
    bool     is_fallback;  // 是否触发细粒度 Fallback (col2 为虚拟0列)
};

/**
 * 步骤 3 输出: 2:4 分组 (四个列组成一个 Group)
 *
 * 每个 Group 的 4 列在每行最多有 2 个非零元。
 * 数组中的 -1 表示该位置为虚拟全0列。
 */
struct Group2x4 {
    int cols[4];        // 4 列原始索引
    int zero_cols = 0;  // 该 Group 中粗粒度 Fallback 贡献的全0列数
};

/**
 * 匹配统计信息 (同时作为 C++ 层输出和 Python 层返回)
 */
struct MatchStats {
    int       total_rows          = 0;
    int       total_cols          = 0;
    long long total_nnz           = 0;
    double    sparsity            = 0.0;
    int       num_row_panels      = 0;    // 16 行条带数 (Row Panel, 把矩阵按 16 行切分)
    int       num_groups          = 0;    // 16×4 基础结构组 (2:4 Group) 总数
    int       num_16x16_blocks    = 0;    // 最终打包出的 16×16 稀疏块 (Sparse Block), 4 Groups → 1 Block
    int       block_padding_groups = 0;   // 为对齐 16×16 Block 而额外填充的全0 Group 数
    int       fine_fb             = 0;    // 细粒度匹配 Fallback 次数 (插入单列0的次数)
    int       coarse_fb           = 0;    // 粗粒度匹配 Fallback 次数 (插入整列0的个数)
    int       fake_zeros          = 0;    // 元素级假0填充总数
};


// ============================================================================
// 核心算法: 单窗口 2:4 结构化匹配
//
// 参数:
//   win_rows    - 该窗口的实际行数 (通常为 16, 末尾窗口可能少于 16)
//   columns     - 窗口内所有候选列的向量 (已排序, 会被修改)
//   col_mask_of - 从列ID到 mask 的快速查找表 (只读)
//   stats       - [出参] 累加统计信息
//
// 返回: 该窗口内构建的 Group2x4 列表
// ============================================================================
static std::vector<Group2x4> match_window_2to4(
    int win_rows,
    std::vector<WinColumn>& columns,
    const std::unordered_map<int, uint16_t>& col_mask_of,
    MatchStats& stats)
{
    std::vector<Group2x4> groups;

    if (columns.empty()) return groups;

    int n = (int)columns.size();

    // ======================================================================
    // 步骤 1: 密度降序排序
    //
    // 使用 __builtin_popcount 作为排序依据 (等价于 columns[i].nnz),
    // 非零元多的列排在前面, 优先匹配。
    // ======================================================================
    std::sort(columns.begin(), columns.end(),
        [](const WinColumn& a, const WinColumn& b) {
            if (a.nnz != b.nnz) return a.nnz > b.nnz;
            return a.col_id < b.col_id;
        });

    // ======================================================================
    // 步骤 2: 细粒度双指针匹配 (1:2 结构) — 带局部候选视窗
    //
    // 升级策略:
    //   - 右指针 R 从最稀疏列 (末尾) 向左扫描
    //   - 不再"碰到 cost<=2 就立刻匹配", 而是收集最多 LOOKAHEAD_SIZE 个候选项
    //   - 从中选 conflict 最小的列; 若遇到 cost==0 的完美匹配则直接锁定
    //   - 避免了贪心匹配忽略左侧 0 冲突完美匹配的问题
    //
    // 关键位运算:
    //   - Col[L] & Col[R]: 找出两列在同一行都有非零元的行 (冲突行)
    //   - __builtin_popcount: O(1) 统计冲突行数
    // ======================================================================
    std::vector<bool> matched(n, false);
    std::vector<Pair1x2> pairs;

    const int LOOKAHEAD_SIZE = 2048;  // 局部候选视窗: 最多收集 2048 个候选列

    for (int L = 0; L < n; ++L) {
        if (matched[L]) continue;

        uint16_t mask_L = columns[L].mask;
        bool found = false;

        // ---- 局部候选视窗扫描 ----
        int best_R = -1;
        int min_cost = 999;
        int candidates_found = 0;

        for (int R = n - 1; R > L; --R) {
            if (matched[R]) continue;

            uint16_t and_val = mask_L & columns[R].mask;
            int cost = __builtin_popcount(and_val);

            if (cost == 0) {
                // 遇到 0 冲突完美匹配, 直接贪心锁定并提前终止扫描
                best_R = R;
                min_cost = 0;
                break;
            }

            if (cost <= 1) {  // T_max = 2
                if (cost < min_cost) {
                    min_cost = cost;
                    best_R = R;
                }
                candidates_found++;
                if (candidates_found >= LOOKAHEAD_SIZE) {
                    break;  // 候选窗口已满, 使用当前 best_R
                }
            }
        }

        // ---- 使用最佳候选构建 Pair ----
        if (best_R != -1) {
            uint16_t and_val = mask_L & columns[best_R].mask;
            Pair1x2 p;
            p.col1 = columns[L].col_id;
            p.col2 = columns[best_R].col_id;
            p.G1   = and_val;
            p.G2   = mask_L | columns[best_R].mask;
            p.is_fallback = false;
            pairs.push_back(p);

            matched[L] = true;
            matched[best_R] = true;
            found = true;
        }

        if (!found) {
            // Fallback: 右侧无可配对列 (窗口列数为奇数, 或所有右侧列冲突都 > 2)
            Pair1x2 p;
            p.col1 = columns[L].col_id;
            p.col2 = -1;               // 虚拟全0列
            p.G1   = 0;                // 全0列的 AND 结果为 0
            p.G2   = mask_L;           // OR 结果就是 L 列本身
            p.is_fallback = true;
            pairs.push_back(p);

            matched[L] = true;
            stats.fine_fb++;           // 统计细粒度 Fallback (1列0)
        }
    }

    // ======================================================================
    // 步骤 3: 粗粒度迭代匹配 (2:4 结构) — 收缩双指针
    //
    // 将步骤2产生的 Pair 两两配对成 4 列 Group。
    //
    // 升级策略:
    //   - pL 从密集端 (左) 开始, pR 从稀疏端 (右) 向中间逼近
    //   - 密集 Pair 优先与稀疏 Pair 配对, 利用"异或互补"最大化 Z_conflict==0 的概率
    //   - 替代旧版"密集优先配密集"的贪心, 显著降低粗粒度 Fallback
    //
    // 逻辑门冲突检测:
    //   对于 Pair A (G1, G2) 和 Pair B (G3, G4):
    //     Z_conflict = (G1 & G4) | (G2 & G3)
    //
    //   解释:
    //     - G1 = A中两列同时存在的行 (A内部已占2个nnz的行)
    //     - G4 = B中至少一列存在的行 (B存在的行)
    //     - (G1 & G4) 检测: A已占2个nnz的行是否在B中也存在 → 会超2
    //     - (G2 & G3) 同理检测 B已占2个nnz的行是否在A中也存在
    //     - Z_conflict == 0 ⇔ 合并后每行 ≤ 2 nnz, 符合2:4要求
    //
    // Fallback: 无法匹配则填充两列全0。
    // ======================================================================
    int np = (int)pairs.size();
    std::vector<bool> pair_used(np, false);

    // 按 Pair 密度 (G2 的 popcnt) 降序排列, 密集在前 (供双指针使用)
    std::vector<int> order(np);
    for (int i = 0; i < np; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return __builtin_popcount(pairs[a].G1) > __builtin_popcount(pairs[b].G1);
    });

    // 收缩双指针: pL 从密集端(左), pR 从稀疏端(右)向中间逼近
    for (int pL = 0; pL < np; ++pL) {
        int i = order[pL];
        if (pair_used[i]) continue;

        const Pair1x2& pa = pairs[i];
        bool found = false;

        // pR 从最稀疏端(右)向左逼近, 密集配稀疏 → 最大化兼容概率
        for (int pR = np - 1; pR > pL; --pR) {
            int j = order[pR];
            if (pair_used[j]) continue;

            const Pair1x2& pb = pairs[j];

            // 三层逻辑门冲突检测 (核心位运算)
            uint16_t Z = (pa.G1 & pb.G2) | (pa.G2 & pb.G1);

            if (Z == 0) {
                // 完美符合 2:4 要求!
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
            // Fallback: 该 Pair 无法匹配 → 填充两列全0
            Group2x4 g;
            g.cols[0] = pa.col1;
            g.cols[1] = pa.col2;
            g.cols[2] = -1;    // 第1列全0
            g.cols[3] = -1;    // 第2列全0
            g.zero_cols = 2;
            groups.push_back(g);

            pair_used[i] = true;
            stats.coarse_fb += 2;  // 粗粒度 Fallback: 插入 2 列0
        }
    }

    stats.num_groups += (int)groups.size();

    // ======================================================================
    // 步骤 4: 元素级填充对齐 (Padding)
    //
    // 对于每个 16x4 的 Group, 逐行检查:
    //   若该行非零元 < 2 个 → 需要插入 "假0" (Fake Zero) 填满 2 个位置
    //
    // 假0 数量 = Σ_group Σ_row max(0, 2 - nnz_in_row_of_group)
    // ======================================================================
    for (const Group2x4& g : groups) {
        // 逐行统计该 Group 4 列中的非零元数
        for (int r = 0; r < win_rows; ++r) {
            uint16_t row_bit = (uint16_t)1 << r;
            int count = 0;

            for (int c = 0; c < 4; ++c) {
                int col_id = g.cols[c];
                if (col_id < 0) continue;  // 跳过虚拟全0列
                auto it = col_mask_of.find(col_id);
                if (it != col_mask_of.end() && (it->second & row_bit)) {
                    count++;
                }
            }

            // 不足 2 个非零元则需要填充假0
            if (count < 2) {
                stats.fake_zeros += (2 - count);
            }
        }
    }

    return groups;
}


// ============================================================================
// 顶层入口: 对整个稀疏矩阵执行 2:4 结构化匹配
//
// 参数:
//   row_ptr     - CSR 行偏移 (长度 rows+1)
//   col_ind     - CSR 列索引 (长度 nnz)
//   values      - CSR 数值 (长度 nnz, 当前仅用于完整性, 不参与结构匹配)
//   rows        - 矩阵行数
//   cols        - 矩阵列数 (从列索引推算)
//   window_size - 窗口行数 (默认16)
//
// 返回: 完整匹配统计信息
// ============================================================================
static MatchStats run_2to4_matching(
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

    // ---- 逐窗口处理 ----
    for (int win = 0; win < num_windows; ++win) {
        int row_start = win * window_size;
        int row_end   = std::min(row_start + window_size, rows);
        int win_rows  = row_end - row_start;

        // ---------------------------------------------------------------
        // 步骤 0: 窗口划分与编码
        //
        // 遍历窗口内所有行, 对每个出现的列用 uint16_t 编码其 nnz 分布:
        //   mask 的第 r 位 = 1 表示该列在窗口第 r 行存在非零元
        //
        // 用 unordered_map 做去重: 同一列在不同行出现时只更新对应的 bit 位
        // ---------------------------------------------------------------
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

        // 转移到 vector 中便于排序
        std::vector<WinColumn> columns;
        columns.reserve(col_mask_of.size());
        for (const auto& kv : col_mask_of) {
            WinColumn wc;
            wc.col_id = kv.first;
            wc.mask   = kv.second;
            wc.nnz    = __builtin_popcount(kv.second);  // O(1) 位计数
            columns.push_back(wc);
        }

        if (columns.empty()) continue;

        // 调用单窗口匹配 (步骤 1~4 在其内部完成)
        std::vector<Group2x4> win_groups =
            match_window_2to4(win_rows, columns, col_mask_of, stats);

        // ---- 硬件映射: 16×4 Group → 16×16 Sparse Block 打包 ----
        // m16n8k16 指令的输入是 16×16 的稀疏块, 每个 Block = 4 个 Group 横向拼接。
        // 每 4 个 Group 打包成 1 个 Block, 不足 4 个则用全0 Group 补齐。
        int N = (int)win_groups.size();
        int blocks = (N + 3) / 4;               // ceil(N / 4)
        int pad_groups = blocks * 4 - N;        // 对齐 Block 所需的空白 Group 数
        stats.num_16x16_blocks += blocks;
        stats.block_padding_groups += pad_groups;
    }

    return stats;
}


// ============================================================================
// pybind11 接口: Python → C++ 的桥梁
//
// 接收 NumPy 数组 (CSR 格式), 调用核心匹配算法, 返回统计字典。
// C++ 层同时通过 printf 输出可读的统计表格。
// ============================================================================
py::dict match_2to4_py(
    py::array_t<int,   py::array::c_style | py::array::forcecast> row_ptr_arr,
    py::array_t<int,   py::array::c_style | py::array::forcecast> col_ind_arr,
    py::array_t<float, py::array::c_style | py::array::forcecast> values_arr,
    int window_size = 16)
{
    // 获取 NumPy 数组的底层指针
    py::buffer_info row_buf = row_ptr_arr.request();
    py::buffer_info col_buf = col_ind_arr.request();
    py::buffer_info val_buf = values_arr.request();

    const int*   row_ptr = static_cast<const int*>(row_buf.ptr);
    const int*   col_ind = static_cast<const int*>(col_buf.ptr);
    const float* values  = static_cast<const float*>(val_buf.ptr);

    // 从 CSR 元数据推算矩阵维度
    int rows = (int)row_buf.shape[0] - 1;
    int nnz  = (int)col_buf.shape[0];

    // 推算列数: 列索引最大值 + 1
    int cols = 0;
    for (int i = 0; i < nnz; ++i) {
        if (col_ind[i] >= cols) cols = col_ind[i] + 1;
    }

    // 参数合法性检查
    if (window_size < 1 || window_size > 64) {
        throw std::runtime_error("window_size 必须在 [1, 64] 范围内");
    }
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("无效的矩阵维度");
    }

    // 执行匹配
    MatchStats stats = run_2to4_matching(row_ptr, col_ind, values, rows, cols, window_size);

    // ---- C++ 层输出统计 (printf) ----
    printf("\n");
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║     2:4 结构化稀疏匹配统计 (C++ 层)         ║\n");
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

    // ---- 构造 Python 返回字典 ----
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
PYBIND11_MODULE(matching_utils, m) {
    m.doc() = R"pbdoc(
        2:4 结构化稀疏匹配模块 (基于底层位运算)

        核心算法:
          1. 16行窗口划分 → uint16_t 编码 nnz 分布
          2. __builtin_popcount 密度降序
          3. 双指针细粒度匹配 (1:2 Pair) + 冲突检测 (popcnt(AND) <= 2)
          4. 粗粒度迭代匹配 (2:4 Group) + 逻辑门约束 (Z_conflict == 0)
          5. 元素级假0对齐填充

        接口:
          match_2to4(row_ptr, col_ind, values, window_size=16) -> dict
    )pbdoc";

    m.def("match_2to4", &match_2to4_py,
          py::arg("row_ptr"),
          py::arg("col_ind"),
          py::arg("values"),
          py::arg("window_size") = 16,
          R"pbdoc(
对 CSR 格式稀疏矩阵执行 2:4 结构化稀疏匹配。

参数:
    row_ptr (np.ndarray):    CSR 行偏移, dtype=int32, 长度 rows+1
    col_ind (np.ndarray):    CSR 列索引, dtype=int32, 长度 nnz
    values (np.ndarray):     CSR 数值, dtype=float32, 长度 nnz
    window_size (int):       窗口行数, 默认 16

返回:
    dict: 包含以下键的统计字典
        - total_rows, total_cols, total_nnz, sparsity
        - num_row_panels      (16行 Row Panel 条带数)
        - num_groups          (16×4 基础 2:4 Group 数)
        - num_16x16_blocks    (硬件 16×16 稀疏块数, 4 Group → 1 Block)
        - block_padding_groups (对齐 Block 填充的全0 Group 数)
        - fine_fallback       (细粒度 Fallback 插入单列0次数)
        - coarse_fallback     (粗粒度 Fallback 插入整列0个数)
        - fake_zeros          (元素级假0填充总数)
          )pbdoc");
<<<<<<< HEAD
}
=======
}
>>>>>>> upstream/main
