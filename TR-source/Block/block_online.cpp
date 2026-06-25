#include <torch/extension.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// ============================================================================
// Data Structures
// ============================================================================

// 每列在 16 行窗口内的非零元信息
struct OnlineColInfo {
    int count = 0;          // 非零元总数
    uint32_t vec = 0;       // 2-bit 编码: 每 2-bit 代表一行的非零元个数(0-2), 16行=32bit
    int col_id = 0;         // 原始列 ID
    std::vector<std::pair<int, int>> rows; // (local_row, csr_edge_id)
};

// 由 1~2 列组成的 Unit (兼容的相邻列打包)
struct SptcUnit {
    int col1 = -1;          // 第一列 ID (始终有效)
    int col2 = -1;          // 第二列 ID (-1 表示单列 Unit)
    uint32_t vec = 0;       // 合并后的 2-bit 编码
    int nnz = 0;            // 总非零元数
    bool matched = false;   // 是否已在匹配中配对
};

// CUDA 行数据
struct OnlineCudaPart {
    int c_row_offset = 0;
    int c_row = 0;
    int c_atomic = 0;
    std::vector<int> c_column;
    std::vector<float> c_value;
    bool cuda_flag = false;
    bool cuda_residue_flag = false;
};

struct OnlineTcuPart {
    std::vector<float> t_value;
    std::vector<int> t_column;
    std::vector<int> row;
    std::vector<int> t_block_offset;
    std::vector<long> t_binary;
    std::vector<int> window;
    std::vector<int> atomic;
    bool tcu_flag = false;
};

struct OnlinePackResult {
    int nnz = 0;
    int slots = 0;
    std::vector<std::vector<int>> groups;
};

struct OnlineCudaWork {
    int large = 0;
    int small = 0;
};

// ============================================================================
// O(1) 位运算冲突检测 (来自 SC'25 MP-SpMM)
//
// 每列用 uint32_t vec 编码: bits (2r, 2r+1) 表示第 r 行的非零元个数 (0-2)
// 两个 Unit 可以合并当且仅当每行的非零元总数不超过 2
//
// 检测方法:
//   a 和 b 的按位或 c = a | b 得到每行至少有的非零元个数
//   (c << 1) & c: 在 bit 1,3,5,... 检测 "2 个或以上" → 标记 bit 1,3,5,...
//   a & b: 两列在同一行都有非零元 → 在 bit 0,2,4,... 标记两列共有的位置
//   ((a & b) | ((c << 1) & c)) & 0xaaaaaaaa : 检测是否有任何行的非零元 > 2
//     如果结果非零 → 冲突 (某行超过 2 个非零元)
//     如果结果为零 → 可以合并
// ============================================================================
static constexpr uint32_t SPTC_CONFLICT_MASK = 0xaaaaaaaa;

inline bool units_match_check(uint32_t a, uint32_t b) {
    const uint32_t c = a | b;
    return (((a & b) | ((c << 1) & c)) & SPTC_CONFLICT_MASK) == 0;
}

inline bool units_match_check(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    const uint32_t ab = a | b;
    const uint32_t cd = c | d;
    const uint32_t ab_cd = ab | cd;
    const uint32_t conflict_ab = ((a & b) | ((ab << 1) & ab)) & SPTC_CONFLICT_MASK;
    const uint32_t conflict_cd = ((c & d) | ((cd << 1) & cd)) & SPTC_CONFLICT_MASK;
    const uint32_t conflict_cross = ((ab & cd) | ((ab_cd << 1) & ab_cd)) & SPTC_CONFLICT_MASK;
    return (conflict_ab | conflict_cd | conflict_cross) == 0;
}

// ============================================================================
// Helpers
// ============================================================================

static OnlineCudaWork estimate_cuda_work_online(int nnz, int part_size, int short_size) {
    OnlineCudaWork work;
    if (nnz <= 0) return work;
    if (nnz <= short_size) { work.small = 1; return work; }
    if (nnz <= part_size) { work.large = 1; return work; }
    work.large = nnz / part_size;
    int residue = nnz % part_size;
    if (residue > 0) {
        if (residue <= short_size) work.small = 1;
        else work.large += 1;
    }
    return work;
}

static bool env_enabled_online(const char* name, bool default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr) return default_value;
    return std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 || std::strcmp(value, "TRUE") == 0;
}

static int env_int_online(const char* name, int default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr) return default_value;
    return std::atoi(value);
}

static float env_float_online(const char* name, float default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr) return default_value;
    return std::atof(value);
}

// ============================================================================
// 安全的 Tensor 构造函数 (torch::empty + memcpy, 避免 from_blob 悬空指针)
// ============================================================================

static torch::Tensor tensor_from_int_vector(const std::vector<int>& values) {
    auto t = torch::empty({(long)values.size()}, torch::kInt32);
    if (!values.empty()) std::memcpy(t.data_ptr(), values.data(), values.size() * sizeof(int));
    return t;
}
static torch::Tensor tensor_from_long_vector(const std::vector<long>& values) {
    auto t = torch::empty({(long)values.size()}, torch::kInt64);
    if (!values.empty()) std::memcpy(t.data_ptr(), values.data(), values.size() * sizeof(long));
    return t;
}
static torch::Tensor tensor_from_float_vector(const std::vector<float>& values) {
    auto t = torch::empty({(long)values.size()}, torch::kFloat32);
    if (!values.empty()) std::memcpy(t.data_ptr(), values.data(), values.size() * sizeof(float));
    return t;
}
static torch::Tensor tensor_from_int8_vector(const std::vector<int8_t>& values) {
    auto t = torch::empty({(long)values.size()}, torch::kInt8);
    if (!values.empty()) std::memcpy(t.data_ptr(), values.data(), values.size() * sizeof(int8_t));
    return t;
}
static torch::Tensor tensor_from_uint8_vector(const std::vector<uint8_t>& values) {
    auto t = torch::empty({(long)values.size()}, torch::kUInt8);
    if (!values.empty()) std::memcpy(t.data_ptr(), values.data(), values.size() * sizeof(uint8_t));
    return t;
}

static std::vector<float> merge_float_vectors(const std::vector<std::vector<float>>& vv) {
    std::vector<float> merged;
    for (const auto& vec : vv) merged.insert(merged.end(), vec.begin(), vec.end());
    return merged;
}
static std::vector<int> count_float_vectors(const std::vector<std::vector<float>>& vv) {
    std::vector<int> counts;
    counts.reserve(vv.size());
    for (const auto& vec : vv) counts.push_back((int)vec.size());
    return counts;
}

static uint16_t float_to_half_bits_online(float value) {
    c10::Half half_value(value);
    uint16_t bits = 0;
    std::memcpy(&bits, &half_value, sizeof(uint16_t));
    return bits;
}
static int pack_half2_int_online(float lo, float hi) {
    uint32_t bits = (uint32_t)float_to_half_bits_online(lo) |
                    ((uint32_t)float_to_half_bits_online(hi) << 16);
    return (int)bits;
}

static void load_sptc_row_chunk_cpu_online(
    const std::vector<float>& values, const std::vector<int8_t>& pos,
    int group_id, int group_limit, int row, int window,
    float& v0, float& v1, int& p0, int& p1) {
    v0 = 0.0f; v1 = 0.0f; p0 = 0; p1 = 1;
    if (group_id >= group_limit || row >= window) return;
    int base = (group_id * window + row) * 2;
    int pos0 = (int)pos[base], pos1 = (int)pos[base + 1];
    float val0 = values[base], val1 = values[base + 1];
    if (pos0 < 0 && pos1 < 0) return;
    if (pos0 >= 0 && pos1 >= 0) {
        if (pos0 <= pos1) { p0 = pos0; p1 = pos1; v0 = val0; v1 = val1; }
        else { p0 = pos1; p1 = pos0; v0 = val1; v1 = val0; }
        return;
    }
    int real_pos = pos0 >= 0 ? pos0 : pos1;
    float real_val = pos0 >= 0 ? val0 : val1;
    int dummy = real_pos == 0 ? 1 : 0;
    if (dummy < real_pos) { p0 = dummy; p1 = real_pos; v0 = 0.0f; v1 = real_val; }
    else { p0 = real_pos; p1 = dummy; v0 = real_val; v1 = 0.0f; }
}

// ============================================================================
// Stage 1: TC Funnel — 极稠密块进 Tensor Core (保持不变)
// ============================================================================
static OnlinePackResult select_tc_columns_online(
    const std::unordered_map<int, OnlineColInfo>& col_info,
    const std::unordered_set<int>& excluded,
    int window, int tc_wide, int tc_density, float tc_min_util,
    std::unordered_set<int>& tc_columns) {
    OnlinePackResult result;
    std::vector<int> candidates;
    for (const auto& item : col_info) {
        if (excluded.count(item.first) != 0) continue;
        if (item.second.count >= tc_density) candidates.push_back(item.first);
    }
    std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
        int ca = col_info.at(a).count, cb = col_info.at(b).count;
        if (ca != cb) return ca > cb;
        return a < b;
    });
    printf("TC候选列数量: %d\n", (int)candidates.size());
    for (int start = 0; start < (int)candidates.size(); start += tc_wide) {
        // printf("评估tc块候选列 [%d, %d)\n", start, std::min(start + tc_wide, (int)candidates.size()));
        int end = std::min(start + tc_wide, (int)candidates.size());
        int real_nnz = 0;
        std::vector<int> group;
        for (int i = start; i < end; ++i) {
            int col = candidates[i];
            real_nnz += col_info.at(col).count;
            group.push_back(col);
        }
        float util = (float)real_nnz / (float)(window * tc_wide);
        printf("util: %.2f%% (nnz=%d, slots=%d)  tc_min_util: %.2f%%\n", util * 100.0f, real_nnz, window * tc_wide, tc_min_util * 100.0f);
        if (util < tc_min_util) continue;
        std::sort(group.begin(), group.end());
        for (int col : group) tc_columns.insert(col);
        result.nnz += real_nnz;
        result.slots += window * tc_wide;
        result.groups.push_back(group);
        // printf("加入tc块");
    }
    return result;
}

// ============================================================================
// Stage 2: SPTC Funnel — 基于最大匹配的 Unit 打包 (全新实现)
//
// 算法流程 (来自 SC'25 MP-SpMM):
//   1. 每列编码为 uint32_t vec (2-bit per row)
//   2. 相邻兼容列打包为 Unit (1~2 列)
//   3. 建冲突图 + 最大度优先贪心匹配
//   4. 非匹配 Unit 零填充保留为 SPTC (不退回 CUDA!)
//   5. 只有 nnz 极低的"残渣"才退给 CUDA
// ============================================================================

// 构建 Unit: 将兼容的相邻列打包
static std::vector<SptcUnit> build_sptc_units(
    const std::unordered_map<int, OnlineColInfo>& col_info,
    const std::unordered_set<int>& excluded,
    int window) {
    std::vector<SptcUnit> units;

    // 收集候选列并按非零元数降序排序
    std::vector<int> candidates;
    for (const auto& item : col_info) {
        if (excluded.count(item.first) != 0) continue;
        candidates.push_back(item.first);
    }
    if (candidates.empty()) return units;

    std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
        int ca = col_info.at(a).count, cb = col_info.at(b).count;
        if (ca != cb) return ca > cb;
        return a < b;
    });

    // 贪心打包相邻兼容列
    std::vector<bool> used(candidates.size(), false);

    for (int i = 0; i < (int)candidates.size(); ++i) {
        if (used[i]) continue;
        int col_a = candidates[i];
        auto& info_a = col_info.at(col_a);
        used[i] = true;

        // 寻找兼容的配对列
        int best_j = -1;
        for (int j = i + 1; j < (int)candidates.size(); ++j) {
            if (used[j]) continue;
            int col_b = candidates[j];
            if (units_match_check(info_a.vec, col_info.at(col_b).vec)) {
                best_j = j;
                break; // 第一个兼容的即可 (已按密度排序)
            }
        }

        if (best_j >= 0) {
            int col_b = candidates[best_j];
            auto& info_b = col_info.at(col_b);
            used[best_j] = true;
            SptcUnit unit;
            unit.col1 = col_a;
            unit.col2 = col_b;
            unit.vec = info_a.vec | info_b.vec;
            unit.nnz = info_a.count + info_b.count;
            units.push_back(unit);
        } else {
            // 单列 Unit
            SptcUnit unit;
            unit.col1 = col_a;
            unit.col2 = -1;
            unit.vec = info_a.vec;
            unit.nnz = info_a.count;
            units.push_back(unit);
        }
    }
    return units;
}

// 最大度优先贪心匹配: 将 Unit 配对成 4 列 SPTC 组
// 匹配结果: match_pairs = [(unit_i, unit_j), ...]
static void match_sptc_units_max_degree(
    std::vector<SptcUnit>& units,
    std::vector<std::pair<int, int>>& match_pairs,
    std::vector<int>& unmatched) {
    int n = (int)units.size();
    if (n == 0) return;

    // 构建冲突图: graph[i] = 与 unit i 兼容的 unit 索引列表
    std::vector<std::vector<int>> graph(n);
    std::vector<int> degree(n, 0);

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (units_match_check(units[i].vec, units[j].vec)) {
                graph[i].push_back(j);
                graph[j].push_back(i);
                degree[i]++;
                degree[j]++;
            }
        }
    }

    // 最大度优先贪心匹配
    std::vector<bool> matched(n, false);

    // 按度数降序排序节点
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (degree[a] != degree[b]) return degree[a] > degree[b];
        return units[a].nnz > units[b].nnz;
    });

    for (int idx : order) {
        if (matched[idx]) continue;

        // 找度数最高的未匹配邻居
        int best_neighbor = -1;
        int best_degree = -1;
        for (int neighbor : graph[idx]) {
            if (matched[neighbor]) continue;
            if (degree[neighbor] > best_degree ||
                (degree[neighbor] == best_degree && units[neighbor].nnz > units[best_neighbor].nnz)) {
                best_neighbor = neighbor;
                best_degree = degree[neighbor];
            }
        }
        if (best_neighbor >= 0) {
            matched[idx] = true;
            matched[best_neighbor] = true;
            units[idx].matched = true;
            units[best_neighbor].matched = true;
            match_pairs.push_back({idx, best_neighbor});
        }
    }

    // 收集未匹配的 unit
    for (int i = 0; i < n; ++i) {
        if (!matched[i]) unmatched.push_back(i);
    }
}

// ============================================================================
// CUDA 行切分 (保持不变)
// ============================================================================
static void append_cuda_part_online(
    int row_id, int atomic, const std::vector<int>& cols, const std::vector<float>& vals,
    int offset, int count, bool residue,
    std::vector<OnlineCudaPart>& large, std::vector<OnlineCudaPart>& small) {
    OnlineCudaPart temp;
    temp.c_row_offset = count;
    temp.c_row = row_id;
    temp.c_atomic = atomic;
    temp.cuda_flag = !residue;
    temp.cuda_residue_flag = residue;
    for (int i = 0; i < count; ++i) {
        temp.c_column.push_back(cols[offset + i]);
        temp.c_value.push_back(vals[offset + i]);
    }
    if (residue) small.push_back(temp);
    else large.push_back(temp);
}

static void split_cuda_row_online(
    int row_id, int base_atomic, int part_size, int short_size,
    const std::vector<int>& cols, const std::vector<float>& vals,
    std::vector<OnlineCudaPart>& large, std::vector<OnlineCudaPart>& small) {
    int nnz = (int)cols.size();
    if (nnz <= 0) return;
    if (nnz <= short_size) {
        append_cuda_part_online(row_id, base_atomic, cols, vals, 0, nnz, true, large, small);
        return;
    }
    if (nnz <= part_size) {
        append_cuda_part_online(row_id, base_atomic, cols, vals, 0, nnz, false, large, small);
        return;
    }
    int offset = 0;
    int parts = nnz / part_size;
    for (int i = 0; i < parts; ++i) {
        append_cuda_part_online(row_id, 1, cols, vals, offset, part_size, false, large, small);
        offset += part_size;
    }
    int residue = nnz % part_size;
    if (residue > 0) {
        append_cuda_part_online(row_id, 1, cols, vals, offset, residue, residue <= short_size, large, small);
    }
}

// ============================================================================
// TCU part builder (保持不变, TC 跳过逻辑在内)
// ============================================================================
static OnlineTcuPart build_tcu_part_online(
    const std::unordered_map<int, OnlineColInfo>& col_info,
    const std::vector<std::vector<int>>& tc_groups,
    const int* row, const int* column, const float* values,
    int window_id, int row_start, int row_stop,
    int window, int wide, int part_size_t, bool has_cuda_fallback) {
    OnlineTcuPart part;
    int blocks = (int)tc_groups.size();
    if (blocks <= 0) return part;
    part.tcu_flag = true;

    std::unordered_map<int, int> colmap;
    for (int block = 0; block < blocks; ++block)
        for (int pos = 0; pos < (int)tc_groups[block].size(); ++pos)
            colmap[tc_groups[block][pos]] = block * wide + pos;

    struct SubBlock { std::vector<int> cols; std::vector<float> vals; uint64_t binary_lo = 0; uint64_t binary_hi = 0; };
    // TC block = 16×8, 直接匹配 m16n8k8 (M=16, K=8), 无需拆分
    // kernel 输出全部 16 行/子块, binary_lo 覆盖 rows 0-7, binary_hi 覆盖 rows 8-15
    std::vector<SubBlock> subs;

    for (int block = 0; block < blocks; ++block) {
        SubBlock sb;
        for (int pos = 0; pos < 8; ++pos) {
            int c = pos < (int)tc_groups[block].size() ? tc_groups[block][pos] : -1;
            sb.cols.push_back(c);
        }
        subs.push_back(sb);
    }

    for (int r = row_start; r < row_stop; ++r) {
        int local_row = r - row_start;  // 0..15, M-row in sub-block
        bool is_hi = local_row >= 8;
        int bin_row = is_hi ? (local_row - 8) : local_row;  // 0..7 in each binary
        for (int edge = row[r]; edge < row[r + 1]; ++edge) {
            auto found = colmap.find(column[edge]);
            if (found == colmap.end()) continue;
            int K_col = found->second % 8;  // 0..7, K-col within the 16×8 TC block
            int sub_idx = found->second / 8; // block index
            uint64_t one = 1;
            // binary bit = bin_row * 8 + K_col (row-major within 8-row group)
            if (is_hi) {
                subs[sub_idx].binary_hi |= one << (bin_row * 8 + K_col);
            } else {
                subs[sub_idx].binary_lo |= one << (bin_row * 8 + K_col);
            }
            subs[sub_idx].vals.push_back(values[edge]);
        }
    }

    int window_row_code = window_id * 16;  // 所有 16 行使用同一个 window_row_code
    printf("TC win=%d row_code=%d sub_blocks=%d\n", window_id, window_row_code, (int)subs.size());
    for (int si = 0; si < (int)subs.size(); ++si) {
        printf("  sub[%d] cols=%d vals=%d bin_lo=0x%lx bin_hi=0x%lx\n",
               si, (int)subs[si].cols.size(), (int)subs[si].vals.size(),
               subs[si].binary_lo, subs[si].binary_hi);
    }

    auto flush = [&](std::vector<SubBlock>& all_subs) {
        int total = (int)all_subs.size();
        if (total == 0) return;
        std::vector<long> binaries; std::vector<float> all_vals; std::vector<int> counts;
        for (auto& sb : all_subs) {
            part.t_column.insert(part.t_column.end(), sb.cols.begin(), sb.cols.end());
            binaries.push_back((long)sb.binary_lo);
            binaries.push_back((long)sb.binary_hi);
            counts.push_back((int)sb.vals.size());
            all_vals.insert(all_vals.end(), sb.vals.begin(), sb.vals.end());
        }
        part.t_binary.insert(part.t_binary.end(), binaries.begin(), binaries.end());
        part.t_value.insert(part.t_value.end(), all_vals.begin(), all_vals.end());
        if (total <= part_size_t) {
            part.row.push_back(total);
            part.t_block_offset.insert(part.t_block_offset.end(), counts.begin(), counts.end());
            part.atomic.push_back(has_cuda_fallback ? 1 : 0);
            part.window.push_back(window_row_code);
        } else {
            for (int p = 0; p < total / part_size_t; ++p) {
                part.row.push_back(part_size_t);
                for (int b = 0; b < part_size_t; ++b)
                    part.t_block_offset.push_back(counts[p * part_size_t + b]);
                part.window.push_back(window_row_code);
                part.atomic.push_back(1);
            }
            int residue = total % part_size_t;
            if (residue > 0) {
                part.row.push_back(residue);
                for (int b = 0; b < residue; ++b)
                    part.t_block_offset.push_back(counts[total - residue + b]);
                part.window.push_back(window_row_code);
                part.atomic.push_back(1);
            }
        }
    };
    flush(subs);
    return part;
}

// ============================================================================
// 主入口: 三级漏斗路由
// ============================================================================
std::vector<torch::Tensor> block_sptc_2to4_online(
    torch::Tensor row1, torch::Tensor column1, torch::Tensor values1,
    int partSize_t, int partSize_c, int shortSize,
    int density, int window, int wide,
    float tc_min_util, float sptc_min_util, int sptc_threshold) {

    TORCH_CHECK(row1.dtype() == torch::kInt32 && column1.dtype() == torch::kInt32);
    TORCH_CHECK(values1.dtype() == torch::kFloat32);
    TORCH_CHECK(window > 0 && window <= 32 && wide > 0 && wide <= 32);

    auto start = std::chrono::high_resolution_clock::now();
    auto* row = row1.data_ptr<int>();
    auto* column = column1.data_ptr<int>();
    auto* values = values1.data_ptr<float>();
    int rows = row1.size(0) - 1;
    int num_windows = (rows + window - 1) / window;

    const int sptc_group = 4;
    const int sptc_per_row = 2;
    const bool skip_tc = env_enabled_online("LIBRA_SKIP_TC", false);

    // SPTC Unit Matching 的最小 nnz 阈值: 低于此值的 unit 退回 CUDA
    const int sptc_unit_min_nnz = env_int_online("LIBRA_SPTC_UNIT_MIN_NNZ", 2);

    std::map<int, OnlineTcuPart> res_t;
    std::map<int, std::vector<OnlineCudaPart>> res_c, res_c_residue;

    // SPTC 输出数据
    std::vector<int> sptc_group_offsets = {0};
    std::vector<int> sptc_window_row, sptc_columns;
    std::vector<uint8_t> sptc_row_masks;
    std::vector<float> sptc_values;
    std::vector<int8_t> sptc_col_positions;

    // 统计
    long long total_tc_nnz = 0, total_tc_groups = 0;
    long long total_sptc_nnz = 0, total_sptc_groups = 0;
    long long total_cuda_nnz = 0;

    for (int win = 0; win < num_windows; ++win) {
        int row_start = win * window;
        int row_stop = std::min(row_start + window, rows);

        // 构建 col_info: uint32_t vec + 传统信息
        std::unordered_map<int, OnlineColInfo> col_info;
        for (int r = row_start; r < row_stop; ++r) {
            int local_row = r - row_start;
            std::unordered_set<int> seen;
            for (int edge = row[r]; edge < row[r + 1]; ++edge) {
                int col = column[edge];
                if (seen.count(col)) continue;
                seen.insert(col);
                auto& info = col_info[col];
                info.col_id = col;
                info.count++;
                info.vec += (uint32_t)1 << (2 * local_row);
                info.rows.emplace_back(local_row, edge);
            }
        }

        // ===== Stage 1: TC Funnel =====
        std::unordered_set<int> tc_column_set;
        OnlinePackResult tc;
        if (!skip_tc) {
            // printf("进入tc块构建");
            std::unordered_set<int> no_exclude;
            tc = select_tc_columns_online(col_info, no_exclude, window, 8, density, tc_min_util, tc_column_set);
            total_tc_nnz += tc.nnz;
            total_tc_groups += (long long)tc.groups.size();
        }

        // ===== Stage 2: SPTC Funnel (基于 Unit 匹配) =====
        std::unordered_set<int> sptc_column_set;

        // Step 1: 构建 Unit
        auto units = build_sptc_units(col_info, tc_column_set, window);

        // Step 2: 最大匹配
        std::vector<std::pair<int, int>> match_pairs;
        std::vector<int> unmatched;
        match_sptc_units_max_degree(units, match_pairs, unmatched);

        // Step 3: 生成 SPTC groups
        // 匹配对 → 4 列 groups
        for (auto& p : match_pairs) {
            auto& ua = units[p.first];
            auto& ub = units[p.second];
            std::vector<int> group;
            group.push_back(ua.col1);
            if (ua.col2 >= 0) group.push_back(ua.col2);
            if (ub.col1 >= 0) group.push_back(ub.col1);
            if (ub.col2 >= 0) group.push_back(ub.col2);
            std::sort(group.begin(), group.end());

            for (int col : group) sptc_column_set.insert(col);
            total_sptc_nnz += ua.nnz + ub.nnz;
            total_sptc_groups++;

            sptc_window_row.push_back(win);
            // 补齐到 4 列
            while ((int)group.size() < sptc_group) group.push_back(-1);
            for (int pos = 0; pos < sptc_group; ++pos) sptc_columns.push_back(group[pos]);
        }

        // 未匹配 → 保留为 SPTC (零填充), 但过于稀疏的退回 CUDA
        for (int idx : unmatched) {
            auto& u = units[idx];
            if (u.nnz < sptc_unit_min_nnz) continue; // 太稀疏 → 退回 CUDA

            std::vector<int> group;
            group.push_back(u.col1);
            if (u.col2 >= 0) group.push_back(u.col2);
            std::sort(group.begin(), group.end());

            for (int col : group) sptc_column_set.insert(col);
            total_sptc_nnz += u.nnz;
            total_sptc_groups++;

            sptc_window_row.push_back(win);
            while ((int)group.size() < sptc_group) group.push_back(-1);
            for (int pos = 0; pos < sptc_group; ++pos) sptc_columns.push_back(group[pos]);
        }

        // Step 4: 生成 SPTC 元数据 (每行每组的 value + position)
        for (int g = sptc_group_offsets.back(); g < (int)sptc_window_row.size(); ++g) {
            // 该 group 的 4 列
            int base = g * sptc_group;
            std::vector<int> gcols;
            for (int p = 0; p < sptc_group; ++p) {
                int c = sptc_columns[base + p];
                if (c >= 0) gcols.push_back(c);
            }

            for (int local_row = 0; local_row < window; ++local_row) {
                uint8_t mask = 0;
                int slots = 0;
                float row_vals[2] = {0.0f, 0.0f};
                int8_t row_pos[2] = {-1, -1};
                for (int p = 0; p < (int)gcols.size(); ++p) {
                    int col = gcols[p];
                    auto& rows_for_col = col_info.at(col).rows;
                    auto found = std::find_if(rows_for_col.begin(), rows_for_col.end(),
                        [&](const auto& item) { return item.first == local_row; });
                    if (found == rows_for_col.end()) continue;
                    mask |= (uint8_t)(1 << p);
                    if (slots < sptc_per_row) {
                        row_vals[slots] = values[found->second];
                        row_pos[slots] = (int8_t)p;
                    }
                    slots++;
                }
                sptc_row_masks.push_back(mask);
                for (int s = 0; s < sptc_per_row; ++s) {
                    sptc_values.push_back(row_vals[s]);
                    sptc_col_positions.push_back(row_pos[s]);
                }
            }
        }
        sptc_group_offsets.push_back((int)sptc_window_row.size());

        // ===== Stage 3: CUDA Funnel =====
        bool has_cuda_fallback = false;
        long long win_cuda_nnz = 0;
        for (const auto& item : col_info) {
            if (tc_column_set.count(item.first) == 0 && sptc_column_set.count(item.first) == 0) {
                has_cuda_fallback = true;
                win_cuda_nnz += item.second.count;
            }
        }
        total_cuda_nnz += win_cuda_nnz;

        OnlineTcuPart tcu = build_tcu_part_online(
            col_info, tc.groups, row, column, values, win, row_start, row_stop,
            window, 8, partSize_t, has_cuda_fallback);
        res_t[win] = tcu;

        std::vector<OnlineCudaPart> large, small;
        int base_cuda_atomic = tc_column_set.empty() ? 0 : 1;
        for (int r = row_start; r < row_stop; ++r) {
            std::vector<int> c_cols; std::vector<float> c_vals;
            for (int edge = row[r]; edge < row[r + 1]; ++edge) {
                int col = column[edge];
                if (tc_column_set.count(col) || sptc_column_set.count(col)) continue;
                c_cols.push_back(col); c_vals.push_back(values[edge]);
            }
            split_cuda_row_online(r, base_cuda_atomic, partSize_c, shortSize, c_cols, c_vals, large, small);
        }
        res_c[win] = large;
        res_c_residue[win] = small;
    }

    // ===== 序列化 =====
    std::vector<int> t_rowkNew_offset = {0}, t_blockNew_offset = {0};
    std::vector<float> t_valueNew; std::vector<int> t_columnNew;
    std::vector<long> t_binaryNew; std::vector<int> t_window_rowNew, t_atomicNew;

    std::vector<int> c_rowNew_offset = {0}, c_rowNew;
    std::vector<int> c_atomicNew, c_columnNew;
    std::vector<float> c_valueNew;

    std::vector<int> c_rowNew_offset_residue = {0}, c_rowNew_residue;
    std::vector<int> c_atomicNew_residue, c_columnNew_residue;
    std::vector<float> c_valueNew_residue;

    for (const auto& pair : res_t) {
        if (!pair.second.tcu_flag) continue;
        for (int sub : pair.second.row) t_rowkNew_offset.push_back(t_rowkNew_offset.back() + sub);
        for (int sub : pair.second.t_block_offset) t_blockNew_offset.push_back(t_blockNew_offset.back() + sub);
        t_valueNew.insert(t_valueNew.end(), pair.second.t_value.begin(), pair.second.t_value.end());
        t_columnNew.insert(t_columnNew.end(), pair.second.t_column.begin(), pair.second.t_column.end());
        t_window_rowNew.insert(t_window_rowNew.end(), pair.second.window.begin(), pair.second.window.end());
        t_atomicNew.insert(t_atomicNew.end(), pair.second.atomic.begin(), pair.second.atomic.end());
        t_binaryNew.insert(t_binaryNew.end(), pair.second.t_binary.begin(), pair.second.t_binary.end());
    }
    for (const auto& pair : res_c) {
        for (const auto& sub : pair.second) {
            if (!sub.cuda_flag) continue;
            c_rowNew_offset.push_back(c_rowNew_offset.back() + sub.c_row_offset);
            c_rowNew.push_back(sub.c_row); c_atomicNew.push_back(sub.c_atomic);
            c_columnNew.insert(c_columnNew.end(), sub.c_column.begin(), sub.c_column.end());
            c_valueNew.insert(c_valueNew.end(), sub.c_value.begin(), sub.c_value.end());
        }
    }
    for (const auto& pair : res_c_residue) {
        for (const auto& sub : pair.second) {
            if (!sub.cuda_residue_flag) continue;
            c_rowNew_offset_residue.push_back(c_rowNew_offset_residue.back() + sub.c_row_offset);
            c_rowNew_residue.push_back(sub.c_row); c_atomicNew_residue.push_back(sub.c_atomic);
            c_columnNew_residue.insert(c_columnNew_residue.end(), sub.c_column.begin(), sub.c_column.end());
            c_valueNew_residue.insert(c_valueNew_residue.end(), sub.c_value.begin(), sub.c_value.end());
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Three-Stage Funnel (SC'25 Unit-Matching): TC_nnz=" << total_tc_nnz
              << " TC_groups=" << total_tc_groups
              << " TC_subs=" << t_columnNew.size()
              << " SPTC_nnz=" << total_sptc_nnz
              << " SPTC_groups=" << total_sptc_groups
              << " CUDA_nnz=" << total_cuda_nnz << std::endl;

    // ===== SPTC 基础张量 =====
    auto s_columns = tensor_from_int_vector(sptc_columns).view({(long)sptc_window_row.size(), 4}).clone();
    auto s_masks = tensor_from_uint8_vector(sptc_row_masks).view({(long)sptc_window_row.size(), window}).clone();
    auto s_values = tensor_from_float_vector(sptc_values).view({(long)sptc_window_row.size(), window, 2}).clone();
    auto s_pos = tensor_from_int8_vector(sptc_col_positions).view({(long)sptc_window_row.size(), window, 2}).clone();
    auto s_window = tensor_from_int_vector(sptc_window_row);
    auto s_offsets = tensor_from_int_vector(sptc_group_offsets);

    // ===== Tile Packing for mma.sp =====
    std::vector<int> s_tile_window, s_tile_group, s_tile_columns, s_packed_a, s_packed_meta;
    std::vector<int> s_window_tile_offset = {0};  // 新kernel的窗口→tile范围映射

    for (int win = 0; win < num_windows; ++win) {
        int group_begin = sptc_group_offsets[win];
        int group_end = sptc_group_offsets[win + 1];
        for (int group_base = group_begin; group_base < group_end; group_base += 4) {
            s_tile_window.push_back(win);
            s_tile_group.push_back(group_base);
            for (int k = 0; k < 16; ++k) {
                int group_id = group_base + (k >> 2);
                int col_pos = k & 3;
                int col = group_id < group_end ? sptc_columns[group_id * 4 + col_pos] : -1;
                s_tile_columns.push_back(col);
            }
            for (int row_group = 0; row_group < 8; ++row_group) {
                float ar0[8], ar1[8];
                int meta_pos0[4], meta_pos1[4], meta_pos2[4], meta_pos3[4];
                for (int chunk = 0; chunk < 4; ++chunk) {
                    float v0, v1; int p0, p1;
                    load_sptc_row_chunk_cpu_online(
                        sptc_values, sptc_col_positions, group_base + chunk, group_end,
                        row_group, window, v0, v1, p0, p1);
                    ar0[chunk * 2] = v0; ar0[chunk * 2 + 1] = v1;
                    meta_pos0[chunk] = p0; meta_pos1[chunk] = p1;
                    load_sptc_row_chunk_cpu_online(
                        sptc_values, sptc_col_positions, group_base + chunk, group_end,
                        row_group + 8, window, v0, v1, p0, p1);
                    ar1[chunk * 2] = v0; ar1[chunk * 2 + 1] = v1;
                    meta_pos2[chunk] = p0; meta_pos3[chunk] = p1;
                }
                uint32_t meta = 0;
                for (int chunk = 0; chunk < 4; ++chunk) {
                    meta |= ((uint32_t)(meta_pos0[chunk] & 3) | ((uint32_t)(meta_pos1[chunk] & 3) << 2)) << (chunk * 4);
                    meta |= ((uint32_t)(meta_pos2[chunk] & 3) | ((uint32_t)(meta_pos3[chunk] & 3) << 2)) << (16 + chunk * 4);
                }
                s_packed_meta.push_back((int)meta);
                for (int thread_id = 0; thread_id < 4; ++thread_id) {
                    s_packed_a.push_back(pack_half2_int_online(ar0[thread_id * 2], ar0[thread_id * 2 + 1]));
                    s_packed_a.push_back(pack_half2_int_online(ar1[thread_id * 2], ar1[thread_id * 2 + 1]));
                }
            }
        }
        s_window_tile_offset.push_back((int)s_tile_window.size());
    }

    int num_tiles = (int)s_tile_window.size();
    return {
        tensor_from_int_vector(t_rowkNew_offset),
        tensor_from_int_vector(t_blockNew_offset),
        tensor_from_int_vector(t_columnNew),
        tensor_from_float_vector(t_valueNew),
        tensor_from_int_vector(t_window_rowNew),
        tensor_from_int_vector(t_atomicNew),
        tensor_from_long_vector(t_binaryNew),
        tensor_from_int_vector(c_rowNew_offset),
        tensor_from_int_vector(c_rowNew),
        tensor_from_int_vector(c_atomicNew),
        tensor_from_int_vector(c_columnNew),
        tensor_from_float_vector(c_valueNew),
        tensor_from_int_vector(c_rowNew_offset_residue),
        tensor_from_int_vector(c_rowNew_residue),
        tensor_from_int_vector(c_atomicNew_residue),
        tensor_from_int_vector(c_columnNew_residue),
        tensor_from_float_vector(c_valueNew_residue),
        s_columns, s_masks, s_values, s_pos, s_window, s_offsets,
        tensor_from_int_vector(s_tile_window),
        tensor_from_int_vector(s_tile_group),
        tensor_from_int_vector(s_tile_columns).view({(long)num_tiles, 16}).clone(),
        tensor_from_int_vector(s_packed_a).view({(long)num_tiles, 8, 4, 2}).clone(),
        tensor_from_int_vector(s_packed_meta).view({(long)num_tiles, 8}).clone(),
        tensor_from_int_vector(s_window_tile_offset),  // 新kernel: 窗口→tile范围映射 [num_windows+1]
        torch::tensor(elapsed.count())
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("block_sptc_2to4_online", &block_sptc_2to4_online,
        "Libra SPTC 2:4 Unit-Matching (SC'25 Graph Max-Match + O(1) Conflict Detection)");
}