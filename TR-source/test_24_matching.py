"""
test_24_matching.py — 2:4 结构化稀疏匹配算法的 Python 端测试脚本

功能:
  1. 自动发现并加载数据集 (.npz 优先, .mtx 兜底)
  2. 将矩阵转为 CSR 格式 (row_ptr, col_ind, values)
  3. 调用 C++ matching_utils.match_2to4() 执行匹配
  4. 展示 C++ 层返回的统计信息
  5. 对结果做 Python 层面的校验 (总 nnz / fake zeros 约束等)

编译方法 (首次运行前):
    cd TR-source/
    CXX=/home/zhaohongyang/miniconda3/envs/libra/bin/g++ \
      /home/zhaohongyang/miniconda3/envs/libra/bin/python \
      setup_24matching.py build_ext --inplace

用法:
    python test_24_matching.py                       # 使用默认数据集
    python test_24_matching.py --path <file>         # 指定数据集路径
    python test_24_matching.py --list                # 列出可用数据集
    python test_24_matching.py --method hash         # 使用 Hash-based 混合匹配
    python test_24_matching.py --method sliding      # 使用滑动视窗匹配 (默认)
    python test_24_matching.py --compare             # 同时运行三种方法并对比
"""

import os
import sys

import argparse
import time
import glob

import numpy as np
import scipy.sparse as sp
from scipy.io import mmread, mminfo


# ====================================================================
# 数据集发现与加载
# ====================================================================

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
DGL_DATASETS_DIR = os.path.join(PROJECT_ROOT, "dgl_datasets")


def find_datasets():
    """
    扫描数据集目录, 返回所有可用矩阵文件的路径列表。
    优先级: .npz > .mtx
    """
    datasets = {}

    # 扫描 dgl_datasets/ 下的 .npz 文件
    if os.path.isdir(DGL_DATASETS_DIR):
        for f in glob.glob(os.path.join(DGL_DATASETS_DIR, "*.npz")):
            name = os.path.splitext(os.path.basename(f))[0]
            datasets[name] = f

    # 扫描 datasets/ 下子目录中的 .mtx 文件 (npz 优先, 不覆盖)
    if os.path.isdir(DATASETS_DIR):
        for root, dirs, files in os.walk(DATASETS_DIR):
            # 先找 .npz
            for f in files:
                if f.endswith(".npz"):
                    name = os.path.splitext(f)[0]
                    if name not in datasets:
                        datasets[name] = os.path.join(root, f)
            # 再找 .mtx (不覆盖已发现的 npz)
            for f in files:
                if f.endswith(".mtx"):
                    name = os.path.splitext(f)[0]
                    if name not in datasets:
                        datasets[name] = os.path.join(root, f)

    return datasets


def _load_custom_npz(filepath):
    """
    加载 DGL 格式的 .npz 稀疏矩阵。

    该格式存储 CSR/COO 三元组: src_li(源节点), dst_li(目标节点), data(值), shape
    与 scipy.sparse.load_npz 不兼容, 需要手动构造 COO → CSR。
    """
    raw = np.load(filepath)
    keys = list(raw.keys())

    # 情况1: 标准 scipy 格式 (用 load_npz 加载)
    if 'format' in keys or 'indices' in keys:
        return sp.load_npz(filepath)

    # 情况2: DGL/自定义格式 — src_li, dst_li, data(可选), shape
    src = raw['src_li']
    dst = raw['dst_li']
    val = raw.get('data', None)
    shape = tuple(raw['shape'].tolist())

    if val is None:
        val = np.ones(len(src), dtype=np.float32)
    else:
        val = val.astype(np.float32)

    mat = sp.coo_matrix((val, (src, dst)), shape=shape)
    return mat.tocsr()


def load_matrix(filepath):
    """
    加载稀疏矩阵, 返回 scipy.sparse.csr_matrix。

    支持格式:
      - .npz: scipy.sparse.load_npz
      - .mtx: scipy.io.mmread (自动处理 pattern/real/integer/array 等变体)
    """
    ext = os.path.splitext(filepath)[1].lower()
    print(f"[加载] {filepath} (格式: {ext})")

    if ext == ".npz":
        mat = _load_custom_npz(filepath)
    elif ext == ".mtx":
        # 先读取 MatrixMarket 元信息, 判断矩阵格式
        # mminfo 返回 (rows, cols, entries, format, field, symmetry)
        mm_rows, mm_cols, mm_entries, mm_format, mm_field, mm_symm = mminfo(filepath)
        is_pattern = (mm_field == 'pattern')
        mat = mmread(filepath)

        # 处理 pattern (无权图) 格式: mmread 返回的 data 为全 1
        if is_pattern and hasattr(mat, 'data'):
            mat.data = np.ones(mat.nnz, dtype=np.float32)

        # 处理 array (稠密向量/矩阵) 格式: mmread 返回 np.ndarray
        if isinstance(mat, np.ndarray):
            mat = sp.csr_matrix(mat)
    else:
        raise ValueError(f"不支持的格式: {ext}")

    # 统一转为 CSR, float32
    if not sp.isspmatrix_csr(mat):
        mat = mat.tocsr()
    mat.sort_indices()
    if mat.dtype != np.float32:
        mat = mat.astype(np.float32)

    return mat


def matrix_to_csr_arrays(mat):
    """
    将 scipy CSR 矩阵拆解为三个 NumPy 数组:
      - row_ptr: int32, 长度 rows+1
      - col_ind: int32, 长度 nnz
      - values:  float32, 长度 nnz
    """
    rows, cols = mat.shape

    # 确保是 CSR 且 indeces 为 int32
    row_ptr = mat.indptr.astype(np.int32)
    col_ind = mat.indices.astype(np.int32)
    values  = mat.data.astype(np.float32)

    return row_ptr, col_ind, values, rows, cols


# ====================================================================
# 结果验证
# ====================================================================

def validate_stats(mat, stats):
    """
    Python 层对 C++ 返回的统计做一致性校验。
    Dual-Routing: SPTC Group 的 4 列 × 16 行 × 2 = 128 槽位上限.
    """
    errors = []

    actual_nnz = mat.nnz
    reported_nnz = stats["total_nnz"]
    if actual_nnz != reported_nnz:
        errors.append(f"nnz 不一致: scipy={actual_nnz}, C++={reported_nnz}")

    # SPTC 理论容量: num_sptc_groups × 32 (每 Group 16行 × 2 = 32 slots)
    num_sptc = stats["num_sptc_groups"]
    max_sptc_slots = num_sptc * 32
    fake = stats["fake_zeros"]
    total_slots = reported_nnz + fake
    if total_slots > expected_total:
        errors.append(
            f"容量不足: real_nnz+fake_zeros={total_slots}, capacity=num_groups*32={expected_total}"
        )

    if errors:
        print("\n[警告] 校验发现问题:")
        for e in errors:
            print(f"  - {e}")
    else:
        print(f"\n[校验通过] nnz={reported_nnz}, "
              f"SPTC Groups={num_sptc}, TC Blocks={stats['num_tc_blocks']}, "
              f"SPTC容量={max_sptc_slots}, 假0={fake}")


# ====================================================================
# 主流程
# ====================================================================

def _run_single(mat, row_ptr, col_ind, values, window_size, method_name, match_fn):
    """运行单次匹配并返回 (stats, elapsed_seconds)."""
    print(f"\n[匹配] 方法: {method_name} (window_size={window_size})...")
    t0 = time.perf_counter()
    stats = match_fn(row_ptr, col_ind, values, window_size=window_size)
    elapsed = time.perf_counter() - t0
    return stats, elapsed


def _display_stats(stats, elapsed, method_name):
    """打印单次匹配的 Python 层统计."""
    print(f"╔══════════════════════════════════════════════╗")
    print(f"║   Python 层结果 [{method_name}]              ║")
    print(f"╠══════════════════════════════════════════════╣")
    print(f"║   C++ 耗时:            {elapsed:.4f} 秒               ║")
    print(f"║   16行 Row Panels:    {stats['num_row_panels']:<8d}                 ║")
    print(f"║   SPTC 2:4 Groups:    {stats['num_sptc_groups']:<8d}                 ║")
    print(f"║   TC 16×8 Blocks:     {stats['num_tc_blocks']:<8d}                 ║")
    print(f"║   16×16 SPTC Block:   {stats['num_16x16_blocks']:<8d}                 ║")
    print(f"║   Block 填充 Group:   {stats['block_padding_groups']:<8d}                 ║")
    print(f"╠══════════════════════════════════════════════╣")
    print(f"║   细粒度孤儿列:       {stats['fine_fallback']:<8d} (→Pool)         ║")
    print(f"║   粗粒度拆散列:       {stats['coarse_fallback']:<8d} (→Pool)         ║")
    print(f"║   假0填充:            {stats['fake_zeros']:<8d}                 ║")
    print(f"╚══════════════════════════════════════════════╝")


def _run_compare(mat, row_ptr, col_ind, values, window_size, methods):
    """运行全部匹配策略并输出对比表."""
    results = {}
    for key, (name, fn) in methods.items():
        stats, elapsed = _run_single(mat, row_ptr, col_ind, values,
                                     window_size, name, fn)
        results[key] = (name, stats, elapsed)

    keys = list(methods.keys())
    s = results[keys[0]][1]
    h = results[keys[1]][1]
    b = results[keys[2]][1] if len(keys) > 2 else None
    st = results[keys[0]][2]
    ht = results[keys[1]][2]
    bt = results[keys[2]][2] if len(keys) > 2 else 0

    print(f"\n╔══════════════════════════════════════════════════════════════════════════╗")
    print(f"║          Sliding Window  vs  Hash-based  vs  Bucketing                  ║")
    print(f"╠══════════════════════════════════════════════════════════════════════════╣")
    print(f"║  指标                  Sliding         Hash            Bucket           ║")
    print(f"╠══════════════════════════════════════════════════════════════════════════╣")
    print(f"║  耗时 (秒):           {st:<11.4f}     {ht:<11.4f}     {bt:<11.4f}     ║")
    print(f"║  16×4 Groups:         {s['num_groups']:<11d}     {h['num_groups']:<11d}     {b['num_groups']:<11d}     ║")
    print(f"║  16×16 Blocks:        {s['num_16x16_blocks']:<11d}     {h['num_16x16_blocks']:<11d}     {b['num_16x16_blocks']:<11d}     ║")
    print(f"║  Block Padding:       {s['block_padding_groups']:<11d}     {h['block_padding_groups']:<11d}     {b['block_padding_groups']:<11d}     ║")
    print(f"╠══════════════════════════════════════════════════════════════════════════╣")
    print(f"║  细粒度 Fallback:     {s['fine_fallback']:<11d}     {h['fine_fallback']:<11d}     {b['fine_fallback']:<11d}     ║")
    print(f"║  粗粒度 Fallback:     {s['coarse_fallback']:<11d}     {h['coarse_fallback']:<11d}     {b['coarse_fallback']:<11d}     ║")
    print(f"║  假0填充:             {s['fake_zeros']:<11d}     {h['fake_zeros']:<11d}     {b['fake_zeros']:<11d}     ║")
    print(f"╚══════════════════════════════════════════════════════════════════════════╝")

    for key in keys:
        print(f"\n[校验] {results[key][0]}:")
        validate_stats(mat, results[key][1])
    print("\n对比测试完成!")


def _run_single_dual(mat, row_ptr, col_ind, values, window_size,
                     method_name, match_fn, dense_threshold=8, t_max=8):
    """运行 Dual-Routing 匹配 (sliding 方法)."""
    print(f"\n[匹配] 方法: {method_name} "
          f"(window={window_size}, dense_th={dense_threshold}, t_max={t_max})...")
    t0 = time.perf_counter()
    stats = match_fn(row_ptr, col_ind, values,
                     window_size=window_size,
                     dense_threshold=dense_threshold,
                     t_max=t_max)
    elapsed = time.perf_counter() - t0
    return stats, elapsed


def _run_single_online(mat, row_ptr, col_ind, values, window_size,
                       method_name, match_fn, dense_threshold=8, t_max=8):
    """运行 CUDA Online 双路路由匹配."""
    print(f"\n[匹配] 方法: {method_name} "
          f"(window={window_size}, dense_th={dense_threshold}, t_max={t_max})...")
    t0 = time.perf_counter()
    stats = match_fn(row_ptr, col_ind, values,
                     window_size=window_size,
                     dense_threshold=dense_threshold,
                     t_max=t_max)
    elapsed = time.perf_counter() - t0
    return stats, elapsed


def _display_stats_online(stats, elapsed, method_name):
    """打印 CUDA Online 双路路由的统计."""
    print(f"╔══════════════════════════════════════════════╗")
    print(f"║   Python 层结果 [{method_name}]    ║")
    print(f"╠══════════════════════════════════════════════╣")
    print(f"║   C++/CUDA 耗时:       {elapsed:.4f} 秒               ║")
    print(f"║   SPTC 2:4 Groups:    {stats['num_sptc_groups']:<8d}                 ║")
    print(f"║   TC 16×8 Blocks:     {stats['num_tc_blocks']:<8d}                 ║")
    print(f"║   → TC 路由列数:      {stats['dense_to_tc_cols']:<8d}                 ║")
    print(f"║   → SPTC Fallback:    {stats['dense_to_sptc_cols']:<8d}                 ║")
    print(f"║   16×16 Block 数:     {stats['num_16x16_blocks']:<8d}                 ║")
    print(f"║   Block 填充 Group:   {stats['block_padding_groups']:<8d}                 ║")
    print(f"╠══════════════════════════════════════════════╣")
    print(f"║   细粒度孤儿列:       {stats['fine_fb']:<8d} (Phase1→Pool)      ║")
    print(f"║   粗粒度拆散列:       {stats['coarse_fb_cols']:<8d} (Phase2→Pool)      ║")
    print(f"║   假0填充:            {stats['fake_zeros']:<8d}                 ║")
    print(f"╚══════════════════════════════════════════════╝")


def _validate_online(mat, stats):
    """对 CUDA Online 匹配的统计做一致性校验."""
    errors = []
    nnz = mat.nnz

    # 1. TC + SPTC 覆盖检验
    tc_cols = stats['dense_to_tc_cols']
    sptc_fb_cols = stats['dense_to_sptc_cols']
    # Online 方法: 所有列最终进入 SPTC 或 TC
    # SPTC 侧的 column-slots = num_sptc_groups * 4
    # TC 侧的 column-slots = num_tc_blocks * 8
    # 实际列数 ≤ slot 总数 (有虚拟0列)
    total_slots = stats['num_sptc_groups'] * 4 + stats['num_tc_blocks'] * 8

    # 2. Block 打包一致性
    ng = stats['num_sptc_groups']
    nb = stats['num_16x16_blocks']
    pad = stats['block_padding_groups']
    if pad != nb * 4 - ng:
        errors.append(f"Block填充不一致: pad={pad}, expected={nb*4 - ng}")

    if errors:
        print("\n[警告] 校验发现问题:")
        for e in errors:
            print(f"  - {e}")
    else:
        print(f"\n[校验通过] SPTC Groups={ng}, TC Blocks={stats['num_tc_blocks']}, "
              f"总slot={total_slots}, 假0={stats['fake_zeros']}")


def main():
    parser = argparse.ArgumentParser(
        description="2:4 结构化稀疏匹配测试脚本"
    )
    parser.add_argument(
        "--path", type=str, default=None,
        help="指定矩阵文件路径 (.npz 或 .mtx)"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="列出所有可用数据集"
    )
    parser.add_argument(
        "--window", type=int, default=16,
        help="窗口行数 (默认 16)"
    )
    parser.add_argument(
        "--method", type=str, default="sliding",
        choices=["sliding", "hash", "bucket", "online"],
        help="匹配策略: sliding | hash | bucket | online (CUDA双路路由)"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="同时运行两种匹配策略并对比结果"
    )
    args = parser.parse_args()

    # ---- 导入 C++ 模块 ----
    # 必须先 import torch 预加载动态库, 否则 .so 链接的 libc10.so 找不到
    try:
        import torch  # noqa: F401  预加载 torch 动态库
        import matching_utils
    except ImportError as e:
        print("=" * 60)
        print(f"  错误: 导入模块失败 — {e}")
        print("  请先编译对应扩展:")
        print("    CPU: cd TR-source/ && CXX=$CONDA_PREFIX/bin/g++ python setup_24matching.py build_ext --inplace")
        print("    CUDA: cd TR-source/ && python setup_cuda.py build_ext --inplace")
        print("=" * 60)
        sys.exit(1)

    if args.cuda:
        if args.window < 1 or args.window > 16:
            print("错误: CUDA 版本当前仅支持 window_size in [1, 16] (uint16 mask 实现)")
            sys.exit(1)
        if not torch.cuda.is_available():
            print("错误: torch.cuda 不可用，无法运行 CUDA 版本。请检查 NVIDIA 驱动 / CUDA / 可见 GPU。")
            sys.exit(1)

    # ---- 列出数据集 ----
    if args.list:
        datasets = find_datasets()
        print(f"\n可用数据集 (共 {len(datasets)} 个):")
        for name, path in sorted(datasets.items()):
            size_kb = os.path.getsize(path) / 1024
            print(f"  {name:30s}  [{size_kb:.1f} KB]  {path}")
        return

    # ---- 选择并加载矩阵 ----
    if args.path:
        filepath = args.path
        if not os.path.exists(filepath):
            print(f"错误: 文件不存在: {filepath}")
            sys.exit(1)
    else:
        datasets = find_datasets()
        if not datasets:
            print("错误: 未找到任何数据集!")
            print(f"  已搜索: {DGL_DATASETS_DIR}, {DATASETS_DIR}")
            print("  提示: 使用 --path 手动指定文件路径")
            sys.exit(1)

        npz_files = {k: v for k, v in datasets.items() if v.endswith(".npz")}
        if npz_files:
            name = sorted(npz_files.keys())[0]
        else:
            name = sorted(datasets.keys())[0]
        filepath = datasets[name]
        print(f"[自动选择] {name}")

    # ---- 加载矩阵 ----
    mat = load_matrix(filepath)
    print(f"  矩阵形状: {mat.shape[0]} × {mat.shape[1]}")
    print(f"  非零元:   {mat.nnz}")
    print(f"  稀疏度:   {100 * (1 - mat.nnz / (mat.shape[0] * mat.shape[1])):.4f}%")

    # ---- 转为 CSR 数组 ----
    row_ptr, col_ind, values, rows, cols = matrix_to_csr_arrays(mat)

    # ---- 调用 C++ 匹配 ----
    print(f"\n[匹配] 开始 2:4 结构化匹配 (window_size={args.window})...")
    t0 = time.perf_counter()

    stats = matching_utils.match_2to4(
        row_ptr, col_ind, values,
        window_size=args.window
    )

    elapsed = time.perf_counter() - t0

    # ---- 展示 Python 层结果 ----
    print(f"╔══════════════════════════════════════════════╗")
    print(f"║   Python 层结果确认                          ║")
    print(f"╠══════════════════════════════════════════════╣")
    print(f"║   C++ 耗时:            {elapsed:.4f} 秒               ║")
    print(f"║   16行 Row Panels:    {stats['num_row_panels']:<8d}                 ║")
    print(f"║   16×4 Groups:        {stats['num_groups']:<8d}                 ║")
    print(f"║   16×16 稀疏块:       {stats['num_16x16_blocks']:<8d}                 ║")
    print(f"║   Block 填充 Group:   {stats['block_padding_groups']:<8d}                 ║")
    print(f"╠══════════════════════════════════════════════╣")
    print(f"║   细粒度 Fallback:    {stats['fine_fallback']:<8d}                 ║")
    print(f"║   粗粒度 Fallback:    {stats['coarse_fallback']:<8d}                 ║")
    print(f"║   假0填充:            {stats['fake_zeros']:<8d}                 ║")
    print(f"╚══════════════════════════════════════════════╝")

    # ---- 校验 ----
    validate_stats(mat, stats)

    print("\n测试完成!")


if __name__ == "__main__":
    main()
