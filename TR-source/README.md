# TR-SPMM
三路路由加速稀疏矩阵乘

## 项目说明
数据集放在datasets/
LIBRA_SKIP_TC=true的意思是跳过tc块，改为false的话tcu就不会等于0
在TR-SPMM_zzx/TR-source目录下，编译命令为：
python setup_online.py build_ext --inplace
在TR-SPMM_zzx/TR-source/spmm目录下，运行命令为：
python spmm_3090_fp16_test.py 128 mip1   
在TR-source/目录下是编译block分块的代码，在TR-source/SpMM/下就是编译内核代码


## block_matching_cuda.cu

### 运行方法
conda activate libra
cd /home/zhangzhixuan/TR-SPMM_zzx/TR-source


编译block_matching_cuda.cu对应的python文件
cd /home/zhangzhixuan/TR-SPMM_zzx/TR-source
python setup_cuda.py build_ext --inplace

使用test_24_matching.py展示cuda和cpp的结构化匹配的结果

显示cpu版本的匹配
python test_24_matching.py
python test_24_matching.py --path /home/zhangzhixuan/TR-SPMM_zzx/datasets/mip1/mip1.mtx --window 16
python test_24_matching.py --path /home/zhangzhixuan/TR-SPMM_zzx/datasets/gupta1/gupta1.mtx --window 16
python test_24_matching.py --path /home/zhangzhixuan/TR-SPMM_zzx/datasets/2D_27628_bjtcai/2D_27628_bjtcai.mtx --window 16
python test_24_matching.py --path /home/zhangzhixuan/TR-SPMM_zzx/datasets/Bump_2911/Bump_2911.mtx --window 16


显示cuda的匹配
python test_24_matching.py --cuda  --window 16
python test_24_matching.py --cuda --path /home/zhangzhixuan/TR-SPMM_zzx/datasets/mip1/mip1.mtx --window 16
python test_24_matching.py --cuda --path /home/zhangzhixuan/TR-SPMM_zzx/datasets/gupta1/gupta1.mtx --window 16
python test_24_matching.py --cuda --path /home/zhangzhixuan/TR-SPMM_zzx/datasets/2D_27628_bjtcai/2D_27628_bjtcai.mtx --window 16
python test_24_matching.py --cuda --path /home/zhangzhixuan/TR-SPMM_zzx/datasets/Bump_2911/Bump_2911.mtx --window 16
临时测试：
conda activate libra
cd /home/zhangzhixuan/TR-SPMM_zzx/TR-source
python setup_cuda.py build_ext --inplace
python test_24_matching.py --cuda --path /home/zhangzhixuan/TR-SPMM_zzx/datasets/mip1/mip1.mtx --window 16

### cuda优化策略

| 优化项 | 说明 | 位置 |
|---------|------|------|
| **预分配全局工作区** | 使用 PyTorch CUDA Tensor 一次性分配所有 workspace（temp/masks/matched/pairs/pair_used/hash），避免 kernel 内反复 `cudaMalloc/cudaFree` | [第1216-1300行](Block/block_match_cuda.cu#L1216-L1300) |
| **PyTorch 缓存分配器** | 使用 `torch::empty` 代替原始 `cudaMalloc/cudaFree`，减少大型矩阵分配开销和抖动 | [第1262-1290行](Block/block_match_cuda.cu#L1262-L1290) |
| **64/128 线程自适应** | 按窗口 nnz 分桶：轻窗口（≤512）用 64 线程，重窗口用 128 线程，减少空转和同步开销 | [第1287-1319行](Block/block_match_cuda.cu#L1287-L1319) |
| **巨型窗口多 block 拆分** | 对超大窗口（nnz≥8192），先用单 block 构建窗口列信息，再用多 block 并行匹配 pair/group，解决负载不均问题 | [第821-1194行](Block/block_match_cuda.cu#L821-L1194) |
| **多阶段 kernel 分离** | 将匹配分为 build/pair/init/group 四个 kernel，分别优化，避免单 kernel 内过度复杂的分支和状态同步 | [第821-1194行](Block/block_match_cuda.cu#L821-L1194) |
| **warp 级任务队列** | 每个 warp 独立从全局任务队列取 i/pair_idx，warp 内并行搜索 j，减少 block 级同步粒度，提升整体吞吐量 | [第948-1061行](Block/block_match_cuda.cu#L948-L1061), [第1085-1194行](Block/block_match_cuda.cu#L1085-L1194) |
| **top-2 重试机制** | warp 内同时计算 best 和 second best，best 被抢占时自动尝试 second best，降低无意义 fallback，改善匹配质量 | [第176-208行](Block/block_match_cuda.cu#L176-L208) |
| **byte atomic claim** | 使用按 4 字节对齐的 atomicCAS 实现按位占用标记，避免对齐问题，提高原子操作可靠性 | [第212-228行](Block/block_match_cuda.cu#L212-L228) |



## block_online.cpp 代码结构

### 一、数据结构定义

| 数据结构 | 位置 | 作用 |
|---------|------|------|
| `OnlineColInfo` | [第20-25行](TR-source/Block/block_online.cpp#L20-L25) | 存储每列在16行窗口内的非零元信息 |
| `SptcUnit` | [第28-34行](TR-source/Block/block_online.cpp#L28-L34) | 由1~2列组成的Unit（兼容的相邻列打包） |
| `OnlineCudaPart` | [第37-45行](TR-source/Block/block_online.cpp#L37-L45) | CUDA行数据结构 |
| `OnlineTcuPart` | [第47-56行](TR-source/Block/block_online.cpp#L47-L56) | TCU部分数据结构 |
| `OnlinePackResult` | [第58-62行](TR-source/Block/block_online.cpp#L58-L62) | 打包结果结构 |
| `OnlineCudaWork` | [第64-67行](TR-source/Block/block_online.cpp#L64-L67) | CUDA工作量估计结构 |

### 二、位运算冲突检测函数

| 函数名 | 位置 | 作用 |
|--------|------|------|
| `units_match_check` (2参数) | [第85-88行](TR-source/Block/block_online.cpp#L85-L88) | 检测两个Unit是否可以合并（O(1)位运算） |
| `units_match_check` (4参数) | [第90-98行](TR-source/Block/block_online.cpp#L90-L98) | 检测四个Unit是否可以合并 |
| `coarse_grained_4col_z_check_online` | [第108-115行](TR-source/Block/block_online.cpp#L108-L115) | 根据 `Z = !(AB(C + D) v CD(A + B))` 判断四列模式是否需要复杂通路 |
| `coarse_grained_4col_route_online` | [第117-123行](TR-source/Block/block_online.cpp#L117-L123) | 返回四列粗粒度路径编号：0跳过，1单pair，2轻量路径，3通用路径 |
| `encode_sptc_row_by_route_online` | [第125-165行](TR-source/Block/block_online.cpp#L125-L165) | 对 4 列 group 的单行先分类，再按不同路径编码 value 和 position |

### 三、辅助函数

| 函数名 | 位置 | 作用 |
|--------|------|------|
| `estimate_cuda_work_online` | [第171-183行](TR-source/Block/block_online.cpp#L171-L183) | 估计CUDA的工作量（大/小任务划分） |
| `env_enabled_online` | [第185-189行](TR-source/Block/block_online.cpp#L185-L189) | 读取环境变量布尔值 |
| `env_int_online` | [第191-195行](TR-source/Block/block_online.cpp#L191-L195) | 读取环境变量整数值 |
| `env_float_online` | [第197-201行](TR-source/Block/block_online.cpp#L197-L201) | 读取环境变量浮点数值 |
| `tensor_from_*_vector` 系列 | [第207-231行](TR-source/Block/block_online.cpp#L207-L231) | 安全的Tensor构造函数，避免from_blob悬空指针 |
| `merge_float_vectors` | [第233-237行](TR-source/Block/block_online.cpp#L233-L237) | 合并多个float向量 |
| `count_float_vectors` | [第238-243行](TR-source/Block/block_online.cpp#L238-L243) | 统计多个float向量的大小 |
| `float_to_half_bits_online` | [第245-250行](TR-source/Block/block_online.cpp#L245-L250) | 将float转换为半精度二进制 |
| `pack_half2_int_online` | [第251-255行](TR-source/Block/block_online.cpp#L251-L255) | 将两个半精度浮点数打包成一个整数 |
| `load_sptc_row_chunk_cpu_online` | [第257-277行](TR-source/Block/block_online.cpp#L257-L277) | 加载SPTC行数据块 |

### 四、Stage 1: TC Funnel - 极稠密块进Tensor Core

| 函数名 | 位置 | 作用 |
|--------|------|------|
| `select_tc_columns_online` | [第282-320行](TR-source/Block/block_online.cpp#L282-L320) | 选择适合Tensor Core的极稠密列并分组 |

### 五、Stage 2: SPTC Funnel - 基于最大匹配的Unit打包

| 函数名 | 位置 | 作用 |
|--------|------|------|
| `build_sptc_units` | [第334-395行](TR-source/Block/block_online.cpp#L334-L395) | 构建Unit：使用双指针将前后兼容列打包 |
| `match_sptc_units_max_degree` | [第399-459行](TR-source/Block/block_online.cpp#L399-L459) | 最大度优先贪心匹配：将Unit配对成4列SPTC组 |

### 六、CUDA行切分

| 函数名 | 位置 | 作用 |
|--------|------|------|
| `append_cuda_part_online` | [第464-480行](TR-source/Block/block_online.cpp#L464-L480) | 添加CUDA部分数据 |
| `split_cuda_row_online` | [第482-506行](TR-source/Block/block_online.cpp#L482-L506) | 切分CUDA行数据 |

### 七、TCU部分构建器

| 函数名 | 位置 | 作用 |
|--------|------|------|
| `build_tcu_part_online` | [第511-607行](TR-source/Block/block_online.cpp#L511-L607) | 构建TCU部分（Tensor Core数据） |

### 八、主入口：三级漏斗路由

| 函数名 | 位置 | 作用 |
|--------|------|------|
| `block_sptc_2to4_online` | [第612-935行](TR-source/Block/block_online.cpp#L612-L935) | 主函数，实现零元素填充，实现三级漏斗路由（TC → SPTC → CUDA） |

### 九、Pybind11模块定义

| 函数名 | 位置 | 作用 |
|--------|------|------|
| `PYBIND11_MODULE` | [第936-939行](TR-source/Block/block_online.cpp#L936-L939) | 将C++函数导出为Python模块 |

## 整体架构说明

该文件实现了SC'25 MP-SpMM论文中的三级漏斗路由算法：
1. **TC Funnel**：极稠密块进Tensor Core
2. **SPTC Funnel**：基于最大匹配的Unit打包（4列组）
3. **CUDA Funnel**：剩余数据走CUDA回退路径

核心创新点在于使用O(1)位运算进行冲突检测和最大度优先贪心匹配算法。


## 流程模拟举例

### 以运行 `python spmm_3090_fp16_test.py 144 2D_27628_bjtcai` 为例

#### 数据流向链：

```
📂 dgl_dataset/sp_matrix/2D_27628_bjtcai.npz
    ↓ [tr_paths.py:22] load_graph_npz() 读取
📊 边列表 (src_li, dst_li)
    ↓ [mdataset2_online.py:70] 构造 CSR 稀疏邻接矩阵
📈 CSR 稀疏邻接矩阵
    ↓ [mdataset2_online.py:94] Rabbit 重排序优化
🔀 优化后的矩阵
    ↓ [block_online.cpp:612] Libra5BlockOnline.block_sptc_2to4_online() 三路分流
    ├─→ 🧊 TC 数据 [block_online.cpp:282] select_tc_columns_online() → Tensor Core 计算
    ├─→ 🧊 SPTC 数据 [block_online.cpp:334] build_sptc_units() + [block_online.cpp:125] encode_sptc_row_by_route_online() → Sparse Tensor Core 计算
    └─→ 🧊 CUDA 数据 → 普通 CUDA 核计算
    ↓ [test_tr_csr_online.py:87] Libra6SpMMOnline.forward_fp16_tcu_cuda_sptc_mma_parallel_online() 计算
✅ 输出 SpMM 结果 + 性能统计
```

#### 流程详情：

| 步骤 | 位置 | 说明 |
|------|------|------|
| **1. 数据加载** | `spmm/tr_csr_fp16/tr_paths.py:22` | 从 `dgl_dataset/sp_matrix/2D_27628_bjtcai.npz` 读取稀疏矩阵 |
| **2. 矩阵构造** | `spmm/tr_csr_fp16/mdataset2_online.py:70` | 将边列表转换为 CSR 格式的稀疏邻接矩阵 |
| **3. 重排序优化** | `spmm/tr_csr_fp16/mdataset2_online.py:94` | Rabbit 重排序 + 可选 TCA 重排序 |
| **4. 三路分流** | `TR-source/Block/block_online.cpp:612` | 主函数 `block_sptc_2to4_online()` 执行三级漏斗路由 |
| **5. TC Funnel** | `TR-source/Block/block_online.cpp:282` | `select_tc_columns_online()` 根据 LIBRA_DENSITY 筛选极稠密列进 Tensor Core |
| **6. SPTC Funnel** | `TR-source/Block/block_online.cpp:334` | `build_sptc_units()` + 最大度优先贪心匹配成 4 列组，并在编码前做粗粒度分类分流 |
| **7. CUDA Funnel** | `TR-source/Block/block_online.cpp:482` | 剩余残渣数据切分为 long/short 两部分 |
| **8. SpMM 计算** | `spmm/tr_csr_fp16/test_tr_csr_online.py:87` | 三路并行计算并合并结果 |

#### 关键参数说明：

| 参数 | 来源 | 作用 |
|------|------|------|
| `144` (dimN) | 命令行第1个参数 | 特征维度，用于 `mdataset2_online.py:166` 生成特征矩阵 |
| `2D_27628_bjtcai` | 命令行第2个参数 | 图数据集名称，定位 `.npz` 文件 |
| `LIBRA_DENSITY` | 环境变量 | `block_online.cpp:291` 控制多少列进 Tensor Core（默认 8） |
| `LIBRA_SPTC_THRESHOLD` | 环境变量 | SPTC 非零元阈值（默认 12） |
| `CUDA_VISIBLE_DEVICES` | 环境变量 | GPU 设备选择（默认 3） |

#### 运行示例：

```bash
cd /home/zhangzhixuan/TR-SPMM_zzx/spmm
python spmm_3090_fp16_test.py 144 2D_27628_bjtcai
```

输出类似：
```
GPU 型号
TCA reorder disabled (LIBRA_TCA_REORDER=0)
TC候选列数量: XXX
util: XX.XX% (nnz=XXX, slots=XXX)  tc_min_util: XX.XX%
tcu: XXXXX;   sptc: XXXXX;   cuda: XXXXX
parts_t: XX;   parts_c: XX;   parts_c_short: XX;   sptc_groups: XX;   sptc_tiles: XX
144-2D_27628_bjtcai online-tcu-sptc-cuda-8-XXXX.XXXX
parallel_tc_cuda_sptc_mma=XXXX.XXXX
kernel_breakdown: tc=XXXX.XXXX, cuda=XXXX.XXXX, cuda_long=XXXX.XXXX, cuda_short=XXXX.XXXX, sptc=XXXX.XXXX
```





