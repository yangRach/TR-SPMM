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
展示24匹配的结果
cd /home/zhangzhixuan/TR-SPMM_zzx/TR-source
python test_24_matching.py --path /home/zhangzhixuan/TR-SPMM_zzx/dgl_dataset/sp_matrix/mip1.npz

编译block_matching_cuda.cu对应的python文件
cd /home/zhangzhixuan/TR-SPMM_zzx/TR-source
python setup_cuda.py build_ext --inplace

显示cpu版本的匹配
python test_24_matching.py
显示cuda的匹配
python test_24_matching.py --cuda [ 矩阵名称] --window 16
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


## 可能的创新点（gemini

### Idea 1: 动态/自适应的 2:4 稀疏度宽容架构 (Adaptive SPTC Padding)
痛点分析 ：
目前的 SPTC 要求 严格的 2:4 稀疏 （每 4 个元素必须恰好有 2 个非零）。在 MP-SpMM 和 Libra 中，为了凑齐 2:4，如果匹配不到完美的 4 列，就会 强行补零 (Zero-padding) 或者 退回给 CUDA Core 。强行补零浪费了 Tensor Core 算力，退回 CUDA 则打破了执行的连续性，导致严重的负载不均衡。

创新点 ：
我们能否 不强求完美的 2:4 匹配 ，而是引入一种 混合/自适应填充策略 ？

- 跨窗口借用 (Cross-Window Borrowing) ：如果当前 16 行窗口凑不齐 4 列，允许从相邻的 16 行窗口中“借用”非零元填入。因为稠密矩阵 `X` 是共享的，只要维护好行索引映射，跨窗口打包可以大幅减少补零。
- 计算重用 (Computation Reuse) ：在强行补零的位置，不填 0，而是填入该行其他列的非零元（相当于让 TC 冗余计算），然后在 Shared Memory 归约阶段进行去重。这利用了 TC 极高的吞吐量来掩盖分支散度。
故事包装 (Storyline) ：
"打破 2:4 刚性约束：一种面向 Sparse Tensor Core 的自适应弹性打包机制。"

### Idea 2: 面向 1:2 或 2:8 混合稀疏的软硬件协同调度 (Mixed-Ratio Sparse Routing)
痛点分析 ：
NVIDIA 的 Sparse Tensor Core 原生支持 2:4，但实际图数据集的度数分布是极其长尾的（Power-law）。三级漏斗（TC -> 2:4 SPTC -> CUDA）对极度稀疏的部分（比如度数 < 2 的节点）处理很差，依然大量依赖 CUDA Core。

创新点 ：

- 虚拟稀疏比 (Virtual Sparse Ratios) ：在软件层面，我们把矩阵划分为 1:2 区域（极稀疏）、2:4 区域（中等）和 8:8 区域（全稠密 TC）。
- 对于 1:2 区域，我们通过特定的内存排布，让两个 1:2 的逻辑块在寄存器中 交织 (Interleave) 成一个物理的 2:4 块，一次 mma.sp 指令计算两个极稀疏区域。
- 创新命名 ：不需要图匹配，而是 多粒度张量折叠 (Multi-granularity Tensor Folding) 。
故事包装 (Storyline) ：
"超越 2:4：针对长尾图数据的多粒度虚拟稀疏张量核心加速器。"

### Idea 3: 延迟掩盖与预处理计算融合 (Compute-Fused Preprocessing)
痛点分析 ：
无论是 Libra 还是 MP-SpMM，它们的预处理开销（包括 O(1) 冲突检测和图匹配）都在 CPU 上进行，或者作为 GPU 上非常昂贵的独立 Kernel。在 GNN 训练的每个 Epoch 甚至推断中，如果图结构发生变化（如 Graph Sampling/Dropout），这种 静态预处理的开销是不可接受的 。

创新点 ：

- 在线/即时打包 (On-the-fly Packing) ：抛弃复杂的全局图匹配。设计一个极其轻量的 GPU Kernel，在执行 SpMM 之前，仅仅利用 Warp-level 的原生原语（如 __match_any_sync , __popc ），在加载数据到 Shared Memory 的瞬间， 即时决定 谁进 TC、谁进 SPTC。
- 将 MP-SpMM 的 O(1) 编码下放到 GPU 的 Warp 调度器中。Warp 内的 32 个线程各自持有 1 列的信息，通过 Warp Shuffle 直接完成 32 列的局部匹配，完全省去 CPU 预处理。
故事包装 (Storyline) ：
"零开销的 Sparse Tensor Core 路由：基于 Warp-level 即时匹配的动态 SpMM。"



nvcc -O3 -arch=sm_86 /home/zhangzhixuan/TR-SPMM_zzx/TR-source/Block/warp_match_poc.cu -o /home/zhangzhixuan/TR-SPMM_zzx/TR-source/Block/warp_match_poc
/home/zhangzhixuan/TR-SPMM_zzx/TR-source/Block/warp_match_poc
