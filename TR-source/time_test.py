import torch
import time
import matching_utils_cuda as backend
from scipy.io import mmread
import numpy as np
import scipy.sparse as sp

mat = mmread('/home/zhangzhixuan/TR-SPMM_zzx/datasets/mip1/mip1.mtx')
if isinstance(mat, np.ndarray): mat = sp.csr_matrix(mat)
else: mat = mat.tocsr()
mat.sort_indices()
if mat.dtype != np.float32: mat = mat.astype(np.float32)

row_ptr = torch.from_numpy(mat.indptr.astype(np.int32)).cuda()
col_ind = torch.from_numpy(mat.indices.astype(np.int32)).cuda()
values = torch.from_numpy(mat.data.astype(np.float32)).cuda()

# warmup
backend.match_2to4(row_ptr, col_ind, values, 16)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(10):
    backend.match_2to4(row_ptr, col_ind, values, 16)
torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"Average Kernel time: {(t1-t0)/10:.4f} s")
