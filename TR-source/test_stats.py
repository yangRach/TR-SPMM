import torch
from scipy.io import mmread
import numpy as np
import scipy.sparse as sp

mat = mmread('/home/zhangzhixuan/TR-SPMM_zzx/datasets/mip1/mip1.mtx')
if isinstance(mat, np.ndarray): mat = sp.csr_matrix(mat)
else: mat = mat.tocsr()

row_ptr = mat.indptr
max_nnz = 0
max_cols = 0
for i in range(0, mat.shape[0], 16):
    start = row_ptr[i]
    end = row_ptr[min(i+16, mat.shape[0])]
    nnz = end - start
    max_nnz = max(max_nnz, nnz)
    cols = len(np.unique(mat.indices[start:end]))
    max_cols = max(max_cols, cols)

print(f"Max window nnz: {max_nnz}")
print(f"Max window cols: {max_cols}")
