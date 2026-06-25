import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5;8.0;8.6;8.9")


setup(
    name='Matching24CUDA',
    ext_modules=[
        CUDAExtension(
            name='matching_utils_cuda',
            sources=[
                './Block/block_match_cuda.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': ['-O3', '-std=c++17', '--use_fast_math'],
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
