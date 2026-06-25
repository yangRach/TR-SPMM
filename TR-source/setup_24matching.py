from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


setup(
    name='Matching24',
    ext_modules=[
        CppExtension(
            name='matching_utils',
            sources=['./Block/block_matching_window_para_dr.cpp'],
            extra_compile_args=['-O3', '-march=native', '-fopenmp'],
        ),
        CppExtension(
            name='matching_utils_hash',
            sources=['./Block/block_matching_hash.cpp'],
            extra_compile_args=['-O3', '-march=native', '-fopenmp'],
        ),
        CppExtension(
            name='matching_utils_bucket',
            sources=['./Block/block_matching_bucket.cpp'],
            extra_compile_args=['-O3', '-march=native'],
        ),
        CUDAExtension(
            name='matching_utils_online',
            sources=['./Block/block_matching_online.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3'],
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
