from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name='Matching24',
    ext_modules=[
        CppExtension(
            name='matching_utils',
            sources=[

                './Block/block_match_windows.cpp',

            ],
            extra_compile_args=['-O3', '-march=native'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
