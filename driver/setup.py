from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name = 'chimera_lib',
    ext_modules = [
        cpp_extension.CUDAExtension(
            name='chimera_lib',
            sources=['src/chimera_lib.cu'],
            extra_compile_args={'nvcc': ['-O2']}
        )  
    ],
    cmdclass = {'build_ext': cpp_extension.BuildExtension},
    author = 'Walther Carballo-Hernández',
    author_email = 'walther.carballo_hernandez@uca.fr'
)