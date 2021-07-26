from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='chimera_lib',
      ext_modules=[cpp_extension.CppExtension('chimera_lib', ['src/chimera_lib.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})