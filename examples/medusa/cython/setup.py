from distutils.core import setup

import numpy
from Cython.Build import cythonize

setup(
    name="fastloop",
    ext_modules=cythonize("test_cython.pyx"),
    include_dirs=[numpy.get_include()],
)
