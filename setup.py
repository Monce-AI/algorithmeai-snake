"""
Build Cython extensions for Snake acceleration.

Usage:
    pip install -e ".[fast]"
    python setup.py build_ext --inplace

Without Cython installed, this script is a no-op.
"""
from setuptools import setup

try:
    from Cython.Build import cythonize
    ext_modules = cythonize(
        "algorithmeai/_accel.pyx",
        compiler_directives={"language_level": "3"},
    )
except ImportError:
    ext_modules = []

setup(
    ext_modules=ext_modules,
)
