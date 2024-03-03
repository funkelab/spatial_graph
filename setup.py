from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

rtree_wrapper = Extension(
    "spatial_graph.rtree_wrapper",
    sources=["spatial_graph/rtree_wrapper.pyx"],
    extra_compile_args=["-O3"],
    include_dirs=["spatial_graph/impl/rtree"],
    library_dirs=[],
    language="c",
)

setup(ext_modules=cythonize([rtree_wrapper]))
