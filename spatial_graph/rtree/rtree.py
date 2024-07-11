import witty
from pathlib import Path
from ..dtypes import (
    DType,
    dtypes_to_struct,
    dtypes_to_arguments,
    dtypes_to_array_pointers,
    dtypes_to_array_pointer_names,
)


class RTree:
    def __new__(
        cls,
        node_dtype,
        coord_dtype,
        dims,
    ):
        node_dtype = DType(node_dtype)
        coord_dtype = DType(coord_dtype)

        src_dir = Path(__file__).parent
        wrapper_pyx = open(src_dir / "src_wrapper.pyx").read()
        wrapper_pyx = wrapper_pyx.replace("NODE_TYPE", node_dtype.to_pyxtype())
        wrapper_pyx = wrapper_pyx.replace("NODE_DTYPE", node_dtype.base)
        wrapper_pyx = wrapper_pyx.replace("COORD_TYPE", coord_dtype.to_pyxtype())
        wrapper_pyx = wrapper_pyx.replace("NUM_DIMS", str(dims))

        wrapper = witty.compile_module(
            wrapper_pyx,
            extra_compile_args=["-O3"],
            include_dirs=[str(src_dir)],
            language="c",
            quiet=True,
        )
        RTreeType = type(cls.__name__, (cls, wrapper.RTree), {})
        return wrapper.RTree.__new__(RTreeType)

    def __init__(self, node_dtype, coord_dtype, dims):
        super().__init__()
