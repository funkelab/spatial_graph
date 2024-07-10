import witty
from pathlib import Path
from .dtypes import (
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
        rtree_pyx = open(src_dir / "rtree_wrapper.pyx").read()
        rtree_pyx = rtree_pyx.replace("NODE_TYPE", node_dtype.to_pyxtype())
        rtree_pyx = rtree_pyx.replace("NODE_DTYPE", node_dtype.base)
        rtree_pyx = rtree_pyx.replace("COORD_TYPE", coord_dtype.to_pyxtype())
        rtree_pyx = rtree_pyx.replace("NUM_DIMS", str(dims))

        rtree = witty.compile_module(
            rtree_pyx,
            extra_compile_args=["-O3"],
            include_dirs=[str(src_dir)],
            language="c",
            quiet=True,
        )
        RTreeType = type(cls.__name__, (cls, rtree.RTree), {})
        return rtree.RTree.__new__(RTreeType)

    def __init__(self, node_dtype, coord_dtype, dims):
        super().__init__()
