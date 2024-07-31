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

        pyx_declarations = f"""
    # coordinate type to use:
    ctypedef {coord_dtype.to_pyxtype()} coord_t

    # C and PYX items are the same here:
    ctypedef {node_dtype.to_pyxtype()} item_t
    ctypedef {node_dtype.to_pyxtype()} pyx_item_t

    # converters
    cdef item_t convert_pyx_to_c_item(pyx_item_t c_item, coord_t *min, coord_t* max)
    cdef void copy_c_to_pyx_item(const item_t c_item, pyx_item_t *pyx_item)
"""

        c_declarations = f"""
// coordinate type to use:
typedef {coord_dtype.to_pyxtype()} coord_t;

// C and PYX items are the same here:
typedef {node_dtype.to_pyxtype()} pyx_item_t;
typedef {node_dtype.to_pyxtype()} item_t;
"""

        c_function_implementations = """
// default PYX<->C converters, just casting
inline item_t convert_pyx_to_c_item(pyx_item_t pyx_item, coord_t *min, coord_t *max) {
    return (item_t)pyx_item;
}
inline void copy_c_to_pyx_item(const item_t c_item, pyx_item_t *pyx_item) {
    memcpy(pyx_item, &c_item, sizeof(item_t));
}
        """

        src_dir = Path(__file__).parent
        wrapper_pyx = open(src_dir / "src_wrapper.pyx").read()
        wrapper_pyx = wrapper_pyx.replace("PYX_DECLARATIONS", pyx_declarations)
        wrapper_pyx = wrapper_pyx.replace("C_DECLARATIONS", c_declarations)
        wrapper_pyx = wrapper_pyx.replace("C_FUNCTION_IMPLEMENTATIONS", c_function_implementations)

        wrapper_pyx = wrapper_pyx.replace("BASE_ITEM_TYPE", node_dtype.base)
        wrapper_pyx = wrapper_pyx.replace("NUM_DIMS", str(dims))

        wrapper = witty.compile_module(
            wrapper_pyx,
            source_files=[
                src_dir / "src" / "rtree.h",
                src_dir / "src" / "rtree.c",
                src_dir / "src" / "config.h",
            ],
            extra_compile_args=["-O3"],
            include_dirs=[str(src_dir)],
            language="c",
            quiet=True,
        )
        RTreeType = type(cls.__name__, (cls, wrapper.RTree), {})
        return wrapper.RTree.__new__(RTreeType)

    def __init__(self, node_dtype, coord_dtype, dims):
        super().__init__()
