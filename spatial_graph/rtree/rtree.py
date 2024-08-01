import witty
import numpy as np
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
"""

        c_declarations = f"""
// coordinate type to use:
typedef {coord_dtype.to_pyxtype()} coord_t;

// C and PYX items are the same here:
typedef {node_dtype.to_pyxtype()} item_t;
typedef {node_dtype.to_pyxtype()} pyx_item_t;
"""

        c_function_implementations = """
// default PYX<->C converters, just casting
inline item_t convert_pyx_to_c_item(pyx_item_t *pyx_item, coord_t *min, coord_t *max) {
    return (item_t)*pyx_item;
}
inline void copy_c_to_pyx_item(const item_t c_item, pyx_item_t *pyx_item) {
    memcpy(pyx_item, &c_item, sizeof(item_t));
}
        """

        src_dir = Path(__file__).parent
        wrapper_pyx = open(src_dir / "src_wrapper.pyx").read()
        wrapper_pyx = wrapper_pyx.replace("PYX_DECLARATIONS", pyx_declarations)
        wrapper_pyx = wrapper_pyx.replace("C_DECLARATIONS", c_declarations)
        wrapper_pyx = wrapper_pyx.replace(
            "C_FUNCTION_IMPLEMENTATIONS", c_function_implementations
        )

        wrapper_pyx = wrapper_pyx.replace("NP_ITEM_DTYPE", node_dtype.base)
        wrapper_pyx = wrapper_pyx.replace(
            "API_ITEMS_MEMVIEW_TYPE", node_dtype.to_pyxtype(add_dim=True)
        )
        wrapper_pyx = wrapper_pyx.replace("ITEM_LENGTH", "")
        wrapper_pyx = wrapper_pyx.replace("ITEMS_EXTRA_DIMS_0", "")
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
        self.node_dtype = node_dtype

    def insert_point_item(self, item, position):
        items = np.array([item], dtype=self.node_dtype)
        positions = position[np.newaxis]
        return self.insert_point_items(items, positions)

    def delete(self, bb_min, bb_max, item):
        items = np.array([item], dtype=self.node_dtype)
        return self.delete_items(bb_min, bb_max, items)


class EdgeRTree:
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

    # C and PYX items are dfferent here:
    ctypedef {node_dtype.to_pyxtype()} pyx_item_t[2]
    cdef struct item_t:
        {node_dtype.to_pyxtype()} u
        {node_dtype.to_pyxtype()} v
        bool corner_mask[NUM_DIMS]
"""

        c_declarations = f"""
// coordinate type to use:
typedef {coord_dtype.to_pyxtype()} coord_t;

// C and PYX items are different here:
typedef {node_dtype.to_pyxtype()} pyx_item_t[2];
typedef struct item_t {{
    {node_dtype.to_pyxtype()} u;
    {node_dtype.to_pyxtype()} v;
    bool corner_mask[NUM_DIMS];
}} item_t;
"""

        c_function_implementations = """
inline item_t convert_pyx_to_c_item(pyx_item_t *pyx_item, coord_t *from, coord_t *to) {
    item_t item;
    item.u = (*pyx_item)[0];
    item.v = (*pyx_item)[1];
    for (int d = 0; d < NUM_DIMS; d++) {
        item.corner_mask[d] = (from[d] < to[d]);
    }
    return item;
}
inline void copy_c_to_pyx_item(const item_t c_item, pyx_item_t *pyx_item) {
    (*pyx_item)[0] = c_item.u;
    (*pyx_item)[1] = c_item.v;
}
        """

        src_dir = Path(__file__).parent
        wrapper_pyx = open(src_dir / "src_wrapper.pyx").read()
        wrapper_pyx = wrapper_pyx.replace("PYX_DECLARATIONS", pyx_declarations)
        wrapper_pyx = wrapper_pyx.replace("C_DECLARATIONS", c_declarations)
        wrapper_pyx = wrapper_pyx.replace(
            "C_FUNCTION_IMPLEMENTATIONS", c_function_implementations
        )

        wrapper_pyx = wrapper_pyx.replace("NP_ITEM_DTYPE", node_dtype.base)
        wrapper_pyx = wrapper_pyx.replace(
            "API_ITEMS_MEMVIEW_TYPE", f"{node_dtype.to_pyxtype()}[:, ::1]"
        )
        wrapper_pyx = wrapper_pyx.replace("ITEM_LENGTH", "2")
        wrapper_pyx = wrapper_pyx.replace("ITEMS_EXTRA_DIMS_0", ", 0")
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
