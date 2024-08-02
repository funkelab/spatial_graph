import witty
import numpy as np
from pathlib import Path
from ..dtypes import DType


class RTree:
    def __new__(
        cls,
        item_dtype,
        coord_dtype,
        dims,
    ):
        item_dtype = DType(item_dtype)
        item_is_array = item_dtype.is_array
        item_size = item_dtype.size
        coord_dtype = DType(coord_dtype)

        if item_is_array:

            # item_t can't be an array in rtree, arrays can't be assigned to
            # (and this is needed inside rtree). So we make item_t a struct
            # with field `data` to hold the array.
            pyx_declarations = f"""
    ctypedef {coord_dtype.to_pyxtype()} coord_t
    ctypedef {item_dtype.to_pyxtype()} pyx_item_t[{item_size}]
    cdef struct item_t:
        {item_dtype.base_c_type} data[{item_size}]
"""
            c_declarations = f"""
typedef {coord_dtype.to_pyxtype()} coord_t;
typedef {item_dtype.base_c_type} pyx_item_t[{item_size}];
typedef struct item_t {{
    {item_dtype.base_c_type} data[{item_size}];
}} item_t;
"""
        else:
            pyx_declarations = f"""
    # coordinate type to use:
    ctypedef {coord_dtype.to_pyxtype()} coord_t

    # C and PYX items are the same by default:
    ctypedef {item_dtype.to_pyxtype()} item_t
    ctypedef {item_dtype.to_pyxtype()} pyx_item_t
"""
            c_declarations = f"""
typedef {coord_dtype.to_pyxtype()} coord_t;
typedef {item_dtype.to_pyxtype()} item_t;
typedef {item_dtype.to_pyxtype()} pyx_item_t;
"""

        if item_is_array:

            c_function_implementations = """
inline item_t convert_pyx_to_c_item(pyx_item_t *pyx_item, coord_t *min, coord_t *max) {
    item_t c_item;
    memcpy(&c_item, *pyx_item, sizeof(item_t));
    return c_item;
}
inline void copy_c_to_pyx_item(const item_t c_item, pyx_item_t *pyx_item) {
    memcpy(pyx_item, &c_item, sizeof(item_t));
}
        """

        else:

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

        wrapper_pyx = wrapper_pyx.replace("NP_ITEM_DTYPE", item_dtype.base)
        wrapper_pyx = wrapper_pyx.replace(
            "API_ITEMS_MEMVIEW_TYPE", item_dtype.to_pyxtype(add_dim=True)
        )
        if item_size:
            wrapper_pyx = wrapper_pyx.replace("ITEM_LENGTH", str(item_size))
            wrapper_pyx = wrapper_pyx.replace("ITEMS_EXTRA_DIMS_0", ", 0")
        else:
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

    def __init__(self, item_dtype, coord_dtype, dims):
        super().__init__()
        self.item_dtype = DType(item_dtype)

    def insert_point_item(self, item, position):
        items = np.array([item], dtype=self.item_dtype.base)
        positions = position[np.newaxis]
        return self.insert_point_items(items, positions)

    def delete(self, bb_min, bb_max, item):
        items = np.array([item], dtype=self.item_dtype.base)
        return self.delete_items(bb_min, bb_max, items)


class EdgeRTree:
    def __new__(
        cls,
        item_dtype,
        coord_dtype,
        dims,
    ):
        item_dtype = DType(item_dtype)
        coord_dtype = DType(coord_dtype)

        pyx_declarations = f"""
    # coordinate type to use:
    ctypedef {coord_dtype.to_pyxtype()} coord_t

    # C and PYX items are dfferent here:
    ctypedef {item_dtype.to_pyxtype()} pyx_item_t[2]
    cdef struct item_t:
        {item_dtype.to_pyxtype()} u
        {item_dtype.to_pyxtype()} v
        bool corner_mask[NUM_DIMS]
"""

        c_declarations = f"""
// coordinate type to use:
typedef {coord_dtype.to_pyxtype()} coord_t;

// C and PYX items are different here:
typedef {item_dtype.to_pyxtype()} pyx_item_t[2];
typedef struct item_t {{
    {item_dtype.to_pyxtype()} u;
    {item_dtype.to_pyxtype()} v;
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

        wrapper_pyx = wrapper_pyx.replace("NP_ITEM_DTYPE", item_dtype.base)
        wrapper_pyx = wrapper_pyx.replace(
            "API_ITEMS_MEMVIEW_TYPE", f"{item_dtype.to_pyxtype()}[:, ::1]"
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

    def __init__(self, item_dtype, coord_dtype, dims):
        super().__init__()
