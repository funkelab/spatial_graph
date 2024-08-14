import witty
import numpy as np
from pathlib import Path
from ..dtypes import DType


class RTree:
    """A generic RTree implementation, compiled on-the-fly during
    instantiation.

    Args:

        item_dtype (``string``):

            The C type of the items to hold. Can be a scalar (e.g. ``uint64``)
            or an array of scalars (e.g., "uint64[3]").

        coord_dtype (``string``):

            The scalar C type to use for coordinates (e.g., ``float``).

        dims (``int``):

            The dimension of the r-tree.

    Subclassing:

        This generic implementation can be subclassed and modified in the
        following ways:

        The class members ``pyx_item_t_declaration`` and
        ``c_item_t_declaration`` can be overwritten to use custom ``item_t``
        structures. This will also require overwriting the
        ``c_converter_functions`` to translate between the PYX interface (where
        items are scalars or C arrays of scalars) and the C interface (the
        custom ``item_t`` type.

        The following constants and typedefs are available to use in the
        provided code:

            NUM_DIMS:

                A constant set to the value of ``dims``.

            item_base_t:

                The scalar type of the item (e.g., ``uint64``), regardless of
                whether this is a scalar or array item.
    """

    # overwrite in subclasses for custom item_t structures
    pyx_item_t_declaration = None
    c_item_t_declaration = None

    # overwrite in subclasses for custom converters
    c_converter_functions = None

    # overwrite in subclasses for custom distance computation
    c_distance_function = None

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

        #######################################################
        # coord_t, pyx_item_t declaration (PYX and C version) #
        #######################################################

        pyx_declarations = f"""
    ctypedef {coord_dtype.to_pyxtype()} coord_t
    ctypedef {item_dtype.base_c_type} item_base_t
"""
        c_declarations = f"""
typedef {coord_dtype.to_pyxtype()} coord_t;
typedef {item_dtype.base_c_type} item_base_t;
"""

        if item_is_array:
            pyx_declarations += f"""
    ctypedef item_base_t pyx_item_t[{item_size}]
"""

            c_declarations += f"""
typedef item_base_t pyx_item_t[{item_size}];
"""
        else:
            pyx_declarations += """
    ctypedef item_base_t pyx_item_t
"""
            c_declarations += """
typedef item_base_t pyx_item_t;
"""

        ##########################################
        # item_t declaration (PYX and C version) #
        ##########################################

        pyx_item_t_declaration = cls.pyx_item_t_declaration
        c_item_t_declaration = cls.c_item_t_declaration

        if not pyx_item_t_declaration:
            if item_is_array:
                # item_t can't be an array in rtree, arrays can't be assigned
                # to (and this is needed inside rtree). So we make item_t a
                # struct with field `data` to hold the array.
                pyx_item_t_declaration = f"""
    cdef struct item_t:
        item_base_t data[{item_size}]
"""
            else:
                pyx_item_t_declaration = """
    ctypedef item_base_t item_t
"""

        if not c_item_t_declaration:
            if item_is_array:
                c_item_t_declaration = f"""
typedef struct item_t {{
    item_base_t data[{item_size}];
}} item_t;
"""
            else:
                c_item_t_declaration = """
typedef item_base_t item_t;
"""

        pyx_declarations += pyx_item_t_declaration
        c_declarations += c_item_t_declaration

        ####################################
        # pyx_item_t <-> item_t converters #
        ####################################

        c_function_implementations = cls.c_converter_functions

        if not c_function_implementations:
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

        ############################
        # custom distance function #
        ############################

        c_macros = ""
        if cls.c_distance_function:
            c_macros += "#define KNN_USE_EXACT_DISTANCE\n"
            c_function_implementations += cls.c_distance_function

        ############################################
        # create wrapper from template and compile #
        ############################################

        src_dir = Path(__file__).parent
        wrapper_pyx = open(src_dir / "src_wrapper.pyx").read()
        wrapper_pyx = wrapper_pyx.replace("C_MACROS", c_macros)
        wrapper_pyx = wrapper_pyx.replace("PYX_DECLARATIONS", pyx_declarations)
        wrapper_pyx = wrapper_pyx.replace("C_DECLARATIONS", c_declarations)
        wrapper_pyx = wrapper_pyx.replace(
            "C_FUNCTION_IMPLEMENTATIONS", c_function_implementations
        )

        wrapper_pyx = wrapper_pyx.replace("NP_ITEM_DTYPE", item_dtype.base)
        wrapper_pyx = wrapper_pyx.replace("NP_COORD_DTYPE", coord_dtype.base)
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
