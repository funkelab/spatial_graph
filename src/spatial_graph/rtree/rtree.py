import sys
from pathlib import Path
from typing import ClassVar

import numpy as np
import witty
from Cheetah.Template import Template

from spatial_graph.dtypes import DType

DEFINE_MACROS = [("RTREE_NOATOMICS", "1")] if sys.platform == "win32" else []
if sys.platform == "win32":
    EXTRA_COMPILE_ARGS = ["/O2"]
else:
    EXTRA_COMPILE_ARGS = ["-O3", "-Wno-unreachable-code"]


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

            DIMS:

                A constant set to the value of ``dims``.

            item_base_t:

                The scalar type of the item (e.g., ``uint64``), regardless of
                whether this is a scalar or array item.

    insert_point_items():

        Args:

            items (ndarray):

                Array of shape `(n,)` (one scalar per item) or `(n, k)` (one
                array of `k` scalars per item).

            points (ndarray):

                Array of shape `(n, dims)`, the positions of the points to
                insert.

    insert_bb_items():

        Args:

            items (ndarray):

                Array of shape `(n,)` (one scalar per item) or `(n, k)` (one
                array of `k` scalars per item).

            bb_mins/bb_maxs (ndarray):

                Array of shape `(n, dims)`, the minimum/maximum points of the
                bounding boxes per item to insert.

    count():

        Count the number of items in a bounding box.

        Args:

            bb_min/bb_max (ndarray):

                Array of shape `(dims,)`, the minimum/maximum point of the
                bounding box to count items in.

    bounding_box():

        Get the total bounding box of all items in this RTree.

    search():

        Get all items contained in a bounding box.

        Args:

            bb_min/bb_max (ndarray):

                Array of shape `(dims,)`, the minimum/maximum point of the
                bounding box to search items in.

    nearest():

        Get the nearest items to a given point.

        Args:

            point (ndarray):

                The coordinates of the query point.

            k (int):

                The maximal number of items to return.

            return_distances (bool):

                If `True`, return a tuple of `(items, distances)`, where
                `distances` contains the distance of each found item to the
                query point.

    delete_items():

        Delete items by their content and bounding box. Only items that match
        both the `items` row (a scalar or array, depending on `item_dtype`) and
        the exact coordinates of their bounding box will be deleted.

        Args:

            items (ndarray):

                Array of shape `(n,)` (one scalar per item) or `(n, k)` (one
                array of `k` scalars per item).

            bb_mins/bb_maxs (ndarray):

                Array of shape `(n, dims)`, the minimum/maximum points of the
                bounding boxes per item to delete. `bb_maxs` is optional for
                point items, where min and max are the same.

    __len__():

        Get the number of items in this RTree.

    """

    # overwrite in subclasses for custom item_t structures
    pyx_item_t_declaration: ClassVar[str] = ""
    c_item_t_declaration: ClassVar[str] = ""

    # overwrite in subclasses for custom converters
    c_converter_functions: ClassVar[str] = ""

    # overwrite in subclasses for custom item comparison code
    c_equal_function: ClassVar[str] = ""

    # overwrite in subclasses for custom distance computation
    c_distance_function: ClassVar[str] = ""

    def __new__(
        cls,
        item_dtype,
        coord_dtype,
        dims,
    ):
        item_dtype = DType(item_dtype)
        coord_dtype = DType(coord_dtype)

        ############################################
        # create wrapper from template and compile #
        ############################################

        src_dir = Path(__file__).parent
        wrapper_template = Template(
            file=str(src_dir / "wrapper_template.pyx"),
            compilerSettings={"directiveStartToken": "%"},
        )
        wrapper_template.item_dtype = item_dtype
        wrapper_template.coord_dtype = coord_dtype
        wrapper_template.dims = dims
        wrapper_template.c_distance_function = cls.c_distance_function
        wrapper_template.pyx_item_t_declaration = cls.pyx_item_t_declaration
        wrapper_template.c_item_t_declaration = cls.c_item_t_declaration
        wrapper_template.c_converter_functions = cls.c_converter_functions
        wrapper_template.c_equal_function = cls.c_equal_function

        wrapper = witty.compile_module(
            str(wrapper_template),
            source_files=[
                src_dir / "src" / "rtree.h",
                src_dir / "src" / "rtree.c",
                src_dir / "src" / "config.h",
            ],
            extra_compile_args=EXTRA_COMPILE_ARGS,
            include_dirs=[str(src_dir)],
            language="c",
            quiet=True,
            define_macros=DEFINE_MACROS,
        )
        RTreeType = type(cls.__name__, (cls, wrapper.RTree), {})
        return wrapper.RTree.__new__(RTreeType)

    def __init__(self, item_dtype, coord_dtype, dims):
        super().__init__()
        self.item_dtype = DType(item_dtype)

    def insert_point_item(self, item, position):
        """Insert a single point item.

        To insert multiple points, use `insert_point_items`.
        """
        items = np.array([item], dtype=self.item_dtype.base)
        positions = position[np.newaxis]
        return self.insert_point_items(items, positions)

    def delete_item(self, item, bb_min, bb_max=None):
        """Delete a single item.

        To delete multiple items, use `delete_items`.
        """
        items = np.array([item], dtype=self.item_dtype.base)
        bb_mins = bb_min[np.newaxis, :]
        bb_maxs = None if bb_max is None else bb_max[np.newaxis, :]
        return self.delete_items(items, bb_mins, bb_maxs)
