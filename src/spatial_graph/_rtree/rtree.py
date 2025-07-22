from __future__ import annotations

import sys
from pathlib import Path
from typing import ClassVar

import numpy as np
import witty
from Cheetah.Template import Template

from spatial_graph._dtypes import DType

DEFINE_MACROS = [("RTREE_NOATOMICS", "1")] if sys.platform == "win32" else []
if sys.platform == "win32":  # pragma: no cover
    EXTRA_COMPILE_ARGS = ["/O2"]
else:
    EXTRA_COMPILE_ARGS = ["-O3", "-Wno-unreachable-code"]

SRC_DIR = Path(__file__).parent


def _build_wrapper(
    cls: type[RTree], item_dtype: str, coord_dtype: str, dims: int
) -> str:
    ############################################
    # create wrapper from template and compile #
    ############################################

    wrapper_template = Template(
        file=str(SRC_DIR / "wrapper_template.pyx"),
        compilerSettings={"directiveStartToken": "%"},
    )
    wrapper_template.item_dtype = DType(item_dtype)
    wrapper_template.coord_dtype = DType(coord_dtype)
    wrapper_template.dims = dims
    wrapper_template.c_distance_function = cls.c_distance_function
    wrapper_template.pyx_item_t_declaration = cls.pyx_item_t_declaration
    wrapper_template.c_item_t_declaration = cls.c_item_t_declaration
    wrapper_template.c_converter_functions = cls.c_converter_functions
    wrapper_template.c_equal_function = cls.c_equal_function

    return str(wrapper_template)


def _compile_tree(
    cls: type[RTree], item_dtype: str, coord_dtype: str, dims: int
) -> type:
    wrapper = _build_wrapper(cls, item_dtype, coord_dtype, dims)
    module = witty.compile_cython(
        wrapper,
        source_files=[
            SRC_DIR / "src" / "rtree.h",
            SRC_DIR / "src" / "rtree.c",
            SRC_DIR / "src" / "config.h",
        ],
        extra_compile_args=EXTRA_COMPILE_ARGS,
        include_dirs=[str(SRC_DIR)],
        language="c",
        quiet=True,
        define_macros=DEFINE_MACROS,
    )
    return module.RTree


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

    def __init__(self, item_dtype: str, coord_dtype: str, dims: int):
        super().__init__()
        self.item_dtype = DType(item_dtype)
        self.coord_dtype = DType(coord_dtype)
        self.dims = dims

        tree_cls = _compile_tree(self.__class__, item_dtype, coord_dtype, dims)
        self._ctree = tree_cls()

    def insert_point_item(self, item, position):
        """Insert a single point item.

        To insert multiple points, use `insert_point_items`.
        """
        items = np.array([item], dtype=self.item_dtype.base)
        positions = position[np.newaxis]
        return self._ctree.insert_point_items(items, positions)

    def insert_point_items(self, items, positions):
        """Insert a list of point items.

        Args:

            items (ndarray):

                Array of shape `(n,)` (one scalar per item) or `(n, k)` (one
                array of `k` scalars per item).

            points (ndarray):

                Array of shape `(n, dims)`, the positions of the points to
                insert.
        """
        return self._ctree.insert_point_items(items, positions)

    def delete_item(self, item, bb_min, bb_max=None):
        """Delete a single item.

        To delete multiple items, use `delete_items`.
        """
        items = np.array([item], dtype=self.item_dtype.base)
        bb_mins = bb_min[np.newaxis, :]
        bb_maxs = None if bb_max is None else bb_max[np.newaxis, :]
        return self._ctree.delete_items(items, bb_mins, bb_maxs)

    def delete_items(self, items, bb_mins, bb_maxs=None):
        """Delete items by their content and bounding box.

        Delete items by their content and bounding box. Only items that match
        both the `items` row (a scalar or array, depending on `item_dtype`) and
        the exact coordinates of their bounding box will be deleted.

        Args:

            items (ndarray):

                Array of shape `(n,)` (one scalar per item) or `(n, k)` (one
                array of `k` scalars per item).

            bb_mins/bb_maxs (ndarray):

                Array of shape `(n, dims)`, the minimum/maximum points of the
                bounding boxes per item to delete.
        """
        return self._ctree.delete_items(items, bb_mins, bb_maxs)

    def count(self, bb_min, bb_max):
        """Count the number of items in a bounding box.

        Args:
            bb_min (np.ndarray): The minimum point of the bounding box.
            bb_max (np.ndarray): The maximum point of the bounding box.
        """
        return self._ctree.count(bb_min, bb_max)

    def search(self, bb_min, bb_max):
        """Search for items in a bounding box.

        Args:
            bb_min (np.ndarray): The minimum point of the bounding box.
            bb_max (np.ndarray): The maximum point of the bounding box.
        """
        return self._ctree.search(bb_min, bb_max)

    def nearest(self, point, k=1, return_distances=False):
        """Find the nearest items to a given point.

        Args:

            point (ndarray):

                The coordinates of the query point.

            k (int):

                The maximal number of items to return.

            return_distances (bool):

                If `True`, return a tuple of `(items, distances)`, where
                `distances` contains the distance of each found item to the
                query point.
        """
        return self._ctree.nearest(point, k, return_distances)

    def insert_bb_items(self, items, bb_mins, bb_maxs):
        """Insert items with bounding boxes.
        Args:

        items (ndarray):

            Array of shape `(n,)` (one scalar per item) or `(n, k)` (one
            array of `k` scalars per item).

        bb_mins/bb_maxs (ndarray):

            Array of shape `(n, dims)`, the minimum/maximum points of the
            bounding boxes per item to insert.
        """
        return self._ctree.insert_bb_items(items, bb_mins, bb_maxs)

    def bounding_box(self):
        """Get the total bounding box of all items in this RTree."""
        return self._ctree.bounding_box()

    def __len__(self):
        """Get the number of items in this RTree."""
        return self._ctree.__len__()
