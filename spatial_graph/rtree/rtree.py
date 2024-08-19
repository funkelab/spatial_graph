import witty
import numpy as np
from Cheetah.Template import Template
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

            DIMS:

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

        wrapper = witty.compile_module(
            str(wrapper_template),
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
