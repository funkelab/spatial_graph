"""Rendering of the RTree pyx wrapper.

Used on the JIT path and by the build hook, so prebuilt and JIT-compiled modules
are always generated from the same source. Requires Cheetah, and is therefore
imported lazily by `rtree.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from Cheetah.Template import Template

from spatial_graph._dtypes import DType

from ._naming import SRC_DIR

if TYPE_CHECKING:
    from .rtree import RTree

TEMPLATE = SRC_DIR / "wrapper_template.pyx"


def build_wrapper(
    cls: type[RTree], item_dtype: str, coord_dtype: str, dims: int
) -> str:
    """Render the pyx wrapper for the given tree parameters."""
    wrapper_template = Template(
        file=str(TEMPLATE),
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
