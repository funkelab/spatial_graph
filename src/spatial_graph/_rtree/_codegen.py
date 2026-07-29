"""What RTree variants get prebuilt, and how their pyx wrappers are rendered.

Used on the JIT path and by `setup.py`, so prebuilt and JIT-compiled modules are
always generated from the same source. Requires Cheetah, and is therefore
imported lazily by `rtree.py` -- installs that stay on the prebuilt path need
neither Cheetah nor witty.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from Cheetah.Template import Template

from spatial_graph._dtypes import DType

from .line_rtree import LineRTree
from .point_rtree import PointRTree

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .rtree import RTree

TEMPLATE = Path(__file__).parent / "wrapper_template.pyx"

# Variants compiled ahead of time into binary wheels. `PointRTree` is what makes
# a compiler unnecessary for rtree-only users; `LineRTree` is only reached via
# `SpatialGraph`, whose graph half is JIT-compiled regardless, so prebuilding it
# saves first-use compile time rather than removing a requirement.
ITEM_BASES = ("int64", "uint64")
COORD_DTYPES = ("float32", "float64")
DIMS = (2, 3, 4, 5)
PREBUILT_LINE_TREES = True


class Spec(NamedTuple):
    cls: type[RTree]
    item_dtype: str
    coord_dtype: str
    dims: int


def iter_specs() -> Iterator[Spec]:
    """Yield every RTree variant that should be compiled into a wheel."""
    for base in ITEM_BASES:
        for coord in COORD_DTYPES:
            for dims in DIMS:
                yield Spec(PointRTree, base, coord, dims)
                if PREBUILT_LINE_TREES:
                    yield Spec(LineRTree, f"{base}[2]", coord, dims)


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
