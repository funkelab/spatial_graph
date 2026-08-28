"""What RTree variants get prebuilt, and how their pyx wrappers are rendered.

Used on the JIT path and by `setup.py`, so prebuilt and JIT-compiled modules are
always generated from the same source. Requires Cheetah, and is therefore
imported lazily by `rtree.py`: Cheetah and witty stay installed either way, but
a tree that resolves to a prebuilt module never invokes them, and so never
needs a C compiler.
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

# Variants compiled ahead of time into binary wheels. Both tree classes are
# public API and usable on their own, so both are prebuilt: it is what lets an
# rtree-only user install without a C compiler. Trimming the matrix means
# dropping entries below; `SPATIAL_GRAPH_NO_PREBUILT=1` skips prebuilding
# entirely, for machines that cannot compile at build time.
ITEM_BASES = ("int64", "uint64")
COORD_DTYPES = ("float32", "float64")
DIMS = (2, 3, 4, 5)


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
