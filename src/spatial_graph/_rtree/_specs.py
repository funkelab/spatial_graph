"""The set of RTree variants compiled ahead of time into binary wheels.

Only `PointRTree` is prebuilt by default: `LineRTree` is only ever used by
`SpatialGraph`, whose graph half is JIT-compiled regardless, so prebuilding it
would double the wheel size without removing anyone's compiler requirement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from .line_rtree import LineRTree
from .point_rtree import PointRTree

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .rtree import RTree

ITEM_BASES = ("int64", "uint64")
COORD_DTYPES = ("float32", "float64")
DIMS = (2, 3, 4, 5)
PREBUILT_LINE_TREES = False


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
