from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

try:
    __version__ = version("spatial_graph")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"


from ._rtree import LineRTree, PointRTree

if TYPE_CHECKING:
    from ._graph import DiGraph, Graph, GraphBase
    from ._spatial_graph import SpatialDiGraph, SpatialGraph, SpatialGraphBase
    from ._util import create_graph

# the graph half is always JIT-compiled, and importing it pulls in witty and
# Cheetah.  Deferring it keeps `PointRTree`/`LineRTree` -- which ship prebuilt --
# usable with numpy alone.
_LAZY = {
    "DiGraph": "._graph",
    "Graph": "._graph",
    "GraphBase": "._graph",
    "SpatialDiGraph": "._spatial_graph",
    "SpatialGraph": "._spatial_graph",
    "SpatialGraphBase": "._spatial_graph",
    "create_graph": "._util",
}

__all__ = [
    "DiGraph",
    "Graph",
    "GraphBase",
    "LineRTree",
    "PointRTree",
    "SpatialDiGraph",
    "SpatialGraph",
    "SpatialGraphBase",
    "create_graph",
]


def __getattr__(name: str) -> Any:
    if module := _LAZY.get(name):
        import importlib

        return getattr(importlib.import_module(module, __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return __all__
