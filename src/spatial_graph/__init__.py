from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("spatial_graph")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"


from ._graph import DiGraph, Graph, GraphBase
from ._rtree import LineRTree, PointRTree
from ._spatial_graph import SpatialDiGraph, SpatialGraph, SpatialGraphBase
from ._util import create_graph

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
