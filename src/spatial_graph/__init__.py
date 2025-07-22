from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("spatial_graph")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"


from ._graph import DiGraph, Graph, GraphBase
from ._graph.spatial_graph import SpatialDiGraph, SpatialGraph, SpatialGraphBase
from ._rtree import LineRTree, PointRTree
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
