from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("spatial_graph")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"


from ._util import create_graph
from .graph import DiGraph, Graph, GraphBase
from .rtree import LineRTree, PointRTree
from .spatial_graph import SpatialDiGraph, SpatialGraph, SpatialGraphBase

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
