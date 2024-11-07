from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version('spatial_graph')
except PackageNotFoundError:
    __version__ = "unknown"


from .rtree import PointRTree
from .rtree import LineRTree
from .graph import Graph
from .spatial_graph import SpatialGraph


__all__ = ["PointRTree", "LineRTree", "Graph", "SpatialGraph"]
