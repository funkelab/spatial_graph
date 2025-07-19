from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("spatial_graph")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"


from .graph import Graph
from .rtree import LineRTree, PointRTree
from .spatial_graph import SpatialGraph

__all__ = ["Graph", "LineRTree", "PointRTree", "SpatialGraph"]
