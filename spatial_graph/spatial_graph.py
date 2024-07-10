from .graph import Graph
from .rtree import RTree
from .dtypes import DType


class SpatialGraph(Graph):
    edge_inclusion_values = ["incident", "leaving", "entering"]

    def __init__(
        self,
        ndims,
        node_dtype,
        node_attr_dtypes,
        edge_attr_dtypes,
        position_attr,
        directed=False,
    ):
        assert position_attr in node_attr_dtypes, (
            f"position attribute '{position_attr}' not defined in " "'node_attr_dtypes'"
        )
        super().__init__(node_dtype, node_attr_dtypes, edge_attr_dtypes, directed)

        self.ndims = ndims
        self.position_attr = position_attr
        coord_dtype = DType(node_attr_dtypes[position_attr]).base
        self._rtree = RTree(node_dtype, coord_dtype, ndims)

    def add_node(self, node, **kwargs):
        position = self._get_position(kwargs)
        self._rtree.insert_point(node, position)
        super().add_node(node, **kwargs)

    def add_nodes(self, nodes, **kwargs):
        positions = self._get_position(kwargs)
        self._rtree.insert_points(nodes, positions)
        super().add_nodes(nodes, **kwargs)

    def query_in_roi(self, roi, edge_inclusion=None):
        nodes = self._rtree.search(roi[0], roi[1])

        if not edge_inclusion:
            return nodes

        if edge_inclusion not in SpatialGraph.edge_inclusion_values:
            raise ValueError("edge_inclusion has to be in {edge_inclusion_values}")

        if edge_inclusion == "incident":
            return nodes, []  # TODO
        elif edge_inclusion == "leaving":
            return nodes, []  # TODO
        elif edge_inclusion == "entering":
            return nodes, []  # TODO

    def _get_position(self, kwargs):
        if self.position_attr in kwargs:
            return kwargs[self.position_attr]
        raise RuntimeError(f"position attribute '{self.position_attr}' not given")
