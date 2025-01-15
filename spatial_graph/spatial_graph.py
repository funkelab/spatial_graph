from .graph import Graph
from .rtree import PointRTree, LineRTree
from .dtypes import DType
import numpy as np


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
        assert (
            position_attr in node_attr_dtypes
        ), f"position attribute '{position_attr}' not defined in 'node_attr_dtypes'"
        super().__init__(node_dtype, node_attr_dtypes, edge_attr_dtypes, directed)

        self.ndims = ndims
        self.position_attr = position_attr
        self.coord_dtype = DType(node_attr_dtypes[position_attr]).base
        self._node_rtree = PointRTree(node_dtype, self.coord_dtype, ndims)
        self._edge_rtree = LineRTree(f"{node_dtype}[2]", self.coord_dtype, ndims)

    def add_node(self, node, **kwargs):
        position = self._get_position(kwargs)
        self._node_rtree.insert_point_item(node, position)
        super().add_node(node, **kwargs)

    def add_nodes(self, nodes, **kwargs):
        positions = self._get_position(kwargs)
        self._node_rtree.insert_point_items(nodes, positions)
        super().add_nodes(nodes, **kwargs)

    def add_edge(self, edge, **kwargs):
        edge = np.array(edge, dtype=self.node_dtype)
        position_u = getattr(self.node_attrs[edge[0]], self.position_attr)
        position_v = getattr(self.node_attrs[edge[1]], self.position_attr)
        self._edge_rtree.insert_line(edge, position_u, position_v)
        super().add_edge(edge, **kwargs)

    def add_edges(self, edges, **kwargs):
        starts = getattr(self.node_attrs[edges[:, 0]], self.position_attr)
        ends = getattr(self.node_attrs[edges[:, 1]], self.position_attr)
        self._edge_rtree.insert_lines(edges, starts, ends)
        super().add_edges(edges, **kwargs)

    @property
    def roi(self):
        return self._node_rtree.bounding_box()

    def query_nodes_in_roi(self, roi):
        return self._node_rtree.search(roi[0], roi[1])

    def query_edges_in_roi(self, roi):
        return self._edge_rtree.search(roi[0], roi[1])

    def query_nearest_nodes(self, point, k, return_distances=False):
        return self._node_rtree.nearest(point, k, return_distances)

    def query_nearest_edges(self, point, k, return_distances=False):
        return self._edge_rtree.nearest(point, k, return_distances)

    @property
    def nodes(self):
        return super().nodes()

    def remove_nodes(self, nodes):
        positions = getattr(self.node_attrs[nodes], self.position_attr)
        self._node_rtree.delete_items(nodes, positions)
        if not self.directed:
            edges = self.edges_by_nodes(nodes)
        else:
            edges = np.concatenate(
                self.in_edges_by_nodes(nodes), self.out_edges_by_nodes(nodes)
            )
        positions_u = getattr(self.node_attrs[edges[:, 0]], self.position_attr)
        positions_v = getattr(self.node_attrs[edges[:, 1]], self.position_attr)
        self._edge_rtree.delete_items(edges, positions_u, positions_v)
        super().remove_nodes(nodes)

    def _get_position(self, kwargs):
        if self.position_attr in kwargs:
            return kwargs[self.position_attr]
        raise RuntimeError(f"position attribute '{self.position_attr}' not given")
