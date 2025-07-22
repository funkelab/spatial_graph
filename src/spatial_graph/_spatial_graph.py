from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from spatial_graph._dtypes import DType
from spatial_graph._rtree import LineRTree, PointRTree

from ._graph.graph import DiGraph, Graph, GraphBase

if TYPE_CHECKING:
    from collections.abc import Mapping


class SpatialGraphBase(GraphBase):
    edge_inclusion_values: ClassVar[list[str]] = ["incident", "leaving", "entering"]

    def __init__(
        self,
        ndims: int,
        node_dtype: str,
        node_attr_dtypes: Mapping[str, str] | None = None,
        edge_attr_dtypes: Mapping[str, str] | None = None,
        position_attr: str = "position",
        directed: bool = False,
    ) -> None:
        node_attr_dtypes = node_attr_dtypes or {}
        if position_attr not in node_attr_dtypes:
            raise ValueError(
                f"position attribute {position_attr!r} not defined in "
                "'node_attr_dtypes'"
            )
        super().__init__(node_dtype, node_attr_dtypes, edge_attr_dtypes)

        self.ndims = ndims
        self.position_attr = position_attr
        self.coord_dtype = DType(node_attr_dtypes[position_attr]).base
        self._node_rtree = PointRTree(node_dtype, self.coord_dtype, ndims)
        self._edge_rtree = LineRTree(f"{node_dtype}[2]", self.coord_dtype, ndims)

    def add_node(self, node: Any, *data: Any, **kwargs: Any) -> int:
        position = self._get_position(kwargs)
        self._node_rtree.insert_point_item(node, position)
        return super().add_node(node, *data, **kwargs)

    def add_nodes(self, nodes: np.ndarray, *data: Any, **kwargs: Any) -> int:
        positions = self._get_position(kwargs)
        self._node_rtree.insert_point_items(nodes, positions)
        return super().add_nodes(nodes, *data, **kwargs)

    def add_edge(self, edge: np.ndarray, *args: Any, **kwargs: Any) -> int:
        edge = np.array(edge, dtype=self.node_dtype)
        position_u = getattr(self.node_attrs[edge[0]], self.position_attr)
        position_v = getattr(self.node_attrs[edge[1]], self.position_attr)
        self._edge_rtree.insert_line(edge, position_u, position_v)
        return super().add_edge(edge, **kwargs)

    def add_edges(
        self, edges: np.ndarray, *args: np.ndarray, **kwargs: np.ndarray
    ) -> int:
        starts = getattr(self.node_attrs[edges[:, 0]], self.position_attr)
        ends = getattr(self.node_attrs[edges[:, 1]], self.position_attr)
        self._edge_rtree.insert_lines(edges, starts, ends)
        return super().add_edges(edges, *args, **kwargs)

    @property
    def roi(self):
        return self._node_rtree.bounding_box()

    def query_nodes_in_roi(self, roi):
        return self._node_rtree._ctree.search(roi[0], roi[1])

    def query_edges_in_roi(self, roi):
        return self._edge_rtree._ctree.search(roi[0], roi[1])

    def query_nearest_nodes(self, point, k, return_distances=False):
        return self._node_rtree._ctree.nearest(point, k, return_distances)

    def query_nearest_edges(self, point, k, return_distances=False):
        return self._edge_rtree._ctree.nearest(point, k, return_distances)

    @property
    def edges(self):
        return self.query_edges_in_roi(self.roi)

    def remove_nodes(self, nodes: np.ndarray) -> None:
        positions = getattr(self.node_attrs[nodes], self.position_attr)
        self._node_rtree.delete_items(nodes, positions)
        if isinstance(self, DiGraph):
            edges = np.concatenate(
                self.in_edges_by_nodes(nodes), self.out_edges_by_nodes(nodes)
            )
        elif isinstance(self, Graph):
            edges = self.edges_by_nodes(nodes)
        positions_u = getattr(self.node_attrs[edges[:, 0]], self.position_attr)
        positions_v = getattr(self.node_attrs[edges[:, 1]], self.position_attr)
        self._edge_rtree.delete_items(edges, positions_u, positions_v)
        super().remove_nodes(nodes)

    def _get_position(self, kwargs):
        if self.position_attr in kwargs:
            return kwargs[self.position_attr]
        raise RuntimeError(f"position attribute '{self.position_attr}' not given")


class SpatialGraph(SpatialGraphBase, Graph):
    """Base class for undirected spatial graph instances."""


class SpatialDiGraph(SpatialGraphBase, DiGraph):
    """Base class for directed spatial graph instances."""
