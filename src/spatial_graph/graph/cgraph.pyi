from collections.abc import Iterator
from typing import Any

import numpy as np

class CGraph:
    def add_node(self, node: Any, *data: Any, **kwargs: Any) -> int:
        """Add a single node to the graph.

        Names/number of args kwargs must match the `node_attr_dtypes`.
        """
    def add_nodes(self, nodes: np.ndarray, *data: Any, **kwargs: Any) -> int:
        """Add multiple nodes to the graph.

        Names/number of args kwargs must match the `node_attr_dtypes`.
        """
    def add_edge(self, edge: np.ndarray, *args: Any, **kwargs: Any) -> int:
        """Add an edge to the graph.

        Names/number of args kwargs must match the `edge_attr_dtypes`.
        """

    def add_edges(
        self, edges: np.ndarray, *args: np.ndarray, **kwargs: np.ndarray
    ) -> int:
        """Add multiple edges to the graph.

        Edges must be 2D and names/number of args kwargs must match the
        `edge_attr_dtypes`.
        """

    def nodes(self) -> np.ndarray:
        """Get all node IDs."""
    def remove_node(self, node: Any) -> None: ...
    def remove_nodes(self, nodes: np.ndarray) -> None: ...
    def nodes_data(
        self, nodes: np.ndarray | None = None
    ) -> Iterator[tuple[Any, Any]]: ...
    def edges_data(self, us: np.ndarray, vs: np.ndarray) -> Iterator: ...
    def num_edges(self) -> int: ...

class UnDirectedCGraph(CGraph):
    def count_neighbors(self, nodes: np.ndarray) -> int: ...
    def edges(self, node: Any = None, data: bool = False) -> Iterator: ...
    def edges_by_nodes(self, nodes: np.ndarray) -> np.ndarray:
        """Same as edges, for fast access to edges incident to an array of nodes."""

class DirectedCGraph(CGraph):
    def count_in_neighbors(self, nodes: np.ndarray) -> int: ...
    def count_out_neighbors(self, nodes: np.ndarray) -> int: ...
    def in_edges(self, node: Any, data: bool): ...
    def in_edges_by_nodes(self, nodes: np.ndarray) -> np.ndarray: ...
    def out_edges(self, node: Any, data: bool): ...
    def out_edges_by_nodes(self, nodes: np.ndarray) -> np.ndarray: ...
