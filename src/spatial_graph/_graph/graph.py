from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

from .graph_base import GraphBase

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np


class Graph(GraphBase):
    directed: Literal[False] = False

    def num_neighbors(self, nodes: np.ndarray) -> np.ndarray:
        """Return the number of neighbors for each node.

        For undirected graphs, this counts all adjacent nodes regardless
        of edge direction since edges are bidirectional.

        Parameters
        ----------
        nodes : np.ndarray
            Array of node identifiers to count neighbors for.

        Returns
        -------
        np.ndarray
            Array of neighbor counts for each node in the input array.
        """
        return self._cgraph.num_neighbors(nodes)

    @overload
    def edges(
        self, node: Any = ..., data: Literal[True] = ...
    ) -> Iterator[tuple[tuple, Any]]: ...
    @overload
    def edges(self, node: Any = ..., data: Literal[False] = ...) -> Iterator[tuple]: ...
    def edges(self, node: Any = None, data: bool = False) -> Iterator[tuple]:
        """Iterate over edges in the graph.

        For undirected graphs, each edge is yielded only once with nodes
        ordered such that node1 < node2 to avoid duplicates.

        Parameters
        ----------
        node : Any, optional
            If provided, only iterate over edges incident to this node.
            If None, iterate over all edges in the graph.
        data : bool, default False
            If True, yield (edge, edge_data) tuples. If False, yield
            only edge tuples.

        Yields
        ------
        tuple or tuple[tuple, Any]
            If `data=False`: tuples of (node1, node2) representing edges.
            If `data=True`: tuples of ((node1, node2), edge_data) where
            edge_data provides access to edge attributes.
        """
        return self._cgraph.edges(node, data)

    def edges_by_nodes(self, nodes: np.ndarray) -> np.ndarray:
        """Get all edges incident to the specified nodes.

        This method provides fast access to edges incident to an array
        of nodes. Note that edges between nodes in the input array will
        be reported multiple times (once for each incident node).

        Parameters
        ----------
        nodes : np.ndarray
            Array of node identifiers to find incident edges for.

        Returns
        -------
        np.ndarray
            2D array of shape (n_edges, 2) where each row contains
            [node1, node2] representing an edge. For undirected graphs,
            node1 <= node2.
        """
        return self._cgraph.edges_by_nodes(nodes)


class DiGraph(GraphBase):
    directed: Literal[True] = True

    def num_in_neighbors(self, nodes: np.ndarray) -> np.ndarray:
        """Return the number of incoming neighbors for each node.

        This counts only nodes that have edges pointing to the specified nodes
        (i.e., predecessors).

        Parameters
        ----------
        nodes : np.ndarray
            Array of node identifiers to count incoming neighbors for.

        Returns
        -------
        np.ndarray
            Array of incoming neighbor counts for each node in the input array.
        """
        return self._cgraph.num_in_neighbors(nodes)

    def num_out_neighbors(self, nodes: np.ndarray) -> np.ndarray:
        """Return the number of outgoing neighbors for each node.

        This counts only nodes that the specified nodes
        have edges pointing to (i.e., successors).

        Parameters
        ----------
        nodes : np.ndarray
            Array of node identifiers to count outgoing neighbors for.

        Returns
        -------
        np.ndarray
            Array of outgoing neighbor counts for each node in the input array.
        """
        return self._cgraph.num_out_neighbors(nodes)

    @overload
    def in_edges(
        self, node: Any = ..., data: Literal[True] = ...
    ) -> Iterator[tuple[tuple, Any]]: ...
    @overload
    def in_edges(
        self, node: Any = ..., data: Literal[False] = ...
    ) -> Iterator[tuple]: ...
    def in_edges(self, node: Any = None, data: bool = False) -> Iterator[tuple]:
        """Iterate over incoming edges to a node.

        Only edges directed toward the specified node are yielded.

        Parameters
        ----------
        node : Any
            The target node to find incoming edges for.
        data : bool
            If True, yield (edge, edge_data) tuples. If False, yield
            only edge tuples.

        Yields
        ------
        tuple or tuple[tuple, Any]
            If `data=False`: tuples of (source_node, target_node) representing
            incoming edges where target_node is the specified node.
            If `data=True`: tuples of ((source_node, target_node), edge_data)
            where edge_data provides access to edge attributes.
        """
        return self._cgraph.in_edges(node, data)

    def in_edges_by_nodes(self, nodes: np.ndarray) -> np.ndarray:
        """Get all incoming edges to the specified nodes.

        This method provides fast access to incoming edges for an array
        of nodes. Edges between nodes in the input array will be reported
        multiple times if both source and target are in the array.

        Parameters
        ----------
        nodes : np.ndarray
            Array of node identifiers to find incoming edges for.

        Returns
        -------
        np.ndarray
            2D array of shape (n_edges, 2) where each row contains
            [source_node, target_node] representing an incoming edge
            to one of the specified nodes.
        """
        return self._cgraph.in_edges_by_nodes(nodes)

    @overload
    def out_edges(
        self, node: Any = ..., data: Literal[True] = ...
    ) -> Iterator[tuple[tuple, Any]]: ...
    @overload
    def out_edges(
        self, node: Any = ..., data: Literal[False] = ...
    ) -> Iterator[tuple]: ...
    def out_edges(self, node: Any = None, data: bool = False) -> Iterator[tuple]:
        """Iterate over outgoing edges from a node.

        Only edges directed away from the specified node are yielded.

        Parameters
        ----------
        node : Any
            The source node to find outgoing edges for.
        data : bool
            If True, yield (edge, edge_data) tuples. If False, yield
            only edge tuples.

        Yields
        ------
        tuple or tuple[tuple, Any]
            If `data=False`: tuples of (source_node, target_node) representing
            outgoing edges where source_node is the specified node.
            If `data=True`: tuples of ((source_node, target_node), edge_data)
            where edge_data provides access to edge attributes.
        """
        return self._cgraph.out_edges(node, data)

    def out_edges_by_nodes(self, nodes: np.ndarray) -> np.ndarray:
        """Get all outgoing edges from the specified nodes.

        This method provides fast access to outgoing edges for an array
        of nodes. Edges between nodes in the input array will be reported
        multiple times if both source and target are in the array.

        Parameters
        ----------
        nodes : np.ndarray
            Array of node identifiers to find outgoing edges for.

        Returns
        -------
        np.ndarray
            2D array of shape (n_edges, 2) where each row contains
            [source_node, target_node] representing an outgoing edge
            from one of the specified nodes.
        """
        return self._cgraph.out_edges_by_nodes(nodes)
