from collections.abc import Iterator
from typing import Any

import numpy as np

class CGraph:
    def add_node(self, node: Any, *data: Any, **kwargs: Any) -> int:
        """Add a single node to the graph.

        The node attributes provided via *data and **kwargs must match the
        data types and names specified in `node_attr_dtypes` when the graph
        was created.

        Parameters
        ----------
        node : Any
            The node identifier to add to the graph.
        *data : Any
            Positional arguments for node attributes. Names/number of args
            must match the `node_attr_dtypes`.
        **kwargs : Any
            Keyword arguments for node attributes. Names/number of kwargs
            must match the `node_attr_dtypes`.

        Returns
        -------
        int
            Number of nodes added (1 if successful, 0 if node already exists).
        """
    def add_nodes(self, nodes: np.ndarray, *data: Any, **kwargs: Any) -> int:
        """Add multiple nodes to the graph.

        Node attributes provided via *data and **kwargs must match the
        data types and names specified in `node_attr_dtypes`. Each attribute
        array must have the same length as the `nodes` array.

        Parameters
        ----------
        nodes : np.ndarray
            Array of node identifiers to add to the graph.
        *data : Any
            Positional arguments for node attributes. Each argument should be
            an array with length matching `nodes`. Names/number of args must
            match the `node_attr_dtypes`.
        **kwargs : Any
            Keyword arguments for node attributes. Each argument should be
            an array with length matching `nodes`. Names/number of kwargs
            must match the `node_attr_dtypes`.

        Returns
        -------
        int
            Number of nodes successfully added.
        """
    def add_edge(self, edge: np.ndarray, *args: Any, **kwargs: Any) -> int:
        """Add an edge to the graph.

        The edge attributes provided via *args and **kwargs must match the
        data types and names specified in `edge_attr_dtypes` when the graph
        was created.

        Parameters
        ----------
        edge : np.ndarray
            Array of length 2 containing [source_node, target_node].
        *args : Any
            Positional arguments for edge attributes. Names/number of args
            must match the `edge_attr_dtypes`.
        **kwargs : Any
            Keyword arguments for edge attributes. Names/number of kwargs
            must match the `edge_attr_dtypes`.

        Returns
        -------
        int
            Number of edges added (1 if successful, 0 if edge already exists).
        """

    def add_edges(
        self, edges: np.ndarray, *args: np.ndarray, **kwargs: np.ndarray
    ) -> int:
        """Add multiple edges to the graph.

        Edge attributes provided via *args and **kwargs must match the
        data types and names specified in `edge_attr_dtypes`. Each attribute
        array must have the same length as the number of edges.

        Parameters
        ----------
        edges : np.ndarray
            2D array of shape (n_edges, 2) where each row contains
            [source_node, target_node].
        *args : np.ndarray
            Positional arguments for edge attributes. Each argument should be
            an array with length matching the number of edges. Names/number
            of args must match the `edge_attr_dtypes`.
        **kwargs : np.ndarray
            Keyword arguments for edge attributes. Each argument should be
            an array with length matching the number of edges. Names/number
            of kwargs must match the `edge_attr_dtypes`.

        Returns
        -------
        int
            Number of edges successfully added.
        """

    def nodes(self) -> np.ndarray:
        """Get all node IDs in the graph.

        The returned array is a copy and modifications will not affect
        the graph structure.

        Returns
        -------
        np.ndarray
            Array containing all node identifiers in the graph, ordered
            by insertion order (earliest added first).
        """
    def remove_node(self, node: Any) -> None:
        """Remove a single node from the graph.

        Removing a node will also remove all edges incident to that node.

        Parameters
        ----------
        node : Any
            The node identifier to remove from the graph.
        """
    def remove_nodes(self, nodes: np.ndarray) -> None:
        """Remove multiple nodes from the graph.

        Removing nodes will also remove all edges incident to those nodes.

        Parameters
        ----------
        nodes : np.ndarray
            Array of node identifiers to remove from the graph.
        """
    def nodes_data(self, nodes: np.ndarray | None = None) -> Iterator[tuple[Any, Any]]:
        """Iterate over nodes and their associated data.

        The node_data object provides access to node attributes as defined
        by the `node_attr_dtypes` when the graph was created.

        Parameters
        ----------
        nodes : np.ndarray, optional
            Array of specific node identifiers to iterate over. If None,
            iterates over all nodes in the graph.

        Yields
        ------
        tuple[Any, Any]
            Tuples of (node_id, node_data) where node_data is a view object
            providing access to the node's attributes.
        """
    def edges_data(self, us: np.ndarray, vs: np.ndarray) -> Iterator:
        """Iterate over edge data for specified edges.

        The arrays `us` and `vs` must have the same length. The edge data
        objects provide access to edge attributes as defined by the
        `edge_attr_dtypes` when the graph was created.

        Parameters
        ----------
        us : np.ndarray
            Array of source node identifiers.
        vs : np.ndarray
            Array of target node identifiers.

        Yields
        ------
        Any
            Edge data view objects providing access to edge attributes
            for each edge (us[i], vs[i]).
        """
    def num_edges(self) -> int:
        """Get the total number of edges in the graph.

        Returns
        -------
        int
            The number of edges in the graph.
        """
    def __len__(self) -> int:
        """Return the number of nodes in the graph.

        Returns
        -------
        int
            The number of nodes in the graph.
        """

class UnDirectedCGraph(CGraph):
    def count_neighbors(self, nodes: np.ndarray) -> int:
        """Count the number of neighbors for each node.

        For undirected graphs, this counts all adjacent nodes regardless
        of edge direction since edges are bidirectional.

        Parameters
        ----------
        nodes : np.ndarray
            Array of node identifiers to count neighbors for.

        Returns
        -------
        int
            Array of neighbor counts for each node in the input array.
        """
    def edges(self, node: Any = None, data: bool = False) -> Iterator:
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
            If data=False: tuples of (node1, node2) representing edges.
            If data=True: tuples of ((node1, node2), edge_data) where
            edge_data provides access to edge attributes.
        """
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

        Notes
        -----

        """

class DirectedCGraph(CGraph):
    def count_in_neighbors(self, nodes: np.ndarray) -> int:
        """Count the number of incoming neighbors for each node.

        Parameters
        ----------
        nodes : np.ndarray
            Array of node identifiers to count incoming neighbors for.

        Returns
        -------
        int
            Array of incoming neighbor counts for each node in the input array.

        Notes
        -----
        For directed graphs, this counts only nodes that have edges pointing
        to the specified nodes (i.e., predecessors).
        """
    def count_out_neighbors(self, nodes: np.ndarray) -> int:
        """Count the number of outgoing neighbors for each node.

        Parameters
        ----------
        nodes : np.ndarray
            Array of node identifiers to count outgoing neighbors for.

        Returns
        -------
        int
            Array of outgoing neighbor counts for each node in the input array.

        Notes
        -----
        For directed graphs, this counts only nodes that the specified nodes
        have edges pointing to (i.e., successors).
        """
    def in_edges(self, node: Any, data: bool):
        """Iterate over incoming edges to a node.

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
            If data=False: tuples of (source_node, target_node) representing
            incoming edges where target_node is the specified node.
            If data=True: tuples of ((source_node, target_node), edge_data)
            where edge_data provides access to edge attributes.

        Notes
        -----
        Only edges directed toward the specified node are yielded.
        """
    def in_edges_by_nodes(self, nodes: np.ndarray) -> np.ndarray:
        """Get all incoming edges to the specified nodes.

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

        Notes
        -----
        This method provides fast access to incoming edges for an array
        of nodes. Edges between nodes in the input array will be reported
        multiple times if both source and target are in the array.
        """
    def out_edges(self, node: Any, data: bool):
        """Iterate over outgoing edges from a node.

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
            If data=False: tuples of (source_node, target_node) representing
            outgoing edges where source_node is the specified node.
            If data=True: tuples of ((source_node, target_node), edge_data)
            where edge_data provides access to edge attributes.

        Notes
        -----
        Only edges directed away from the specified node are yielded.
        """
    def out_edges_by_nodes(self, nodes: np.ndarray) -> np.ndarray:
        """Get all outgoing edges from the specified nodes.

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

        Notes
        -----
        This method provides fast access to outgoing edges for an array
        of nodes. Edges between nodes in the input array will be reported
        multiple times if both source and target are in the array.
        """
