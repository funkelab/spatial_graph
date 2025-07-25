from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import witty
from Cheetah.Template import Template

from spatial_graph._dtypes import DType

from .views import EdgeAttrs, NodeAttrs

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    import numpy as np


# Set platform-specific compile arguments
if sys.platform == "win32":  # pragma: no cover
    # Use /O2 for optimization and /std:c++20 for C++20
    EXTRA_COMPILE_ARGS = ["/O2", "/std:c++20", "/wd4101"]
else:
    # -O3 for optimization and -std=c++20 for C++20
    EXTRA_COMPILE_ARGS = [
        "-O3",
        "-std=c++20",
        "-Wno-unused-variable",
        "-Wno-unreachable-code",
    ]

SRC_DIR = Path(__file__).parent


def _build_wrapper(
    node_dtype: str,
    node_attr_dtypes: Mapping[str, str] | None = None,
    edge_attr_dtypes: Mapping[str, str] | None = None,
    directed: bool = False,
) -> str:
    if node_attr_dtypes is None:
        node_attr_dtypes = {}
    if edge_attr_dtypes is None:
        edge_attr_dtypes = {}
    if not all(str.isidentifier(name) for name in node_attr_dtypes):
        raise ValueError("Node attribute names must be valid identifiers")
    if not all(str.isidentifier(name) for name in edge_attr_dtypes):
        raise ValueError("Edge attribute names must be valid identifiers")

    wrapper_template = Template(
        file=str(SRC_DIR / "wrapper_template.pyx"),
        compilerSettings={"directiveStartToken": "%"},
    )
    wrapper_template.node_dtype = DType(node_dtype)
    wrapper_template.node_attr_dtypes = {
        name: DType(dtype) for name, dtype in node_attr_dtypes.items()
    }
    wrapper_template.edge_attr_dtypes = {
        name: DType(dtype) for name, dtype in edge_attr_dtypes.items()
    }
    wrapper_template.directed = directed

    return str(wrapper_template)


def _compile_graph(
    node_dtype: str,
    node_attr_dtypes: Mapping[str, str] | None = None,
    edge_attr_dtypes: Mapping[str, str] | None = None,
    directed: bool = False,
) -> type:
    wrapper_template = _build_wrapper(
        node_dtype=node_dtype,
        node_attr_dtypes=node_attr_dtypes,
        edge_attr_dtypes=edge_attr_dtypes,
        directed=directed,
    )
    wrapper = witty.compile_cython(
        wrapper_template,
        source_files=[str(SRC_DIR / "src" / "graph_lite.h")],
        extra_compile_args=EXTRA_COMPILE_ARGS,
        include_dirs=[str(SRC_DIR)],
        language="c++",
        quiet=True,
    )
    return wrapper.Graph


class GraphBase:
    directed: bool = False

    def __init__(
        self,
        node_dtype: str,
        node_attr_dtypes: Mapping[str, str] | None = None,
        edge_attr_dtypes: Mapping[str, str] | None = None,
    ):
        super().__init__()
        self.node_dtype = node_dtype
        self.node_attr_dtypes = node_attr_dtypes or {}
        self.edge_attr_dtypes = edge_attr_dtypes or {}

        cgraph_cls = _compile_graph(
            node_dtype=self.node_dtype,
            node_attr_dtypes=self.node_attr_dtypes,
            edge_attr_dtypes=self.edge_attr_dtypes,
            directed=self.directed,
        )
        self._cgraph = cgraph_cls()

        self.node_attrs = NodeAttrs(self)
        self.edge_attrs = EdgeAttrs(self)

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
        return self._cgraph.add_node(node, *data, **kwargs)

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
        return self._cgraph.add_nodes(nodes, *data, **kwargs)

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
        return self._cgraph.add_edge(edge, *args, **kwargs)

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
        return self._cgraph.add_edges(edges, *args, **kwargs)

    @property
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
        return self._cgraph.nodes()

    def remove_node(self, node: Any) -> None:
        """Remove a single node from the graph.

        Removing a node will also remove all edges incident to that node.

        Parameters
        ----------
        node : Any
            The node identifier to remove from the graph.
        """
        return self._cgraph.remove_node(node)

    def remove_nodes(self, nodes: np.ndarray) -> None:
        """Remove multiple nodes from the graph.

        Removing nodes will also remove all edges incident to those nodes.

        Parameters
        ----------
        nodes : np.ndarray
            Array of node identifiers to remove from the graph.
        """
        return self._cgraph.remove_nodes(nodes)

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
        return self._cgraph.nodes_data(nodes)

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
        return self._cgraph.edges_data(us, vs)

    def num_edges(self) -> int:
        """Get the total number of edges in the graph.

        Returns
        -------
        int
            The number of edges in the graph.
        """
        return self._cgraph.num_edges()

    def __len__(self) -> int:
        """Return the number of nodes in the graph.

        Returns
        -------
        int
            The number of nodes in the graph.
        """
        return self._cgraph.__len__()
