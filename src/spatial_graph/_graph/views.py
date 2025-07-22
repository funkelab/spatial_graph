from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .graph_base import GraphBase


class NodeAttrsView:
    graph: GraphBase
    nodes: np.ndarray | None

    def __init__(
        self, graph: GraphBase, nodes: np.ndarray | Iterable | None = None
    ) -> None:
        super().__setattr__("graph", graph)
        for name in graph.node_attr_dtypes.keys():
            super().__setattr__(
                f"get_attr_{name}", getattr(graph._cgraph, f"get_nodes_data_{name}")
            )
            super().__setattr__(
                f"set_attr_{name}", getattr(graph._cgraph, f"set_nodes_data_{name}")
            )

        if nodes is not None and not isinstance(nodes, np.ndarray):
            # nodes is not an ndarray, can it be converted into one?
            try:
                # does it have a length?
                _ = len(nodes)  # type: ignore
                # if so, convert to ndarray
                nodes = np.array(nodes, dtype=graph.node_dtype)
            except Exception:
                # must be a single node
                for name in graph.node_attr_dtypes.keys():
                    super().__setattr__(
                        f"set_attr_{name}",
                        getattr(graph._cgraph, f"set_node_data_{name}"),
                    )
                    super().__setattr__(
                        f"get_attr_{name}",
                        getattr(graph._cgraph, f"get_node_data_{name}"),
                    )

        # at this point, nodes is either
        # 1. a numpy array
        # 2. a scalar (python or numpy)
        # 3. None
        super().__setattr__("nodes", nodes)

    def __getattr__(self, name: str) -> np.ndarray:
        if name in self.graph.node_attr_dtypes:
            return getattr(self, f"get_attr_{name}")(self.nodes)
        else:
            raise AttributeError(name)

    def __setattr__(self, name, values):
        if name in self.graph.node_attr_dtypes:
            return getattr(self, f"set_attr_{name}")(self.nodes, values)
        else:
            return super().__setattr__(name, values)

    def __iter__(self):
        # TODO: shouldn't be possible if nodes is a single node
        yield from self.graph._cgraph.nodes_data(self.nodes)


class EdgeAttrsView:
    graph: GraphBase
    edges: np.ndarray | tuple[float, float] | None

    def __init__(self, graph: GraphBase, edges: np.ndarray | Iterable | None) -> None:
        super().__setattr__("graph", graph)
        for name in graph.edge_attr_dtypes:
            super().__setattr__(
                f"get_attr_{name}", getattr(graph._cgraph, f"get_edges_data_{name}")
            )
            super().__setattr__(
                f"set_attr_{name}", getattr(graph._cgraph, f"set_edges_data_{name}")
            )

        # edges types we support:
        #
        # 1. edges = None                   all edges           leave as is
        # 2. edges = iteratible of 2-tuples selected edges      to (n,2) ndarray
        # 3. edges = iteratible of 2-lists  selected edges      to (n,2) ndarray
        # 4. edges = (n,2) ndarray          selected edges      leave as is
        # 5. edges = 2-tuple                a single edge       leave as is
        # 6. edges = (2,) ndarray           a single edge       to 2-tuple

        if edges is not None:
            if isinstance(edges, np.ndarray):
                if len(edges) == 2 and len(edges.shape) == 1:
                    # case 6
                    edges = tuple(edges)
                else:
                    # case 4 with multiple edges
                    edges = edges.astype(graph.node_dtype)
            elif isinstance(edges, tuple):
                # case 5
                assert len(edges) == 2, "Single edges should be given as a 2-tuple"
            else:
                # edges should be an iteratable
                try:
                    # does it have a length?
                    len(edges)  # type: ignore
                    # case 2 and 3
                    edges = np.array(edges, dtype=graph.node_dtype)
                except Exception as e:  # pragma: no cover
                    raise RuntimeError(
                        f"Can not handle edges type {type(edges)}"
                    ) from e

        if isinstance(edges, np.ndarray):
            if len(edges) == 0:
                edges = edges.reshape((0, 2))
            assert edges.shape[1] == 2, "Edge arrays should have shape (n, 2)"  # type: ignore
            edges = np.ascontiguousarray(edges.T)  # type: ignore
        elif isinstance(edges, tuple):
            # a single edge
            for name in graph.edge_attr_dtypes.keys():
                super().__setattr__(
                    f"get_attr_{name}", getattr(graph._cgraph, f"get_edge_data_{name}")
                )
                super().__setattr__(
                    f"set_attr_{name}", getattr(graph._cgraph, f"set_edge_data_{name}")
                )

        # at this point, edges is either
        # 1. a nx2 numpy array
        # 2. a 2-tuple of scalars (python or numpy)
        # 3. None
        super().__setattr__("edges", edges)

    def __getattr__(self, name: str) -> np.ndarray:
        if name in self.graph.edge_attr_dtypes:
            if self.edges is not None:
                return getattr(self, f"get_attr_{name}")(self.edges[0], self.edges[1])
            else:
                return getattr(self, f"get_attr_{name}")(None, None)
        else:
            raise AttributeError(name)

    def __setattr__(self, name, values):
        if name in self.graph.edge_attr_dtypes:
            return getattr(self, f"set_attr_{name}")(
                self.edges[0], self.edges[1], values
            )
        else:
            return super().__setattr__(name, values)

    def __iter__(self):
        # TODO: shouldn't be possible if edges is a single edge
        yield from self.graph._cgraph.edges_data(self.edges)


class NodeAttrs(NodeAttrsView):
    def __init__(self, graph: GraphBase) -> None:
        super().__init__(graph, nodes=None)

    def __getitem__(self, nodes) -> NodeAttrsView:
        return NodeAttrsView(self.graph, nodes)


class EdgeAttrs(EdgeAttrsView):
    def __init__(self, graph: GraphBase) -> None:
        super().__init__(graph, edges=None)

    def __getitem__(self, edges) -> EdgeAttrsView:
        return EdgeAttrsView(self.graph, edges)
