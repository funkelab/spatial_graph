from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Iterator, Self, TypeVar

import numpy as np
import witty
from Cheetah.Template import Template

from ..dtypes import DType

if TYPE_CHECKING:
    import numbers
    from collections.abc import Mapping
    from typing_extensions import Self

# Set platform-specific compile arguments
if sys.platform == "win32":
    # Use /O2 for optimization and /std:c++20 for C++20
    EXTRA_COMPILE_ARGS = ["/O2", "/std:c++20"]
else:
    # -O3 for optimization and -std=c++20 for C++20
    EXTRA_COMPILE_ARGS = ["-O3", "-std=c++20"]

NT = TypeVar("NT")


class Graph(Generic[NT]):
    if TYPE_CHECKING:

        def add_edge(self, edge: np.ndarray, *args: Any, **kwargs: Any) -> int:
            """Add an edge to the graph.

            Names/number of args kwargs must match the edge_attr_dtypes.
            """

        def add_edges(
            self, edges: np.ndarray, *args: np.ndarray, **kwargs: np.ndarray
        ) -> int:
            """Add multiple edges to the graph.

            edges must be 2D and names/number of args kwargs must match the
            edge_attr_dtypes.
            """

        def add_node(self, node: NT, *data: Any) -> int: ...
        def add_nodes(self, nodes: np.ndarray, *data: Any) -> int: ...
        def edges(self, node: Any = None, data: bool = False) -> Iterator: ...
        def edges_by_nodes(self, nodes: np.ndarray) -> np.ndarray: ...
        def edges_data(self, us: np.ndarray, vs: np.ndarray) -> Iterator: ...
        def nodes(self) -> np.ndarray: ...
        def nodes_data(
            self, nodes: np.ndarray | None = None
        ) -> Iterator[tuple[Any, Any]]: ...
        def num_edges(self) -> int: ...
        def remove_node(self, node: Any) -> None: ...
        def remove_nodes(self, nodes: np.ndarray) -> None: ...

        # only for undirected
        def count_neighbors(self, nodes: np.ndarray) -> int: ...
        # only for directed
        def count_in_neighbors(self, nodes: np.ndarray) -> int: ...
        def count_out_neighbors(self, nodes: np.ndarray) -> int: ...

    def __new__(
        cls,
        node_dtype: str,
        node_attr_dtypes: Mapping[str, str] | None = None,
        edge_attr_dtypes: Mapping[str, str] | None = None,
        directed: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        src_dir = Path(__file__).parent
        wrapper_template = Template(
            file=str(src_dir / "wrapper_template.pyx"),
            compilerSettings={"directiveStartToken": "%"},
        )
        wrapper_template.node_dtype = DType(node_dtype)
        wrapper_template.node_attr_dtypes = {
            name: DType(dtype) for name, dtype in (node_attr_dtypes or {}).items()
        }
        wrapper_template.edge_attr_dtypes = {
            name: DType(dtype) for name, dtype in (edge_attr_dtypes or {}).items()
        }
        wrapper_template.directed = directed

        # dynamically compile a specialized C++ implementation of the graph
        # tailored to the user's specific type requirements.
        wrapper = witty.compile_module(
            str(wrapper_template),
            source_files=[str(src_dir / "src" / "graph_lite.h")],
            extra_compile_args=EXTRA_COMPILE_ARGS,
            include_dirs=[str(src_dir)],
            language="c++",
            quiet=True,
        )

        # create a new class that inherits from both this class (Graph or a subclass)
        # and the compiled c++ implementation `wrapper.Graph`
        GraphType = type(cls.__name__, (cls, wrapper.Graph), {})

        # call the __new__ method of the native C++ class, but pass the dynamically
        # created class as the type.  This ensures the object will be an instance
        # of the dynamically created class, but using the C++ allocation logic and
        # initialization code.
        return wrapper.Graph.__new__(GraphType)

    def __init__(
        self,
        node_dtype: str,
        node_attr_dtypes: Mapping[str, str] | None = None,
        edge_attr_dtypes: Mapping[str, str] | None = None,
        directed: bool = False,
    ) -> None:
        super().__init__()
        self.node_dtype = node_dtype
        self.node_attr_dtypes = node_attr_dtypes or {}
        self.edge_attr_dtypes = edge_attr_dtypes or {}
        self.directed = directed

        self.node_attrs = NodeAttrs(self)
        self.edge_attrs = EdgeAttrs(self)


class NodeAttrsView:
    graph: Graph
    nodes: np.ndarray | numbers.Number | None

    def __init__(self, graph: Graph, nodes: Any = None) -> None:
        object.__setattr__(self, "graph", graph)
        for name in graph.node_attr_dtypes.keys():
            object.__setattr__(
                self, f"get_attr_{name}", getattr(graph, f"get_nodes_data_{name}")
            )
            object.__setattr__(
                self, f"set_attr_{name}", getattr(graph, f"set_nodes_data_{name}")
            )

        if nodes is not None and not isinstance(nodes, np.ndarray):
            # nodes is not an ndarray, can it be converted into one?
            try:
                # does it have a length?
                _ = len(nodes)
                # if so, convert to ndarray
                nodes = np.array(nodes, dtype=graph.node_dtype)
            except Exception:
                # must be a single node
                for name in graph.node_attr_dtypes.keys():
                    object.__setattr__(
                        self,
                        f"set_attr_{name}",
                        getattr(graph, f"set_node_data_{name}"),
                    )
                    object.__setattr__(
                        self,
                        f"get_attr_{name}",
                        getattr(graph, f"get_node_data_{name}"),
                    )

        # at this point, nodes is either
        # 1. a numpy array
        # 2. a scalar (python or numpy)
        # 3. None
        object.__setattr__(self, "nodes", nodes)

    def __getattr__(self, name):
        if name in self.graph.node_attr_dtypes:
            return getattr(self, f"get_attr_{name}")(self.nodes)
        else:
            raise AttributeError(name)

    def __setattr__(self, name, values):
        if name in self.graph.node_attr_dtypes:
            return getattr(self, f"set_attr_{name}")(self.nodes, values)
        else:
            return object.__setattr__(self, name, values)

    def __iter__(self):
        # TODO: shouldn't be possible if nodes is a single node
        yield from self.graph.nodes_data(self.nodes)


class EdgeAttrsView:
    graph: Graph
    edges: np.ndarray | tuple[numbers.Number, numbers.Number] | None

    def __init__(self, graph: Graph, edges: Any = None) -> None:
        object.__setattr__(self, "graph", graph)
        for name in graph.edge_attr_dtypes.keys():
            object.__setattr__(
                self, f"get_attr_{name}", getattr(graph, f"get_edges_data_{name}")
            )
            object.__setattr__(
                self, f"set_attr_{name}", getattr(graph, f"set_edges_data_{name}")
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
                    len(edges)
                    # case 2 and 3
                    edges = np.array(edges, dtype=graph.node_dtype)
                except Exception:
                    raise RuntimeError(f"Can not handle edges type {type(edges)}")

        if isinstance(edges, np.ndarray):
            if len(edges) == 0:
                edges = edges.reshape((0, 2))
            assert edges.shape[1] == 2, "Edge arrays should have shape (n, 2)"
            edges = np.ascontiguousarray(edges.T)
        elif isinstance(edges, tuple):
            # a single edge
            for name in graph.edge_attr_dtypes.keys():
                object.__setattr__(
                    self, f"get_attr_{name}", getattr(graph, f"get_edge_data_{name}")
                )
                object.__setattr__(
                    self, f"set_attr_{name}", getattr(graph, f"set_edge_data_{name}")
                )

        # at this point, edges is either
        # 1. a nx2 numpy array
        # 2. a 2-tuple of scalars (python or numpy)
        # 3. None
        object.__setattr__(self, "edges", edges)

    def __getattr__(self, name):
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
            return object.__setattr__(self, name, values)

    def __iter__(self):
        # TODO: shouldn't be possible if edges is a single edge
        yield from self.graph.edges_data(self.edges)


class NodeAttrs(NodeAttrsView):
    def __getitem__(self, nodes: Any) -> NodeAttrsView:
        return NodeAttrsView(self.graph, nodes)


class EdgeAttrs(EdgeAttrsView):
    def __getitem__(self, edges: Any) -> EdgeAttrsView:
        return EdgeAttrsView(self.graph, edges)
