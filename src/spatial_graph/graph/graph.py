from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import witty
from Cheetah.Template import Template
from typing_extensions import Self

from spatial_graph.dtypes import DType

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .cgraph import CGraph, DirectedCGraph, UnDirectedCGraph


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


@overload
def _compile_graph(
    node_dtype: str,
    node_attr_dtypes: Mapping[str, str] | None = None,
    edge_attr_dtypes: Mapping[str, str] | None = None,
    directed: Literal[True] = ...,
) -> type[DirectedCGraph]: ...
@overload
def _compile_graph(
    node_dtype: str,
    node_attr_dtypes: Mapping[str, str] | None = None,
    edge_attr_dtypes: Mapping[str, str] | None = None,
    directed: bool = ...,
) -> type[UnDirectedCGraph]: ...
def _compile_graph(
    node_dtype: str,
    node_attr_dtypes: Mapping[str, str] | None = None,
    edge_attr_dtypes: Mapping[str, str] | None = None,
    directed: bool = False,
) -> type[CGraph]:
    wrapper_template = _build_wrapper(
        node_dtype=node_dtype,
        node_attr_dtypes=node_attr_dtypes,
        edge_attr_dtypes=edge_attr_dtypes,
        directed=directed,
    )
    wrapper = witty.compile_module(
        wrapper_template,
        source_files=[str(SRC_DIR / "src" / "graph_lite.h")],
        extra_compile_args=EXTRA_COMPILE_ARGS,
        include_dirs=[str(SRC_DIR)],
        language="c++",
        quiet=True,
    )
    return wrapper.Graph


if TYPE_CHECKING:

    class Graph(CGraph):
        node_dtype: str
        node_attr_dtypes: Mapping[str, str] | None
        edge_attr_dtypes: Mapping[str, str] | None
        directed: bool
        node_attrs: NodeAttrs
        edge_attrs: EdgeAttrs

        def __init__(
            self,
            node_dtype: str,
            node_attr_dtypes: Mapping[str, str] | None = None,
            edge_attr_dtypes: Mapping[str, str] | None = None,
            directed: bool = False,
        ): ...
else:

    class Graph:
        def __new__(
            cls,
            node_dtype: str,
            node_attr_dtypes: Mapping[str, str] | None = None,
            edge_attr_dtypes: Mapping[str, str] | None = None,
            directed: bool = False,
            *args,
            **kwargs,
        ) -> Self:
            # dynamically compile a specialized C++ implementation of the graph
            # tailored to the user's specific type requirements.
            CGraph = _compile_graph(
                node_dtype=node_dtype,
                node_attr_dtypes=node_attr_dtypes,
                edge_attr_dtypes=edge_attr_dtypes,
                directed=directed,
            )
            # create a new class that inherits from both this class
            # and the compiled c++ implementation `wrapper.Graph`
            GraphType = type(cls.__name__, (cls, CGraph), {})
            # call the __new__ method of the native C++ class, but pass the dynamically
            # created class as the type.  This ensures the object will be an instance
            # of the dynamically created class, but using the C++ allocation logic and
            # initialization code.
            return CGraph.__new__(GraphType)

        def __init__(
            self,
            node_dtype: str,
            node_attr_dtypes: Mapping[str, str] | None = None,
            edge_attr_dtypes: Mapping[str, str] | None = None,
            directed: bool = False,
        ):
            if node_attr_dtypes is None:
                node_attr_dtypes = {}
            if edge_attr_dtypes is None:
                edge_attr_dtypes = {}
            super().__init__()
            self.node_dtype = node_dtype
            self.node_attr_dtypes = node_attr_dtypes
            self.edge_attr_dtypes = edge_attr_dtypes
            self.directed = directed

            self.node_attrs = NodeAttrs(self)
            self.edge_attrs = EdgeAttrs(self)


class NodeAttrsView:
    def __init__(self, graph, nodes):
        super().__setattr__("graph", graph)
        for name in graph.node_attr_dtypes.keys():
            super().__setattr__(
                f"get_attr_{name}", getattr(graph, f"get_nodes_data_{name}")
            )
            super().__setattr__(
                f"set_attr_{name}", getattr(graph, f"set_nodes_data_{name}")
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
                    super().__setattr__(
                        f"set_attr_{name}", getattr(graph, f"set_node_data_{name}")
                    )
                    super().__setattr__(
                        f"get_attr_{name}", getattr(graph, f"get_node_data_{name}")
                    )

        # at this point, nodes is either
        # 1. a numpy array
        # 2. a scalar (python or numpy)
        # 3. None
        super().__setattr__("nodes", nodes)

    def __getattr__(self, name):
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
        yield from self.graph.nodes_data(self.nodes)


class EdgeAttrsView:
    def __init__(self, graph, edges):
        super().__setattr__("graph", graph)
        for name in graph.edge_attr_dtypes.keys():
            super().__setattr__(
                f"get_attr_{name}", getattr(graph, f"get_edges_data_{name}")
            )
            super().__setattr__(
                f"set_attr_{name}", getattr(graph, f"set_edges_data_{name}")
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
                except Exception as e:  # pragma: no cover
                    raise RuntimeError(
                        f"Can not handle edges type {type(edges)}"
                    ) from e

        if isinstance(edges, np.ndarray):
            if len(edges) == 0:
                edges = edges.reshape((0, 2))
            assert edges.shape[1] == 2, "Edge arrays should have shape (n, 2)"
            edges = np.ascontiguousarray(edges.T)
        elif isinstance(edges, tuple):
            # a single edge
            for name in graph.edge_attr_dtypes.keys():
                super().__setattr__(
                    f"get_attr_{name}", getattr(graph, f"get_edge_data_{name}")
                )
                super().__setattr__(
                    f"set_attr_{name}", getattr(graph, f"set_edge_data_{name}")
                )

        # at this point, edges is either
        # 1. a nx2 numpy array
        # 2. a 2-tuple of scalars (python or numpy)
        # 3. None
        super().__setattr__("edges", edges)

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
            return super().__setattr__(name, values)

    def __iter__(self):
        # TODO: shouldn't be possible if edges is a single edge
        yield from self.graph.edges_data(self.edges)


class NodeAttrs(NodeAttrsView):
    def __init__(self, graph):
        super().__init__(graph, nodes=None)

    def __getitem__(self, nodes):
        return NodeAttrsView(self.graph, nodes)


class EdgeAttrs(EdgeAttrsView):
    def __init__(self, graph):
        super().__init__(graph, edges=None)

    def __getitem__(self, edges):
        return EdgeAttrsView(self.graph, edges)
