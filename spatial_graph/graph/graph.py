import witty
from pathlib import Path
import numpy as np
from ..dtypes import (
    DType,
    dtypes_to_cppclass,
    dtypes_to_arguments,
    dtypes_to_array_pointers,
    dtypes_to_array_pointer_names,
)


node_data_template = """
    def get_node_data_NAME(self, NodeType node):
        return self._graph.node_prop(node).NAME

    def set_node_data_NAME(self, NodeType node, DTYPE NAME):
        self._graph.node_prop(node).NAME = RVALUE
"""

nodes_data_template = """
    def get_nodes_data_NAME(self, NodeType[::1] nodes):

        cdef NodeIterator node_it = self._graph.begin()
        cdef NodeIterator node_end = self._graph.end()
        cdef Py_ssize_t i = 0

        # allocate array for data
        cdef Py_ssize_t num_nodes = 0
        if nodes is None:
            num_nodes = self._graph.size()
        else:
            num_nodes = len(nodes)
        data = np.empty(shape=(num_nodes,) + SHAPE, dtype="NPTYPE")
        cdef PYXTYPE view = data

        # all nodes requested
        if nodes is None:
            while node_it != node_end:
                view[i] = self._graph.node_prop(node_it).NAME
                inc(node_it)
                i += 1
        else:
            for i in range(num_nodes):
                view[i] = self._graph.node_prop(nodes[i]).NAME

        return data

    def set_nodes_data_NAME(self, NodeType[::1] nodes, PYXTYPE NAME):

        cdef NodeIterator node_it = self._graph.begin()
        cdef NodeIterator node_end = self._graph.end()
        cdef Py_ssize_t i = 0

        # all nodes requested
        if nodes is None:
            while node_it != node_end:
                self._graph.node_prop(node_it).NAME = RVALUE
                inc(node_it)
                i += 1
        else:
            assert len(nodes) == len(NAME)
            for i in range(len(nodes)):
                self._graph.node_prop(nodes[i]).NAME = RVALUE
"""

edge_data_template = """
    def get_edge_data_NAME(self, NodeType u, NodeType v):
        return self._graph.edge_prop(u, v).NAME

    def set_edge_data_NAME(self, NodeType u, NodeType v, DTYPE NAME):
        self._graph.edge_prop(u, v).NAME = RVALUE
"""

edges_data_template = """
    def get_edges_data_NAME(self, NodeType[::1] us, NodeType[::1] vs):

        cdef Py_ssize_t i = 0
        cdef Py_ssize_t num_edges = 0
        cdef NodeIterator node_it = self._graph.begin()
        cdef NodeIterator node_end = self._graph.end()
        cdef pair[NeighborsIterator, NeighborsIterator] edges_view
        cdef NodeType u, v

        # allocate array for data
        if us is None and vs is None:
            num_edges = self._graph.num_edges()
        elif us is None or vs is None:
            raise RuntimeError("Either both us and vs are None, or neither")
        else:
            num_edges = len(us)
        data = np.empty(shape=(num_edges,) + SHAPE, dtype="NPTYPE")
        cdef PYXTYPE view = data

        if us is None:

            # iterate over all edges by iterating over all nodes u and their
            # neighbors v with u < v

            while node_it != node_end:
                edges_view = self._graph.neighbors(node_it)
                u = deref(node_it)
                it = edges_view.first
                end = edges_view.second
                while it != end:
                    v = deref(it).first
                    if u < v:
                        view[i] = deref(it).second.prop().NAME
                        i += 1
                    inc(it)
                inc(node_it)

        else:
            for i in range(num_edges):
                view[i] = self._graph.edge_prop(us[i], vs[i]).NAME

        return data

    def set_edges_data_NAME(self, NodeType[::1] us, NodeType[::1] vs, PYXTYPE NAME):

        cdef Py_ssize_t i = 0
        cdef Py_ssize_t num_edges = 0

        assert len(us) == len(vs)
        num_edges = len(us)

        for i in range(num_edges):
            self._graph.edge_prop(us[i], vs[i]).NAME = RVALUE
"""


class Graph:
    def __new__(
        cls,
        node_dtype,
        node_attr_dtypes,
        edge_attr_dtypes=None,
        directed=False,
        *args,
        **kwargs,
    ):
        if edge_attr_dtypes is None:
            edge_attr_dtypes = {}

        node_dtype = DType(node_dtype)
        node_attr_dtypes = {
            name: DType(dtype) for name, dtype in node_attr_dtypes.items()
        }
        edge_attr_dtypes = {
            name: DType(dtype) for name, dtype in edge_attr_dtypes.items()
        }

        src_dir = Path(__file__).parent
        wrapper_pyx = open(src_dir / "src_wrapper.pyx").read()
        wrapper_pyx = wrapper_pyx.replace(
            "NODE_TYPE_DECLARATION", f"ctypedef {node_dtype.to_pyxtype()} NodeType"
        )
        wrapper_pyx = wrapper_pyx.replace("NODE_NPTYPE", node_dtype.base)
        wrapper_pyx = wrapper_pyx.replace(
            "NODE_DATA_DECLARATION", dtypes_to_cppclass("NodeData", node_attr_dtypes)
        )
        wrapper_pyx = wrapper_pyx.replace(
            "EDGE_DATA_DECLARATION", dtypes_to_cppclass("EdgeData", edge_attr_dtypes)
        )
        wrapper_pyx = wrapper_pyx.replace(
            "NODE_DATA_ARGS", dtypes_to_arguments(node_attr_dtypes)
        )
        wrapper_pyx = wrapper_pyx.replace(
            "EDGE_DATA_ARGS", dtypes_to_arguments(edge_attr_dtypes)
        )
        wrapper_pyx = wrapper_pyx.replace(
            "NODE_DATA_ARRAY_ARGS",
            dtypes_to_arguments(node_attr_dtypes, add_dim=True),
        )
        wrapper_pyx = wrapper_pyx.replace(
            "EDGE_DATA_ARRAY_ARGS",
            dtypes_to_arguments(edge_attr_dtypes, add_dim=True),
        )
        wrapper_pyx = wrapper_pyx.replace(
            "NODE_DATA_ARRAY_POINTERS_SET",
            dtypes_to_array_pointers(node_attr_dtypes, indent=2),
        )
        wrapper_pyx = wrapper_pyx.replace(
            "NODE_DATA_ARRAY_POINTERS_NAMES",
            dtypes_to_array_pointer_names(node_attr_dtypes),
        )
        wrapper_pyx = wrapper_pyx.replace(
            "NODE_DATA_ARRAYS_POINTERS_DEF",
            dtypes_to_array_pointers(node_attr_dtypes, indent=2, definition_only=True),
        )
        wrapper_pyx = wrapper_pyx.replace(
            "NODE_DATA_ARRAYS_POINTERS_SET",
            dtypes_to_array_pointers(
                node_attr_dtypes, indent=3, assignment_only=True, array_index="i"
            ),
        )
        wrapper_pyx = wrapper_pyx.replace(
            "NODE_DATA_ARRAYS_POINTERS_NAMES",
            dtypes_to_array_pointer_names(node_attr_dtypes, array_index="i"),
        )
        wrapper_pyx = wrapper_pyx.replace(
            "EDGE_DATA_ARRAY_POINTERS_SET",
            dtypes_to_array_pointers(edge_attr_dtypes, indent=2),
        )
        wrapper_pyx = wrapper_pyx.replace(
            "EDGE_DATA_ARRAY_POINTERS_NAMES",
            dtypes_to_array_pointer_names(edge_attr_dtypes),
        )
        wrapper_pyx = wrapper_pyx.replace(
            "EDGE_DATA_ARRAYS_POINTERS_DEF",
            dtypes_to_array_pointers(edge_attr_dtypes, indent=2, definition_only=True),
        )
        wrapper_pyx = wrapper_pyx.replace(
            "EDGE_DATA_ARRAYS_POINTERS_SET",
            dtypes_to_array_pointers(
                edge_attr_dtypes, indent=3, assignment_only=True, array_index="i"
            ),
        )
        wrapper_pyx = wrapper_pyx.replace(
            "EDGE_DATA_ARRAYS_POINTERS_NAMES",
            dtypes_to_array_pointer_names(edge_attr_dtypes, array_index="i"),
        )

        node_data_by_name = "\n\n".join(
            [
                node_data_template.replace("NAME", name)
                .replace("DTYPE", dtype.to_pyxtype(use_memory_view=True))
                .replace("RVALUE", dtype.to_rvalue(name=name))
                for name, dtype in node_attr_dtypes.items()
            ]
        )
        wrapper_pyx = wrapper_pyx.replace("NODE_DATA_BY_NAME", node_data_by_name)

        nodes_data_by_name = "\n\n".join(
            [
                nodes_data_template.replace("NAME", name)
                .replace("PYXTYPE", dtype.to_pyxtype(add_dim=True))
                .replace("NPTYPE", dtype.base)
                .replace("RVALUE", dtype.to_rvalue(name=name, array_index="i"))
                .replace("SHAPE", str(dtype.shape))
                for name, dtype in node_attr_dtypes.items()
            ]
        )
        wrapper_pyx = wrapper_pyx.replace("NODES_DATA_BY_NAME", nodes_data_by_name)

        edge_data_by_name = "\n\n".join(
            [
                edge_data_template.replace("NAME", name)
                .replace("DTYPE", dtype.to_pyxtype(use_memory_view=True))
                .replace("RVALUE", dtype.to_rvalue(name=name))
                for name, dtype in edge_attr_dtypes.items()
            ]
        )
        wrapper_pyx = wrapper_pyx.replace("EDGE_DATA_BY_NAME", edge_data_by_name)

        edges_data_by_name = "\n\n".join(
            [
                edges_data_template.replace("NAME", name)
                .replace("PYXTYPE", dtype.to_pyxtype(add_dim=True))
                .replace("NPTYPE", dtype.base)
                .replace("RVALUE", dtype.to_rvalue(name=name, array_index="i"))
                .replace("SHAPE", str(dtype.shape))
                for name, dtype in edge_attr_dtypes.items()
            ]
        )
        wrapper_pyx = wrapper_pyx.replace("EDGES_DATA_BY_NAME", edges_data_by_name)

        wrapper = witty.compile_module(
            wrapper_pyx,
            extra_compile_args=["-O3", "-std=c++17"],
            include_dirs=[str(src_dir)],
            language="c++",
            quiet=True,
        )
        Graph = wrapper.DirectedGraph if directed else wrapper.UndirectedGraph
        GraphType = type(cls.__name__, (cls, Graph), {})
        return Graph.__new__(GraphType)

    def __init__(self, node_dtype, node_attr_dtypes, edge_attr_dtypes, directed=False):
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
            except Exception as e:
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
                    num_edges = len(edges)
                    # case 2 and 3
                    edges = np.array(edges, dtype=graph.node_dtype)
                except Exception as e:
                    raise RuntimeError(f"Can not handle edges type {type(edges)}")

        if isinstance(edges, np.ndarray):
            assert edges.shape[1] == 2, "Edge arrays should have shape (n, 2)"
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
