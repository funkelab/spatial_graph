import witty
from pathlib import Path
import numpy as np
from .dtypes import (
    DType,
    dtypes_to_struct,
    dtypes_to_arguments,
    dtypes_to_array_pointers,
    dtypes_to_array_pointer_names,
)


node_data_template = """
    def node_data_NAME(self, NodeType node):
        return self._graph.node_prop(node).NAME
"""

nodes_data_template = """
    def nodes_data_NAME(self, NodeType[::1] nodes):

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
        graphlite_pyx = open(src_dir / "graphlite_wrapper.pyx").read()
        graphlite_pyx = graphlite_pyx.replace(
            "NODE_TYPE_DECLARATION", f"ctypedef {node_dtype.to_pyxtype()} NodeType"
        )
        graphlite_pyx = graphlite_pyx.replace(
            "NODE_DATA_DECLARATION", dtypes_to_struct("NodeData", node_attr_dtypes)
        )
        graphlite_pyx = graphlite_pyx.replace(
            "EDGE_DATA_DECLARATION", dtypes_to_struct("EdgeData", edge_attr_dtypes)
        )
        graphlite_pyx = graphlite_pyx.replace(
            "NODE_DATA_ARGS", dtypes_to_arguments(node_attr_dtypes)
        )
        graphlite_pyx = graphlite_pyx.replace(
            "EDGE_DATA_ARGS", dtypes_to_arguments(edge_attr_dtypes)
        )
        graphlite_pyx = graphlite_pyx.replace(
            "NODE_DATA_ARRAY_ARGS",
            dtypes_to_arguments(node_attr_dtypes, as_arrays=True),
        )
        graphlite_pyx = graphlite_pyx.replace(
            "EDGE_DATA_ARRAY_ARGS",
            dtypes_to_arguments(edge_attr_dtypes, as_arrays=True),
        )
        graphlite_pyx = graphlite_pyx.replace(
            "NODE_DATA_ARRAY_POINTERS_SET",
            dtypes_to_array_pointers(node_attr_dtypes, indent=2),
        )
        graphlite_pyx = graphlite_pyx.replace(
            "NODE_DATA_ARRAY_POINTERS_NAMES",
            dtypes_to_array_pointer_names(node_attr_dtypes),
        )
        graphlite_pyx = graphlite_pyx.replace(
            "NODE_DATA_ARRAYS_POINTERS_DEF",
            dtypes_to_array_pointers(node_attr_dtypes, indent=2, definition_only=True),
        )
        graphlite_pyx = graphlite_pyx.replace(
            "NODE_DATA_ARRAYS_POINTERS_SET",
            dtypes_to_array_pointers(
                node_attr_dtypes, indent=3, assignment_only=True, array_index="i"
            ),
        )
        graphlite_pyx = graphlite_pyx.replace(
            "NODE_DATA_ARRAYS_POINTERS_NAMES",
            dtypes_to_array_pointer_names(node_attr_dtypes, array_index="i"),
        )
        graphlite_pyx = graphlite_pyx.replace(
            "EDGE_DATA_ARRAY_POINTERS_SET",
            dtypes_to_array_pointers(edge_attr_dtypes, indent=2),
        )
        graphlite_pyx = graphlite_pyx.replace(
            "EDGE_DATA_ARRAY_POINTERS_NAMES",
            dtypes_to_array_pointer_names(edge_attr_dtypes),
        )
        graphlite_pyx = graphlite_pyx.replace(
            "EDGE_DATA_ARRAYS_POINTERS_DEF",
            dtypes_to_array_pointers(edge_attr_dtypes, indent=2, definition_only=True),
        )
        graphlite_pyx = graphlite_pyx.replace(
            "EDGE_DATA_ARRAYS_POINTERS_SET",
            dtypes_to_array_pointers(
                edge_attr_dtypes, indent=3, assignment_only=True, array_index="i"
            ),
        )
        graphlite_pyx = graphlite_pyx.replace(
            "EDGE_DATA_ARRAYS_POINTERS_NAMES",
            dtypes_to_array_pointer_names(edge_attr_dtypes, array_index="i"),
        )

        node_data_by_name = "\n\n".join(
            [
                node_data_template.replace("NAME", name)
                for name in node_attr_dtypes.keys()
            ]
        )
        graphlite_pyx = graphlite_pyx.replace("NODE_DATA_BY_NAME", node_data_by_name)

        nodes_data_by_name = "\n\n".join(
            [
                nodes_data_template.replace("NAME", name)
                .replace("PYXTYPE", dtype.to_pyxtype(as_arrays=True))
                .replace("NPTYPE", dtype.base)
                .replace("SHAPE", str(dtype.shape))
                for name, dtype in node_attr_dtypes.items()
            ]
        )
        graphlite_pyx = graphlite_pyx.replace("NODES_DATA_BY_NAME", nodes_data_by_name)

        graphlite = witty.compile_module(
            graphlite_pyx,
            extra_compile_args=["-O3", "-std=c++17"],
            include_dirs=[str(src_dir)],
            language="c++",
            quiet=True,
        )
        Graph = graphlite.DirectedGraph if directed else graphlite.UndirectedGraph
        GraphType = type(cls.__name__, (cls, Graph), {})
        return Graph.__new__(GraphType)

    def __init__(self, node_dtype, node_attr_dtypes, edge_attr_dtypes, directed=False):
        super().__init__()
        self.node_dtype = node_dtype
        self.node_attr_dtypes = node_attr_dtypes
        self.edge_attr_dtypes = edge_attr_dtypes
        self.directed = directed

        self.node_attrs = NodeAttrs(self)


class NodeAttrsView:
    def __init__(self, graph, nodes):
        super().__setattr__("graph", graph)
        for name in graph.node_attr_dtypes.keys():
            super().__setattr__(f"attr_{name}", getattr(graph, f"nodes_data_{name}"))

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
                        f"attr_{name}", getattr(graph, f"node_data_{name}")
                    )

        # at this point, nodes is either
        # 1. a numpy array
        # 2. a scalar (python or numpy)
        # 3. None
        super().__setattr__("nodes", nodes)

    def __getattr__(self, name):
        if name in self.graph.node_attr_dtypes:
            return getattr(self, f"attr_{name}")(self.nodes)
        else:
            raise AttributeError(name)

    def __setattr__(self, name, values):
        if name in self.graph.node_attr_dtypes:
            raise RuntimeError("not yet implemented")
        else:
            return super().__setattr__(name, values)

    def __iter__(self):
        # TODO: shouldn't be possible if nodes is a single node
        yield from self.graph.nodes_data(self.nodes)


class NodeAttrs(NodeAttrsView):
    def __init__(self, graph):
        super().__init__(graph, nodes=None)

    def __getitem__(self, nodes):
        return NodeAttrsView(self.graph, nodes)
