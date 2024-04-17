import witty
from pathlib import Path
import numpy as np


def is_array(dtype):
    if "[" in dtype:
        if "]" not in dtype:
            raise RuntimeError(f"invalid array(?) dtype {dtype}")
        return True
    return False


def parse_array_dtype(dtype):
    dtype, size = dtype.split("[")
    size = int(size.split("]")[0])

    return dtype, size


def length_from_dtype(dtype):
    if is_array(dtype):
        return str(parse_array_dtype(dtype)[1])
    else:
        return ""


def strip_array(dtype):
    if is_array(dtype):
        return parse_array_dtype(dtype)[0]
    else:
        return dtype


def dtype_to_pyxtype(dtype, use_memory_view=False, as_arrays=False):
    """Convert from numpy-like dtype strings to their equivalent C/C++/PYX types.

    Args:

        use_memory_view:

            If set, will produce "[::1]" instead of "[dim]" for array types.

        as_arrays:

            Create arrays of the types, e.g., "int32_t[::1]" instead of
            "int32_t" for dtype "int32".
    """

    # is this an array type?
    if is_array(dtype):
        dtype, size = parse_array_dtype(dtype)
        if as_arrays:
            suffix = "[:, ::1]"
        else:
            if use_memory_view:
                suffix = "[::1]"
            else:
                suffix = f"[{size}]"
    else:
        suffix = "" if not as_arrays else "[::1]"

    if dtype == "float32" or dtype == "float":
        dtype = "float"
    elif dtype == "float64" or dtype == "double":
        dtype = "double"
    else:
        # this might not work for all of them, this is just a fallback
        dtype = np.dtype(dtype).name + "_t"

    return dtype + suffix


def dtypes_to_struct(struct_name, dtypes):
    pyx_code = f"cdef struct {struct_name}:\n"
    for name, dtype in dtypes.items():
        pyx_code += f"    {dtype_to_pyxtype(dtype)} {name}\n"
    return pyx_code


def dtypes_to_arguments(dtypes, as_arrays=False):
    return ", ".join(
        [
            f"{dtype_to_pyxtype(dtype, use_memory_view=True, as_arrays=as_arrays)} "
            f"{name}"
            for name, dtype in dtypes.items()
        ]
    )


def dtypes_to_array_pointers(
    dtypes, indent, definition_only=False, assignment_only=False, array_index=None
):
    pyx_code = ""

    for name, dtype in dtypes.items():
        if is_array(dtype):
            dtype, size = parse_array_dtype(dtype)
            pyx_code = "    " * indent
            if not assignment_only:
                pyx_code += f"cdef {dtype_to_pyxtype(dtype)}[{size}] "
            pyx_code += f"_p_{name}"
            if not definition_only:
                if array_index:
                    pyx_code += f" = &{name}[{array_index}, 0]\n"
                else:
                    pyx_code += f" = &{name}[0]\n"
            else:
                pyx_code += "\n"

    return pyx_code


def dtypes_to_array_pointer_names(dtypes, array_index=None):
    return ", ".join(
        [
            f"_p_{name}"
            if is_array(dtype)
            else (f"{name}[{array_index}]" if array_index else name)
            for name, dtype in dtypes.items()
        ]
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
        data = np.empty(shape=(num_nodes, LENGTH), dtype="DTYPE")
        cdef PYXDTYPE view = data

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
        edge_attr_dtypes,
        directed=False,
        *args,
        **kwargs,
    ):
        src_dir = Path(__file__).parent
        graphlite_pyx = open(src_dir / "graphlite_wrapper.pyx").read()
        graphlite_pyx = graphlite_pyx.replace(
            "NODE_TYPE_DECLARATION", f"ctypedef {dtype_to_pyxtype(node_dtype)} NodeType"
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
                .replace("PYXDTYPE", dtype_to_pyxtype(dtype, as_arrays=True))
                .replace("DTYPE", strip_array(dtype))
                .replace("LENGTH", length_from_dtype(dtype))
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
