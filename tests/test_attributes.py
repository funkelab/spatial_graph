import spatial_graph as sg
import numpy as np
import pytest


def test_node_access():
    graph = sg.SpatialGraph(
        ndims=3,
        node_dtype="uint64",
        node_attr_dtypes={"position": "double[3]"},
        edge_attr_dtypes={"score": "double"},
        position_attr="position",
    )
    graph.add_node(1, position=np.array([1.0, 1.0, 1.0]))
    graph.add_node(2, position=np.array([2.0, 2.0, 2.0]))
    graph.add_node(3, position=np.array([3.0, 3.0, 3.0]))
    graph.add_node(4, position=np.array([4.0, 4.0, 4.0]))

    nodes = np.array([1, 2, 3], dtype=np.uint64)

    # attribute of all nodes
    graph.node_attrs.position
    # attribute of nodes as ndarray
    graph.node_attrs[nodes].position
    # attribute of nodes as list
    graph.node_attrs[[1, 2, 3]].position
    # attribute of nodes as tuple
    graph.node_attrs[(1, 2, 3)].position
    # attribute of single node as numpy scalar
    graph.node_attrs[nodes[0]].position
    # attribute of single node as python scalar
    graph.node_attrs[1].position


dtypes = ["float", "double", "int8", "uint8", "int16", "uint16"]


@pytest.mark.parametrize("dtype", dtypes)
def test_dtype(dtype):
    graph = sg.SpatialGraph(
        ndims=3,
        node_dtype="uint64",
        node_attr_dtypes={f"node_attr_{dtype}": dtype, "position": "double[3]"},
        edge_attr_dtypes={f"edge_attr_{dtype}": dtype},
        position_attr="position",
    )

    graph.add_node(1, position=np.array([0.0, 0.0, 0.0]), **{f"node_attr_{dtype}": 0})
    graph.add_node(2, position=np.array([0.0, 0.0, 0.0]), **{f"node_attr_{dtype}": 1})
    graph.add_edge(1, 2, **{f"edge_attr_{dtype}": 0})
