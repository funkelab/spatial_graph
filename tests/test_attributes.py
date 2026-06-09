import numpy as np
import pytest

import spatial_graph as sg


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


@pytest.mark.parametrize("cls", [sg.SpatialGraph, sg.SpatialDiGraph])
def test_edge_access(cls):
    graph = cls(
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
    graph.add_edge([1, 2], score=0.5)
    graph.add_edge([2, 3], score=0.4)
    graph.add_edge([4, 2], score=0.3)
    graph.add_edge([4, 3], score=0.2)

    edges = np.array([[1, 2], [2, 3]], dtype=np.uint64)

    # attribute of all edges
    np.testing.assert_equal(
        np.sort(graph.edge_attrs.score), np.array([0.2, 0.3, 0.4, 0.5], dtype="double")
    )
    # attribute of edges as ndarray
    np.testing.assert_equal(
        graph.edge_attrs[edges].score, np.array([0.5, 0.4], dtype="double")
    )
    # attribute of edges as list
    np.testing.assert_equal(
        graph.edge_attrs[[[1, 2], [2, 3]]].score, np.array([0.5, 0.4], dtype="double")
    )
    # attribute of edges as tuple
    np.testing.assert_equal(
        graph.edge_attrs[[(1, 2), (2, 3)]].score, np.array([0.5, 0.4], dtype="double")
    )
    # attribute of single edge as numpy array
    np.testing.assert_equal(
        graph.edge_attrs[edges[0]].score, np.array([0.5], dtype="double")
    )
    # attribute of single edge as python tuple
    np.testing.assert_equal(
        graph.edge_attrs[(1, 2)].score, np.array([0.5], dtype="double")
    )


dtypes = ["float", "double", "int8", "uint8", "int16", "uint16"]


@pytest.mark.parametrize("dtype", dtypes)
def test_attr_dtypes(dtype):
    graph = sg.SpatialGraph(
        ndims=3,
        node_dtype="uint64",
        node_attr_dtypes={f"node_attr_{dtype}": dtype, "position": "double[3]"},
        edge_attr_dtypes={f"edge_attr_{dtype}": dtype},
        position_attr="position",
    )

    graph.add_node(1, position=np.array([0.0, 0.0, 0.0]), **{f"node_attr_{dtype}": 0})
    graph.add_node(2, position=np.array([0.0, 0.0, 0.0]), **{f"node_attr_{dtype}": 1})
    graph.add_edge([1, 2], **{f"edge_attr_{dtype}": 0})


# Attribute names that collide with C++ keywords or MSVC built-in type
# specifiers (but are not Python keywords, so they are valid attribute names).
# These cannot appear verbatim as C++ struct members. In particular MSVC (with
# Microsoft extensions) treats ``int8``/``int16``/``int32``/``int64`` as aliases
# for the ``__intN`` keywords, so a member named ``int16`` expands to
# ``int16_t __int16;`` and fails to compile.
@pytest.mark.parametrize("attr_name", ["int8", "int16", "int32", "int64", "new", "double"])
def test_attr_name_collides_with_cpp_keyword(attr_name):
    graph = sg.SpatialGraph(
        ndims=3,
        node_dtype="uint64",
        node_attr_dtypes={attr_name: "int16", "position": "double[3]"},
        edge_attr_dtypes={attr_name: "int16"},
        position_attr="position",
    )

    graph.add_node(1, position=np.array([0.0, 0.0, 0.0]), **{attr_name: 5})
    graph.add_node(2, position=np.array([1.0, 1.0, 1.0]), **{attr_name: 7})
    graph.add_edge([1, 2], **{attr_name: 9})

    # round-trip through the Python-facing API, which keeps the user's name
    assert getattr(graph.node_attrs[1], attr_name) == 5
    assert getattr(graph.node_attrs[2], attr_name) == 7
    assert getattr(graph.edge_attrs[(1, 2)], attr_name) == 9
