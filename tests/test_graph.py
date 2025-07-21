import numpy as np
import pytest

import spatial_graph as sg

node_dtypes = ["uint16"]
node_attr_dtypes = [{"position": "double[2]"}]
edge_attr_dtypes = [{"score": "float64", "color": "uint8"}]


@pytest.mark.parametrize("node_dtype", node_dtypes)
@pytest.mark.parametrize("node_attr_dtypes", node_attr_dtypes)
@pytest.mark.parametrize("edge_attr_dtypes", edge_attr_dtypes)
@pytest.mark.parametrize("cls", [sg.Graph, sg.DiGraph])
def test_construction(node_dtype, node_attr_dtypes, edge_attr_dtypes, cls):
    obj1 = cls(node_dtype, node_attr_dtypes, edge_attr_dtypes)
    obj2 = cls(node_dtype, node_attr_dtypes, edge_attr_dtypes)
    assert type(obj1) is type(obj2)


@pytest.mark.parametrize("cls", [sg.Graph, sg.DiGraph])
def test_operations(cls):
    graph = cls("uint64", {"score": "float"}, {"score": "float"})

    nodes = [1, 2, 3, 4, 5]
    graph.add_nodes(
        np.array(nodes, dtype="uint64"),
        score=np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype="float32"),
    )
    num_added = 0
    for u in nodes:
        for v in nodes:
            if v == u:
                continue
            if not isinstance(graph, sg.DiGraph) and u > v:
                continue
            num_added += graph.add_edge(
                np.array([u, v], dtype="uint64"), score=u * 100 + v
            )

    assert graph.num_edges() == num_added

    if isinstance(graph, sg.DiGraph):
        assert graph.num_edges() == len(nodes) ** 2 - len(nodes)

        for node in nodes:
            in_neighbors = graph.num_in_neighbors(np.array([node], dtype="uint64"))
            out_neighbors = graph.num_in_neighbors(np.array([node], dtype="uint64"))
            assert len(in_neighbors) == 1
            assert len(out_neighbors) == 1
            assert in_neighbors[0] == len(nodes) - 1
            assert out_neighbors[0] == len(nodes) - 1

        for edge, attrs in graph.out_edges(data=True):
            assert attrs.score == edge[0] * 100 + edge[1]

        for edge, attrs in graph.in_edges(data=True):
            assert attrs.score == edge[0] * 100 + edge[1]

    else:
        assert graph.num_edges() == (len(nodes) ** 2 - len(nodes)) / 2

        for node in nodes:
            neighbors = graph.num_neighbors(np.array([node], dtype="uint64"))
            assert len(neighbors) == 1
            assert neighbors[0] == len(nodes) - 1

        for edge, attrs in graph.edges(data=True):
            assert attrs.score == edge[0] * 100 + edge[1]


def test_directed_edges():
    graph = sg.DiGraph("uint64")
    graph.add_nodes(np.array([0, 1, 2], dtype="uint64"))
    graph.add_edges(np.array([[0, 1], [1, 2], [2, 0]], dtype="uint64"))

    # all in edges
    in_edges = sorted(list(graph.in_edges()))
    assert len(in_edges) == 3
    np.testing.assert_array_equal(in_edges, [[0, 1], [1, 2], [2, 0]])

    # all out edges
    out_edges = sorted(list(graph.out_edges()))
    assert len(out_edges) == 3
    np.testing.assert_array_equal(out_edges, [[0, 1], [1, 2], [2, 0]])

    # in/out edges per node
    in_edges_0 = list(graph.in_edges(0))
    assert len(in_edges_0) == 1
    np.testing.assert_array_equal(in_edges_0, [[2, 0]])
    out_edges_0 = list(graph.out_edges(0))
    assert len(out_edges_0) == 1
    np.testing.assert_array_equal(out_edges_0, [[0, 1]])

    # in/out edges for list of nodes
    in_edges_01 = graph.in_edges_by_nodes(np.array([0, 1], dtype="uint64"))
    np.testing.assert_array_equal(in_edges_01, [[2, 0], [0, 1]])
    out_edges_01 = graph.out_edges_by_nodes(np.array([0, 1], dtype="uint64"))
    np.testing.assert_array_equal(out_edges_01, [[0, 1], [1, 2]])


def test_attribute_modification():
    graph = sg.Graph(
        "uint64",
        {"attr1": "double", "attr2": "int", "attr3": "float32[3]"},
        {"attr1": "int[4]"},
    )

    graph.add_nodes(
        np.array([1, 2, 3, 4, 5], dtype="uint64"),
        attr1=np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype="double"),
        attr2=np.array([1, 2, 3, 4, 5], dtype="int"),
        attr3=np.array(
            [
                [0.1, 0.2, 0.3],
                [0.1, 0.2, 0.3],
                [0.1, 0.2, 0.3],
                [0.1, 0.2, 0.3],
                [0.1, 0.2, 0.3],
            ],
            dtype="float32",
        ),
    )

    graph.add_edges(
        np.array([[1, 2], [3, 4], [5, 1]], dtype="uint64"),
        attr1=np.array(
            [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]],
            dtype="int",
        ),
    )

    # modify via generators:

    for node, attrs in graph.node_attrs:
        attrs.attr1 += 10.0
        attrs.attr2 *= 2
        attrs.attr3 *= np.float32(3.0)

    for node, attrs in graph.node_attrs:
        assert attrs.attr1 == (node / 10.0) + 10.0
        assert attrs.attr2 == node * 2
        np.testing.assert_array_almost_equal(attrs.attr3, [0.3, 0.6, 0.9])

    # modify via attribute views (single item):

    graph.node_attrs[1].attr1 = 1.0
    graph.node_attrs[1].attr1 += 1.0
    assert graph.node_attrs[1].attr1 == 2.0

    # modify via attribute views (bulk):

    graph.node_attrs[[2, 3, 4]].attr2 = np.array([20, 30, 40])
    assert graph.node_attrs[2].attr2 == 20
    assert graph.node_attrs[3].attr2 == 30
    assert graph.node_attrs[4].attr2 == 40

    graph.node_attrs[[2, 3, 4]].attr2 += np.array([20, 30, 40])
    assert graph.node_attrs[2].attr2 == 40
    assert graph.node_attrs[3].attr2 == 60
    assert graph.node_attrs[4].attr2 == 80

    graph.node_attrs[[2, 3, 4]].attr3 += np.array(
        [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
        ],
        dtype="float32",
    )
    np.testing.assert_array_almost_equal(graph.node_attrs[2].attr3, [1.3, 1.6, 1.9])
    np.testing.assert_array_almost_equal(graph.node_attrs[3].attr3, [2.3, 2.6, 2.9])
    np.testing.assert_array_almost_equal(graph.node_attrs[4].attr3, [3.3, 3.6, 3.9])

    # modify edge attribute
    np.testing.assert_array_equal(
        graph.edge_attrs[[[1, 2], [5, 1]]].attr1,
        [
            [1, 2, 3, 4],
            [3, 4, 5, 6],
        ],
    )
    graph.edge_attrs[[[1, 2], [5, 1]]].attr1 = np.array(
        [
            [11, 22, 33, 44],
            [30, 40, 50, 60],
        ],
        dtype="int",
    )
    np.testing.assert_array_equal(
        graph.edge_attrs[[[1, 2], [3, 4], [5, 1]]].attr1,
        [
            [11, 22, 33, 44],
            [2, 3, 4, 5],
            [30, 40, 50, 60],
        ],
    )


def test_missing_nodes_edges():
    graph = sg.Graph("uint64", {"node_attr": "float32"}, {"edge_attr": "float32"})
    graph.add_nodes(
        np.array([1, 2, 3, 4, 5], dtype="uint64"),
        node_attr=np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype="float32"),
    )
    graph.add_edges(
        np.array([[1, 2], [3, 4], [5, 1]], dtype="uint64"),
        edge_attr=np.array([0.1, 0.2, 0.3], dtype="float32"),
    )

    with pytest.raises(IndexError):
        graph.node_attrs[6].node_attr

    with pytest.raises(IndexError):
        graph.node_attrs[[4, 5, 6]].node_attr

    with pytest.raises(IndexError):
        graph.edge_attrs[(1, 3)].edge_attr

    with pytest.raises(IndexError):
        graph.edge_attrs[[(1, 2), (2, 4), (5, 1)]].edge_attr

    assert len(graph.node_attrs[[]].node_attr) == 0
    assert len(graph.edge_attrs[[]].edge_attr) == 0


def test_missing_attribute():
    graph = sg.Graph("uint64")
    graph.add_nodes(np.array([1, 2, 3, 4, 5], dtype="uint64"))
    graph.add_edges(np.array([[1, 2], [3, 4], [5, 1]], dtype="uint64"))

    with pytest.raises(AttributeError):
        graph.node_attrs[5].doesntexist
