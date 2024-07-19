import spatial_graph as sg
import pytest
import numpy as np


node_dtypes = ["int8", "uint8", "int16", "uint16"]
node_attr_dtypes = [
    {"position": "double"},
    {"position": "double[2]"},
    {"position": "int[4]"},
]
edge_attr_dtypes = [
    {"score": "float64"},
    {"score": "float64", "color": "uint8"},
]


@pytest.mark.parametrize("node_dtype", node_dtypes)
@pytest.mark.parametrize("node_attr_dtypes", node_attr_dtypes)
@pytest.mark.parametrize("edge_attr_dtypes", edge_attr_dtypes)
@pytest.mark.parametrize("directed", [True, False])
def test_construction(node_dtype, node_attr_dtypes, edge_attr_dtypes, directed):
    # TODO (directed graphs not yet wrapped)
    if directed:
        return
    graph = sg.Graph(node_dtype, node_attr_dtypes, edge_attr_dtypes, directed)


def test_attribute_modification():
    graph = sg.Graph(
        "uint64",
        {"attr1": "double", "attr2": "int"},
        {"attr1": "int[4]"},
        directed=False,
    )

    graph.add_nodes(
        np.array([1, 2, 3, 4, 5], dtype="uint64"),
        attr1=np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype="double"),
        attr2=np.array([1, 2, 3, 4, 5], dtype="int"),
    )

    graph.add_edges(
        np.array([[1, 2], [3, 4], [5, 1]], dtype="uint64"),
        attr1=np.array(
            [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]],
            dtype="int",
        ),
    )

    # modify via generators:

    for node, attrs in graph.nodes(data=True):
        attrs.attr1 += 10.0
        attrs.attr2 *= 2

    for node, attrs in graph.nodes(data=True):
        assert attrs.attr1 == (node / 10.0) + 10.0
        assert attrs.attr2 == node * 2

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
