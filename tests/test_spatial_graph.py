import numpy as np
import pytest

import spatial_graph as sg
from spatial_graph._util import create_graph

node_dtypes = ["uint16"]
node_attr_dtypes = [{"position": "double[4]"}]
edge_attr_dtypes = [{"score": "float64", "color": "uint8"}]


@pytest.mark.parametrize("ndims", [None, 4])
@pytest.mark.parametrize("directed", [True, False])
def test_util_construction(ndims: int, directed: bool) -> None:
    graph = create_graph(  # type: ignore
        node_dtype="uint16",
        ndims=ndims,
        node_attr_dtypes={"position": "double[4]"},
        edge_attr_dtypes={"score": "float64", "color": "uint8"},
        directed=directed,
    )
    assert isinstance(graph, sg.GraphBase)
    assert isinstance(graph, sg.DiGraph) == directed
    assert isinstance(graph, sg.SpatialGraphBase) == (ndims is not None)


def test_roi_query():
    graph = sg.SpatialGraph(
        ndims=3,
        node_dtype="uint64",
        node_attr_dtypes={"position": "double[3]"},
        edge_attr_dtypes={"score": "float32"},
        position_attr="position",
    )

    graph.add_nodes(
        np.array([1, 2, 3, 4, 5], dtype="uint64"),
        position=np.array(
            [
                [0.1, 0.1, 0.1],
                [0.2, 0.2, 0.2],
                [0.3, 0.3, 0.3],
                [0.4, 0.4, 0.4],
                [0.5, 0.5, 0.5],
            ],
            dtype="double",
        ),
    )

    graph.add_edges(
        np.array([[1, 2], [3, 4], [5, 1]], dtype="uint64"),
        score=np.array([0.2, 0.3, 0.4], dtype="float32"),
    )

    nodes = graph.query_nodes_in_roi(np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]))
    edges = graph.query_edges_in_roi(np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]))

    assert list(sorted(nodes)) == [1, 2]
    np.testing.assert_array_equal(edges, [[1, 2], [5, 1]])

    # query in a ROI that does not contain any nodes/edges

    nodes = graph.query_nodes_in_roi(np.array([[1.0, 1.0, 1.0], [1.25, 1.25, 1.25]]))
    edges = graph.query_edges_in_roi(np.array([[1.0, 1.0, 1.0], [1.25, 1.25, 1.25]]))

    assert len(nodes) == 0
    assert len(edges) == 0


def test_delete():
    graph = sg.SpatialGraph(
        ndims=3,
        node_dtype="uint64",
        node_attr_dtypes={"position": "double[3]"},
        edge_attr_dtypes={"score": "float32"},
        position_attr="position",
    )
    nodes = np.arange(0, 100_000).astype("uint64")
    graph.add_nodes(
        nodes,
        position=np.random.random(size=(100_000, 3)).astype("double"),
    )
    edges = np.random.randint(0, 100_000, size=(10_000, 2)).astype("uint64")
    graph.add_edges(
        edges,
        score=np.random.random(size=(10_000,)).astype("float32"),
    )

    graph.remove_nodes(nodes[:1000])

    assert len(graph) == 99_000
