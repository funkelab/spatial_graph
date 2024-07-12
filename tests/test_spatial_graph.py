import spatial_graph as sg
import numpy as np


def test_roi_query():
    graph = sg.SpatialGraph(
        ndims=3,
        node_dtype="uint64",
        node_attr_dtypes={"position": "double[3]"},
        edge_attr_dtypes={"score": "float32"},
        position_attr="position",
        directed=False,
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

    nodes, edges = graph.query_in_roi(
        np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]), edge_inclusion="incident"
    )

    assert list(sorted(nodes)) == [1, 2]
    np.testing.assert_array_equal(edges, [[1, 2], [1, 5]])
