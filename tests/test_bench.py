import sys

import numpy as np
import pytest

import spatial_graph as sg

# either run this file directly or with pytest --codspeed
if all(x not in {"--codspeed", "tests/test_bench.py"} for x in sys.argv):
    pytest.skip(
        "use 'pytest tests/test_bench.py' to run benchmark", allow_module_level=True
    )


def _make_graph(
    ndims=3,
    node_dtype="uint64",
    node_attr_dtypes=None,
    edge_attr_dtypes=None,
    n_nodes=100_000,
):
    """Helper to create a SpatialGraph instance with default parameters."""
    graph = sg.create_graph(
        ndims=ndims,
        node_dtype=node_dtype,
        node_attr_dtypes=node_attr_dtypes or {"position": "double[3]"},
        edge_attr_dtypes=edge_attr_dtypes or {"score": "float32"},
        position_attr="position",
    )
    nodes = np.arange(n_nodes, dtype="uint64")
    positions = np.random.random((n_nodes, ndims))
    graph.add_nodes(nodes, position=positions)

    return graph


@pytest.mark.parametrize("num_queries", [100])
@pytest.mark.parametrize("k", [1000, 10000])
@pytest.mark.parametrize("n_nodes", [100_000, 1_000_000])
def test_bench_query_nearest_nodes(n_nodes: int, k: int, num_queries: int, benchmark):
    """Benchmark query_nearest_nodes."""
    graph = _make_graph(n_nodes=n_nodes)
    query_points = np.random.random((num_queries, 3))

    def _run():
        for i in range(num_queries):
            # Query nearest nodes
            closest, distances = graph.query_nearest_nodes(
                query_points[i], k=k, return_distances=True
            )
            positions = graph.node_attrs[closest].position
        return closest, distances, positions

    closest, distances, positions = benchmark(_run)

    # Verify results
    assert len(distances) == len(closest)
    assert positions.shape[1] == 3


@pytest.mark.parametrize("n_nodes", [100_000, 1_000_000])
def test_roi_query_performance(n_nodes, benchmark):
    """Benchmark ROI (region of interest) queries."""
    large_graph = _make_graph(n_nodes=n_nodes)
    # Define a ROI that should contain a reasonable number of nodes
    roi = np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])

    nodes_in_roi = benchmark(lambda: large_graph.query_nodes_in_roi(roi))

    # Verify results
    assert len(nodes_in_roi) > 0
    assert len(nodes_in_roi) < n_nodes  # Should be subset

    # Verify nodes are actually in ROI
    positions = large_graph.node_attrs[nodes_in_roi].position
    assert np.all(positions >= roi[0])
    assert np.all(positions <= roi[1])
