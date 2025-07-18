"""
Benchmark tests for spatial graph query performance.

These tests extract the core spatial query functionality from the VisPy demonstration
to provide atomic, reproducible benchmarks without GUI dependencies.
"""

import sys

import numpy as np
import pytest

from spatial_graph import SpatialGraph

if all(x not in {"--codspeed", "tests/test_bench.py"} for x in sys.argv):
    pytest.skip(
        "use 'pytest tests/test_bench.py' to run benchmark", allow_module_level=True
    )


def _make_graph(
    ndims=3,
    node_dtype="uint64",
    node_attr_dtypes=None,
    edge_attr_dtypes=None,
    directed=False,
    n_nodes=100_000,
):
    """Helper to create a SpatialGraph instance with default parameters."""
    if node_attr_dtypes is None:
        node_attr_dtypes = {"position": "double[3]"}
    if edge_attr_dtypes is None:
        edge_attr_dtypes = {"score": "float32"}

    graph = SpatialGraph(
        ndims=ndims,
        node_dtype=node_dtype,
        node_attr_dtypes=node_attr_dtypes,
        edge_attr_dtypes=edge_attr_dtypes,
        position_attr="position",
        directed=directed,
    )
    nodes = np.arange(n_nodes, dtype="uint64")
    positions = np.random.random((n_nodes, ndims))
    graph.add_nodes(nodes, position=positions)

    return graph


@pytest.mark.parametrize("n_nodes", [100_000])
def test_query_nearest_nodes_performance(n_nodes, benchmark):
    """Benchmark the core nearest neighbor query operation."""
    large_graph = _make_graph(n_nodes=n_nodes)
    query_point = np.array([0.5, 0.5, 0.5])

    # Benchmark the key operation from the VisPy demo
    def _run():
        return large_graph.query_nearest_nodes(
            query_point, k=10_000, return_distances=True
        )

    closest, distances = benchmark(_run)

    # Verify results are reasonable
    assert len(closest) == 10_000
    assert len(distances) == 10_000
    assert np.all(distances >= 0)
    assert np.all(np.diff(distances) >= 0)  # distances should be sorted


@pytest.mark.parametrize("n_nodes", [100_000])
def test_node_attribute_access_performance(n_nodes, benchmark):
    """Benchmark node attribute access as done in the VisPy demo."""
    large_graph = _make_graph(n_nodes=n_nodes)
    nodes = np.arange(n_nodes, dtype="uint64")

    # This is the operation that was being timed in the demo
    positions = benchmark(lambda: large_graph.node_attrs[nodes].position)

    # Verify results
    assert positions.shape == (n_nodes, 3)
    assert positions.dtype == np.float64


@pytest.mark.parametrize("n_nodes", [10_000])
def test_repeated_nearest_queries_performance(n_nodes, benchmark):
    """Benchmark repeated nearest neighbor queries as in mouse movement."""

    medium_graph = _make_graph(n_nodes=n_nodes)
    # Simulate multiple mouse positions (like in the VisPy demo)
    num_queries = 100
    query_points = np.random.random((num_queries, 3))

    def _run():
        for i in range(num_queries):
            # Query nearest nodes
            closest, distances = medium_graph.query_nearest_nodes(
                query_points[i], k=1000, return_distances=True
            )
            positions = medium_graph.node_attrs[closest].position
        return closest, distances, positions

    closest, distances, positions = benchmark(_run)

    # Verify results
    assert len(closest) <= 1000  # may be fewer if graph is smaller
    assert len(distances) == len(closest)
    assert positions.shape[1] == 3


@pytest.mark.parametrize("k_value", [1000, 10000])
@pytest.mark.parametrize("n_nodes", [100_000, 100_000])
def test_various_k_values_performance(n_nodes, k_value, benchmark):
    """Benchmark performance across different k values."""
    medium_graph = _make_graph(n_nodes=n_nodes)
    query_point = np.array([0.5, 0.5, 0.5])

    closest, distances = benchmark(
        lambda: medium_graph.query_nearest_nodes(
            query_point, k=k_value, return_distances=True
        )
    )

    expected_k = min(k_value, len(medium_graph.nodes))
    assert len(closest) == expected_k
    assert len(distances) == expected_k


@pytest.mark.parametrize("n_nodes", [100_000])
def test_roi_query_performance(n_nodes, benchmark):
    """Benchmark ROI (region of interest) queries."""
    large_graph = _make_graph(n_nodes=n_nodes)
    # Define a ROI that should contain a reasonable number of nodes
    roi = np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])

    nodes_in_roi = benchmark(lambda: large_graph.query_nodes_in_roi(roi))

    # Verify results
    assert len(nodes_in_roi) > 0
    assert len(nodes_in_roi) < 100_000  # Should be subset

    # Verify nodes are actually in ROI
    positions = large_graph.node_attrs[nodes_in_roi].position
    assert np.all(positions >= roi[0])
    assert np.all(positions <= roi[1])


@pytest.mark.parametrize("k", [1, 100, 1000, 10000])
@pytest.mark.parametrize("n_nodes", [10_000])
def test_nearest_query_correctness_and_performance(k: int, n_nodes: int, benchmark):
    """Test both correctness and performance of nearest neighbor queries."""
    medium_graph = _make_graph(n_nodes=n_nodes)
    query_point = np.array([0.0, 0.0, 0.0])  # Corner point

    closest, distances = benchmark(
        lambda: medium_graph.query_nearest_nodes(
            query_point, k=k, return_distances=True
        )
    )

    expected_k = min(k, len(medium_graph.nodes))

    # Correctness checks
    assert len(closest) == expected_k
    assert len(distances) == expected_k
    assert np.all(distances >= 0)
    assert np.all(np.diff(distances) >= 0)  # Should be sorted by distance
