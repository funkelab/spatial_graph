# spatial-graph

`spatial-graph` provides a data structure for directed and undirected graphs,
where each node has an nD position (in time or space).

It leverages well-in-time compiled C++ code for efficient graph operations,
coupled with an rtree implementation for fast spatial queries.

## Goals

- Support for arbitrary number of dimensions
- Typed node identifiers and attributes
    - Any fixed-length type that is supported by `numpy`
- Efficient node/edge queries by
    - ROI
    - kNN (by points / lines)
- numpy-like interface for efficient:
    - Graph population and manipulation
    - Query results
    - Attribute access
- Minimal memory footprint
- Minimal dependencies
- PYX API for graph algorithms in C/C++

## Basic Usage

Graph creation:

```python
graph = sg.SpatialGraph(
    ndims=3,
    node_dtype="uint64",
    node_attr_dtypes={"position": "double[3]"},
    edge_attr_dtypes={"score": "float32"},
    position_attr="position",
)
```

Adding nodes/edges:

```python
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
```

Query nodes/edges in ROI:

```python
# nodes/edges will be numpy arrays of dtype uint64 and shape (n,)/(n, 2)
nodes = graph.query_nodes_in_roi(np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]))
edges = graph.query_edges_in_roi(np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]))
```

Query nodes/edges by position:

```python
nodes = graph.query_nearest_nodes(np.array([0.3, 0.3, 0.3]), k=3)
edges = graph.query_nearest_edges(np.array([0.3, 0.3, 0.3]), k=3)
```

Access node/edge attributes:

```python
node_positions = graph.node_attrs[nodes].position
edge_scores = graph.edge_attrs[edges].score
```

Delete nodes/edges:

```python
graph.remove_nodes(nodes[:1000])
```

See the [API documentation](./reference) for more details.
