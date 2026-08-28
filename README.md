# spatial-graph

[![License](https://img.shields.io/pypi/l/spatial-graph.svg?color=green)](https://github.com/funkelab/spatial_graph/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/spatial-graph.svg?color=green)](https://pypi.org/project/spatial-graph)
[![Python Version](https://img.shields.io/pypi/pyversions/spatial-graph.svg?color=green)](https://python.org)
[![CI](https://github.com/funkelab/spatial_graph/actions/workflows/ci.yml/badge.svg)](https://github.com/funkelab/spatial_graph/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/funkelab/spatial_graph/branch/main/graph/badge.svg)](https://codecov.io/gh/funkelab/spatial_graph)
[![CodSpeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/funkelab/spatial_graph)

`spatial_graph` provides a data structure for directed and undirected graphs,
where each node has an nD position (in time or space).

## Design Principles

### Goals

* support for arbitrary number of dimensions
* typed node identifiers and attributes
    * any fixed-length type that is supported by `numpy`
* efficient node/edge queries by
    * ROI
    * kNN (by points / lines)
* numpy-like interface for efficient:
    * graph population and manipulation
    * query results
    * attribute access
* minimal memory footprint
* minimal dependencies
    * `cython` / `witty` / `cheetah3`, used only when something has to be
      compiled at runtime (see Cross-Platform Support)
    * numpy for array interfaces
* PYX API for graph algorithms in C/C++

### Non-Goals

* graph algorithms
* I/O
* non-typed arguments
* non-spatial graphs
* out-of-memory support
* networkx compatibility

## Python API

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

## Implementation Details

A `SpatialGraph` consists of three data structures:

* The `Graph` itself, holding nodes, edges, and their attributes
  ([graphlite](https://github.com/haasdo95/graphlite)).
* Two R-trees for spatial node and edge queries (based on
  [rtree.c](https://github.com/tidwall/rtree.c)). We modified the original code
  to also include a fast kNN search.

## Cross-Platform Support

`spatial_graph` generates specialized C/C++ for the exact data types you ask
for. Where those types can be known in advance we compile them ahead of time
and ship them in the wheels; everything else is compiled on your machine the
first time it is used, which needs a C compiler.

**No compiler needed.** The PyPI wheels contain prebuilt `PointRTree` and
`LineRTree` variants for the common combinations: `int64`/`uint64` items,
`float32`/`float64` coordinates, and 2 to 5 dimensions. If your R-tree matches
one of those -- as most do -- nothing is compiled, on any supported Python.

**Compiler needed.** Two cases fall back to compiling at runtime:

1. `Graph`, `DiGraph`, `SpatialGraph` and `SpatialDiGraph`. Their node and edge
   attribute types are only known when you construct the graph, so they cannot
   be enumerated ahead of time.
2. R-trees outside the prebuilt set above (an `int32` item type, say, or 6
   dimensions).

If you or your users need those without a compiler, you can still install
`spatial_graph` from `conda-forge`, where we include a compiler (`clang`) in
its dependencies.

The wheels are `abi3` (stable ABI) and require Python 3.11 or newer, so one
wheel per platform covers every supported CPython. Python 3.10 users should
pin to a release before this one.

### Why can't everything be prebuilt?

There is no cross-platform C/C++ compiler that we can install using `pip`.
[`numba`](https://github.com/numba/numba) is maybe the closest to having solved
that problem: `numba` does compile during runtime even if you don't have a
compiler locally installed. This works because `numba` is generating LLVM IR,
an intermediate representation language that LLVM can compile into machine
code. `numba` depends on [`llvmlite`](https://github.com/numba/llvmlite), which
provides a subset of the LLVM API, statically linked into the binaries in that
package. This is just enough to compile the `numba` generated LLVM IR into
machine code. We can't use this strategy, because we compile general C/C++
code. Converting that into LLVM IR is exactly what we need a compiler for.

## For Developers

To create a new release, tag the current commit with a
version number and push it to the `upstream` remote:

```bash
git tag -a "vX.Y.Z" -m "vX.Y.Z"
git push upstream --follow-tags
```

This will trigger the CI workflow, which will build the package and upload it to PyPI.

### Testing in a conda environment

To simulate a naive user environment, with *no* assumptions made about the
availability of a C/C++ compiler, you can run the included Dockerfile
(where the key part of the conda env is the `compilers` package):

```bash
docker build -t spatial_graph .
docker run --rm spatial_graph
```
