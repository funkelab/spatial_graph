from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal, overload

from ._graph import DiGraph, Graph
from ._spatial_graph import SpatialDiGraph, SpatialGraph

if TYPE_CHECKING:
    from collections.abc import Mapping


@overload
def create_graph(
    node_dtype: str,
    ndims: int,
    node_attr_dtypes: Mapping[str, str] | None = ...,
    edge_attr_dtypes: Mapping[str, str] | None = ...,
    position_attr: str | None = ...,
    directed: Literal[False] = ...,
) -> SpatialGraph: ...
@overload
def create_graph(
    node_dtype: str,
    ndims: int,
    node_attr_dtypes: Mapping[str, str] | None = ...,
    edge_attr_dtypes: Mapping[str, str] | None = ...,
    position_attr: str | None = ...,
    directed: Literal[True] = ...,
) -> SpatialDiGraph: ...
@overload
def create_graph(
    node_dtype: str,
    ndims: Literal[None] = ...,
    node_attr_dtypes: Mapping[str, str] | None = ...,
    edge_attr_dtypes: Mapping[str, str] | None = ...,
    position_attr: str | None = ...,
    directed: Literal[False] = ...,
) -> Graph: ...
@overload
def create_graph(
    node_dtype: str,
    ndims: Literal[None] = ...,
    node_attr_dtypes: Mapping[str, str] | None = ...,
    edge_attr_dtypes: Mapping[str, str] | None = ...,
    position_attr: str | None = ...,
    directed: Literal[True] = ...,
) -> DiGraph: ...
def create_graph(
    node_dtype: str,
    ndims: int | None = None,
    node_attr_dtypes: Mapping[str, str] | None = None,
    edge_attr_dtypes: Mapping[str, str] | None = None,
    position_attr: str | None = None,
    directed: bool = False,
) -> Graph | DiGraph | SpatialGraph | SpatialDiGraph:
    """Convenience factory function to create a graph instance.

    This will create the appropriate graph type based on the parameters provided.
    If `ndims` is specified, it will create a spatial graph; otherwise, it will create
    a non-spatial graph. If `directed` is True, it will create a directed graph;
    otherwise, it will create an undirected graph.

    Parameters
    ----------
    node_dtype : str
        The data type of the nodes in the graph.
    ndims : int or None, optional
        The number of dimensions for spatial graphs. If None, the graph is non-spatial.
    node_attr_dtypes : Mapping[str, str], optional
        A mapping of node attribute names to their data types.
    edge_attr_dtypes : Mapping[str, str], optional
        A mapping of edge attribute names to their data types.
    position_attr : str, optional
        The name of the attribute that holds the position of nodes in spatial graphs.
    directed : bool, optional
        Whether the graph is directed or not. Defaults to False.
    """
    if ndims is not None:  # Spatial graph
        cls = SpatialDiGraph if directed else SpatialGraph
        return cls(
            ndims=ndims,
            node_dtype=node_dtype,
            node_attr_dtypes=node_attr_dtypes,
            edge_attr_dtypes=edge_attr_dtypes,
            position_attr=position_attr or "position",
        )
    else:
        if position_attr is not None:  # pragma: no cover
            warnings.warn(
                "'position_attr' is ignored when 'ndims' is not specified.",
                UserWarning,
                stacklevel=2,
            )

        cls_ = DiGraph if directed else Graph
        return cls_(
            node_dtype=node_dtype,
            node_attr_dtypes=node_attr_dtypes,
            edge_attr_dtypes=edge_attr_dtypes,
        )
