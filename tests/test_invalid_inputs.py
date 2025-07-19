import pytest

import spatial_graph as sg


def test_non_identifier_names():
    with pytest.raises(
        ValueError, match="Node attribute names must be valid identifiers"
    ):
        sg.Graph(
            ndims=4,
            node_dtype="uint16",
            node_attr_dtypes={"3Dposition": "double[4]"},
        )

    with pytest.raises(
        ValueError, match="Edge attribute names must be valid identifiers"
    ):
        sg.Graph(
            ndims=4,
            node_dtype="uint16",
            edge_attr_dtypes={"3Dposition": "double[4]"},
        )


def test_invalid_spatial_graphs():
    with pytest.raises(
        ValueError, match="position attribute 'not_position' not defined in"
    ):
        sg.SpatialGraph(
            ndims=4,
            node_dtype="uint16",
            node_attr_dtypes={"position": "double[4]"},
            edge_attr_dtypes={"score": "double[4]"},
            position_attr="not_position",
        )


def test_deprecation():
    with pytest.warns(
        DeprecationWarning, match="The 'directed' argument is deprecated"
    ):
        graph = sg.Graph(
            node_dtype="uint64",
            node_attr_dtypes={"position": "double[3]"},
            edge_attr_dtypes={"score": "float32"},
            directed=True,  # This should raise a deprecation warning
        )

    assert isinstance(graph, sg.DiGraph)
