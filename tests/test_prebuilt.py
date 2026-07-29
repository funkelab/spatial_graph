"""Tests for ahead-of-time compiled rtree modules.

The `prebuilt` marked tests only mean something against an installed wheel; in a
source checkout there is no `_prebuilt` subpackage and they are skipped.
"""

from __future__ import annotations

import numpy as np
import pytest

from spatial_graph import PointRTree
from spatial_graph._rtree._codegen import iter_specs
from spatial_graph._rtree._naming import module_name
from spatial_graph._rtree.rtree import _load_prebuilt

# the `_prebuilt` package always exists but is empty in a source checkout, so
# probe for a real module rather than for the package
has_prebuilt = _load_prebuilt(PointRTree, "int64", "float32", 2) is not None
requires_prebuilt = pytest.mark.skipif(
    not has_prebuilt, reason="no prebuilt modules in this install"
)


@requires_prebuilt
@pytest.mark.parametrize("spec", list(iter_specs()), ids=str)
def test_every_declared_spec_is_shipped(spec):
    """Every variant in `_specs` must actually resolve to a prebuilt module."""
    assert _load_prebuilt(spec.cls, spec.item_dtype, spec.coord_dtype, spec.dims)


@requires_prebuilt
def test_prebuilt_is_used_and_correct():
    tree = PointRTree("int64", "float32", 3)
    assert "_prebuilt" in type(tree._ctree).__module__

    items = np.array([10, 20, 30], dtype="int64")
    points = np.ascontiguousarray([[0, 0, 0], [1, 1, 1], [9, 9, 9]], dtype="float32")
    tree.insert_point_items(items, points)

    lo, hi = np.array([0, 0, 0], "float32"), np.array([2, 2, 2], "float32")
    assert sorted(tree.search(lo, hi).ravel().tolist()) == [10, 20]
    assert tree.nearest(np.array([8.9, 8.9, 8.9], "float32"), 1).ravel()[0] == 30


def test_dtype_aliases_share_a_module():
    """`int`/`int64` and `float32`/`float` must not compile separate modules."""
    assert module_name(PointRTree, "int", "float32", 3) == module_name(
        PointRTree, "int64", "float", 3
    )


@pytest.mark.parametrize(
    ("item_dtype", "coord_dtype", "dims"),
    [("int32", "float32", 3), ("int64", "float32", 99)],
)
def test_unlisted_combination_falls_back_to_jit(item_dtype, coord_dtype, dims):
    assert _load_prebuilt(PointRTree, item_dtype, coord_dtype, dims) is None


def test_subclass_with_custom_code_is_not_served_a_prebuilt_module():
    class CustomEquality(PointRTree):
        c_equal_function = """
inline bool equal(const item_t a, const item_t b) { return a == b; }
"""

    assert module_name(CustomEquality, "int64", "float32", 3) != module_name(
        PointRTree, "int64", "float32", 3
    )
    assert _load_prebuilt(CustomEquality, "int64", "float32", 3) is None


def test_no_prebuilt_env_var_forces_jit(monkeypatch):
    monkeypatch.setenv("SPATIAL_GRAPH_NO_PREBUILT", "1")
    assert _load_prebuilt(PointRTree, "int64", "float32", 3) is None
