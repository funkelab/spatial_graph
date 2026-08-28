"""Tests for ahead-of-time compiled rtree modules.

The `requires_prebuilt` tests only mean something against an install that
actually shipped them, and are skipped otherwise -- except when
`SPATIAL_GRAPH_REQUIRE_PREBUILT` is set (as CI does), where their absence is
the very regression we want to catch.
"""

from __future__ import annotations

import numpy as np
import pytest
import witty

from spatial_graph import PointRTree
from spatial_graph._rtree._codegen import iter_specs
from spatial_graph._rtree._naming import env_enabled, module_name
from spatial_graph._rtree.rtree import _load_prebuilt

# the `_prebuilt` package always exists but is empty in a source checkout, so
# probe for a real module rather than for the package
has_prebuilt = _load_prebuilt(PointRTree, "int64", "float32", 2) is not None
requires_prebuilt = pytest.mark.skipif(
    not has_prebuilt, reason="no prebuilt modules in this install"
)


def test_prebuilt_modules_were_shipped():
    """Guard against a wheel that silently degraded to pure Python."""
    if not env_enabled("SPATIAL_GRAPH_REQUIRE_PREBUILT"):
        pytest.skip("SPATIAL_GRAPH_REQUIRE_PREBUILT not enabled")
    assert has_prebuilt, "install shipped no prebuilt rtree modules"


@requires_prebuilt
@pytest.mark.parametrize("spec", list(iter_specs()), ids=str)
def test_declared_specs_are_shipped_and_never_compile(spec, monkeypatch):
    """Every declared variant must construct without invoking the compiler.

    This is the property prebuilding exists for. Blocking `compile_cython`
    asserts it directly, in the environment users actually get -- witty is an
    unconditional dependency, so it is always installed; what must not happen
    is that it gets *used*.
    """

    def no_compiling(*args, **kwargs):
        raise AssertionError(f"{spec} triggered runtime compilation")

    monkeypatch.setattr(witty, "compile_cython", no_compiling)
    tree = spec.cls(spec.item_dtype, spec.coord_dtype, spec.dims)
    assert "_prebuilt" in type(tree._ctree).__module__


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
