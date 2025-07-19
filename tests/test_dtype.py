from __future__ import annotations

import pytest

from spatial_graph._dtypes import VALID_BASE_TYPES, DType


@pytest.mark.parametrize("size", [None, 2])
@pytest.mark.parametrize("base", VALID_BASE_TYPES.keys())
def test_dtype(base: str, size: int | None) -> None:
    if size is None:
        dtype = DType(base)
    else:
        dtype = DType(f"{base}[{size}]")
    assert dtype.base == base
    assert dtype.size == size
    assert dtype.is_array == (size is not None)
    c_base = VALID_BASE_TYPES[base]
    assert dtype.base_c_type == c_base
    assert dtype.to_c_decl("test") == f"{c_base} test" + (f"[{size}]" if size else "")

    assert dtype.to_pyxtype() == c_base + (f"[{size}]" if size else "")
    assert dtype.to_pyxtype(use_memory_view=True) == c_base + ("[::1]" if size else "")
    assert dtype.to_pyxtype(add_dim=True) == c_base + ("[:, ::1]" if size else "[::1]")
    assert dtype.to_pyxtype(use_memory_view=True, add_dim=True) == c_base + (
        "[:, ::1]" if size else "[::1]"
    )

    if size == 2:
        assert dtype.to_rvalue("test") == "{test[0], test[1]}"
        assert dtype.to_rvalue("test", "i") == "{test[i, 0], test[i, 1]}"
    elif size is None:
        assert dtype.to_rvalue("test") == "test"
        assert dtype.to_rvalue("test", "i") == "test[i]"


def test_bad_dtype() -> None:
    with pytest.raises(ValueError, match="Invalid dtype string"):
        DType("not-a-valid-dtype")
