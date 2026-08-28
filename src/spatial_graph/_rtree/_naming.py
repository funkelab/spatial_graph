"""Deterministic naming for prebuilt RTree extension modules.

Shared by the runtime lookup and `setup.py`, so the two can never disagree.
Deliberately depends only on `_dtypes` -- it sits on the import path of every
`PointRTree`, including installs with neither Cheetah nor witty available.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from spatial_graph._dtypes import DType

if TYPE_CHECKING:
    from .rtree import RTree

# subpackage holding ahead-of-time compiled modules; empty in a source checkout
PREBUILT_PACKAGE = f"{__package__}._prebuilt"


def _c_name(dtype: DType) -> str:
    """Canonical, identifier-safe name for a dtype ("int64", "float", "int64x2")."""
    base = dtype.base_c_type.removesuffix("_t")
    return f"{base}x{dtype.size}" if dtype.is_array else base


def module_name(cls: type[RTree], item_dtype: str, coord_dtype: str, dims: int) -> str:
    """Deterministic module name for the given tree parameters.

    Dtypes are canonicalized (so `int` and `int64` agree) and spelled out for
    readability. The trailing digest covers the C/pyx code `cls` injects into the
    template, so a subclass with custom code can never be served a prebuilt
    module compiled from different code. It deliberately does not cover the
    template itself, the vendored C, or the compiler flags: keeping those in
    step is the build's job (see `depends` in `setup.py`).
    """
    parts = (
        cls.pyx_item_t_declaration,
        cls.c_item_t_declaration,
        cls.c_converter_functions,
        cls.c_equal_function,
        cls.c_distance_function,
    )
    digest = hashlib.sha256("\0".join(parts).encode()).hexdigest()[:8]
    item = _c_name(DType(item_dtype))
    coord = _c_name(DType(coord_dtype))
    return f"rtree_{item}_{coord}_d{dims}_{digest}"
