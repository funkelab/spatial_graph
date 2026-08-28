"""Compile RTree variants ahead of time into a stable-ABI (abi3) wheel.

Renders the same pyx wrappers the runtime would JIT-compile (via
`_rtree._codegen`) for every variant in `iter_specs()`, so the two paths are
generated from one source. One wheel per platform then covers every supported
CPython, and those variants are never compiled on the user's machine -- so no
C compiler is invoked for them.

Set `SPATIAL_GRAPH_NO_PREBUILT=1` to build a pure-Python wheel instead, or
`SPATIAL_GRAPH_REQUIRE_PREBUILT=1` (as CI does) to turn a failure to compile
into a hard error rather than a silent fall back to JIT.
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from pathlib import Path

from setuptools import Extension, setup

ROOT = Path(__file__).parent
SRC = ROOT / "src"
PREBUILT_PKG = "spatial_graph._rtree._prebuilt"

# The wrappers pass numpy arrays as typed memoryviews, which compile to
# PyObject_GetBuffer/PyBuffer_Release. Those entered the limited API in 3.11
# (moved from cpython/object.h, excluded under Py_LIMITED_API, to pybuffer.h),
# so 3.11 is the floor for a stable-ABI build -- and matches requires-python.
ABI3_MIN = (3, 11)
ABI3_TAG = f"cp{ABI3_MIN[0]}{ABI3_MIN[1]}"
ABI3_HEX = f"0x{ABI3_MIN[0]:02x}{ABI3_MIN[1]:02x}0000"

WIN = sys.platform == "win32"


def _src_on_path() -> None:
    """Make the package under `src/` importable, for the helpers shared with it.

    Callers import `spatial_graph` only once they know they need it: importing
    it at module scope would make every command -- `sdist` and `egg_info`
    included -- fail if anything the package imports is missing from the build
    environment.
    """
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))


def prebuilt_extensions() -> list[Extension]:
    """Render every prebuilt RTree variant and declare it as an extension."""
    from Cython.Build import cythonize

    _src_on_path()
    from spatial_graph._rtree._codegen import build_wrapper, iter_specs
    from spatial_graph._rtree._naming import module_name

    pyx_dir = ROOT / "build" / "prebuilt-pyx"
    pyx_dir.mkdir(parents=True, exist_ok=True)

    # the template `#include`s these at compile time, so they never appear in
    # `sources`; without `depends` an edit to the C core would leave every
    # prebuilt module stale (the JIT path declares the same set to witty)
    rtree_dir = SRC / "spatial_graph" / "_rtree"
    depends = [str(rtree_dir / "src" / f) for f in ("rtree.c", "rtree.h", "config.h")]

    extensions = []
    for spec in iter_specs():
        name = module_name(spec.cls, spec.item_dtype, spec.coord_dtype, spec.dims)
        source = build_wrapper(spec.cls, spec.item_dtype, spec.coord_dtype, spec.dims)
        path = pyx_dir / f"{name}.pyx"
        # only rewrite when changed, so cythonize can skip unchanged variants
        if not path.is_file() or path.read_text() != source:
            path.write_text(source)
        extensions.append(
            Extension(
                f"{PREBUILT_PKG}.{name}",
                sources=[str(path)],
                depends=depends,
                include_dirs=[str(rtree_dir)],
                extra_compile_args=["/O2"] if WIN else ["-O3", "-Wno-unreachable-code"],
                define_macros=[
                    ("Py_LIMITED_API", ABI3_HEX),
                    *([("RTREE_NOATOMICS", "1")] if WIN else []),
                ],
                py_limited_api=True,
            )
        )

    return cythonize(
        extensions,
        language_level=3,
        quiet=True,
        nthreads=0 if WIN else os.cpu_count(),
    )


def can_compile() -> bool:
    """Whether this machine can build a C extension at all.

    Includes `Python.h` so that a box with a C compiler but no development
    headers -- the usual shape of this failure -- is caught here rather than
    part-way through the real build.
    """
    try:
        from distutils.ccompiler import new_compiler
        from distutils.sysconfig import customize_compiler, get_python_inc

        compiler = new_compiler()
        customize_compiler(compiler)  # picks up CC/CFLAGS, as build_ext does
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp, "probe.c")
            probe.write_text("#include <Python.h>\nint main(void) { return 0; }\n")
            compiler.compile(
                [str(probe)],
                output_dir=tmp,
                # get_python_inc, not sysconfig: inside a venv the latter
                # points at the venv, which holds no headers
                include_dirs=[get_python_inc()],
            )
    except Exception:
        return False
    return True


# Rendering and cythonizing every variant is wasted work for commands that only
# want metadata -- without this, `build --sdist` pays for a full codegen pass.
METADATA_ONLY = {"egg_info", "dist_info", "sdist"}
NEEDS_EXTENSIONS = {
    "bdist_egg",
    "bdist_wheel",
    "build",
    "build_ext",
    "build_py",
    "develop",
    "editable_wheel",
    "install",
}


def metadata_only() -> bool:
    """Whether this invocation asks for nothing that needs the extensions.

    Matches on recognized command names only, so option values (`--dist-dir
    /tmp/x`) are ignored, and anything unrecognized falls through to building
    -- skipping wrongly would silently yield a wheel with no prebuilt modules.
    """
    seen = {arg for arg in sys.argv[1:] if arg in METADATA_ONLY | NEEDS_EXTENSIONS}
    return bool(seen) and seen <= METADATA_ONLY


def should_prebuild() -> bool:
    """Whether to compile prebuilt variants into this wheel."""
    if metadata_only():
        return False

    _src_on_path()
    from spatial_graph._rtree._naming import env_enabled

    if env_enabled("SPATIAL_GRAPH_NO_PREBUILT"):
        return False
    if env_enabled("SPATIAL_GRAPH_REQUIRE_PREBUILT"):
        return True  # CI: never let a build silently degrade
    if can_compile():
        return True
    # Installing from an sdist without a compiler must keep working: fall back to
    # a pure-Python wheel that JIT-compiles on first use, as it did before
    # prebuilding existed.
    warnings.warn(
        "No usable C compiler found; building spatial-graph without prebuilt "
        "rtree modules. A C compiler will be needed the first time an RTree is "
        "used.",
        stacklevel=1,
    )
    return False


if __name__ == "__main__":
    if should_prebuild():
        setup(
            ext_modules=prebuilt_extensions(),
            options={
                "bdist_wheel": {"py_limited_api": ABI3_TAG},
                "build_ext": {"parallel": os.cpu_count()},
            },
        )
    else:
        setup(ext_modules=[])
