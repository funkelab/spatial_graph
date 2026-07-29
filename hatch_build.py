"""Compile RTree variants ahead of time into stable-ABI (abi3) wheels.

Renders the same pyx wrappers the runtime would JIT-compile (via
`_rtree._codegen`) for every variant in `_rtree._specs`, builds them against
`Py_LIMITED_API`, and force-includes the result as
`spatial_graph/_rtree/_prebuilt/`. One wheel per platform then covers every
supported CPython, and users never need a C compiler for those variants.

Set `SPATIAL_GRAPH_NO_PREBUILT=1` to build a pure-Python wheel instead.
"""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

# oldest CPython with the buffer protocol (memoryviews) in the limited API
ABI3_MIN = (3, 11)
ABI3_HEX = f"0x{ABI3_MIN[0]:02x}{ABI3_MIN[1]:02x}0000"

ROOT = Path(__file__).parent
SRC = ROOT / "src"
PKG = "spatial_graph/_rtree/_prebuilt"


def _platform_tag() -> str:
    return sysconfig.get_platform().replace("-", "_").replace(".", "_")


def _stub_spatial_graph_package() -> None:
    """Make `spatial_graph.*` submodules importable without running its `__init__`.

    `spatial_graph/__init__.py` pulls in the graph half, and with it witty; the
    build only needs the rtree codegen, so we register a bare namespace package
    pointing at the source tree instead.
    """
    import types

    pkg = types.ModuleType("spatial_graph")
    pkg.__path__ = [str(SRC / "spatial_graph")]  # type: ignore[attr-defined]
    sys.modules.setdefault("spatial_graph", pkg)


class PrebuiltRTreeHook(BuildHookInterface):
    PLUGIN_NAME = "prebuilt-rtree"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if self.target_name != "wheel":
            return
        if os.getenv("SPATIAL_GRAPH_NO_PREBUILT"):
            return

        # 3.10 lacks the buffer protocol in the limited API, so it gets a plain
        # version-specific wheel; 3.11+ all share one abi3 wheel per platform.
        abi3 = sys.version_info >= ABI3_MIN

        _stub_spatial_graph_package()
        from spatial_graph._rtree._codegen import build_wrapper
        from spatial_graph._rtree._naming import module_name
        from spatial_graph._rtree._specs import iter_specs

        build_dir = ROOT / "build" / "prebuilt" / f"{_platform_tag()}-{abi3}"
        pyx_dir = build_dir / "pyx"
        pyx_dir.mkdir(parents=True, exist_ok=True)

        names = []
        for spec in iter_specs():
            name = module_name(spec.cls, spec.item_dtype, spec.coord_dtype, spec.dims)
            source = build_wrapper(
                spec.cls, spec.item_dtype, spec.coord_dtype, spec.dims
            )
            path = pyx_dir / f"{name}.pyx"
            # only rewrite when changed, so cythonize can skip unchanged variants
            if not path.is_file() or path.read_text() != source:
                path.write_text(source)
            names.append(name)

        try:
            built = self._compile(pyx_dir, names, build_dir, abi3)
        except Exception as e:
            # Installing from an sdist on a machine with no usable compiler must
            # keep working: fall back to a pure-Python wheel that JIT-compiles on
            # first use, exactly as before prebuilding existed.  CI sets
            # SPATIAL_GRAPH_REQUIRE_PREBUILT so this can never pass silently there.
            if os.getenv("SPATIAL_GRAPH_REQUIRE_PREBUILT"):
                raise
            self.app.display_warning(
                f"Could not prebuild rtree modules ({e}); building a pure-Python "
                "wheel. A C compiler will be needed the first time an RTree is used."
            )
            return

        force_include = build_data.setdefault("force_include", {})
        init = build_dir / "__init__.py"
        init.write_text("")
        force_include[str(init)] = f"{PKG}/__init__.py"
        for artifact in built:
            force_include[str(artifact)] = f"{PKG}/{artifact.name}"

        build_data["pure_python"] = False
        if abi3:
            build_data["tag"] = f"cp{ABI3_MIN[0]}{ABI3_MIN[1]}-abi3-{_platform_tag()}"
        else:
            build_data["infer_tag"] = True

    def _compile(
        self, pyx_dir: Path, names: list[str], build_dir: Path, abi3: bool
    ) -> list[Path]:
        from Cython.Build import cythonize
        from setuptools import Distribution, Extension

        rtree_src = SRC / "spatial_graph" / "_rtree"
        win = sys.platform == "win32"
        extensions = [
            Extension(
                f"{PKG.replace('/', '.')}.{name}",
                sources=[str(pyx_dir / f"{name}.pyx")],
                include_dirs=[str(rtree_src)],
                extra_compile_args=["/O2"] if win else ["-O3", "-Wno-unreachable-code"],
                define_macros=[
                    *([("Py_LIMITED_API", ABI3_HEX)] if abi3 else []),
                    *([("RTREE_NOATOMICS", "1")] if win else []),
                ],
                py_limited_api=abi3,
            )
            for name in names
        ]

        out = build_dir / "lib"
        dist = Distribution(
            {
                "name": "spatial_graph_prebuilt",
                "ext_modules": cythonize(
                    extensions, language_level=3, quiet=True, nthreads=os.cpu_count()
                ),
            }
        )
        cmd = dist.get_command_obj("build_ext")
        cmd.build_lib = str(out)
        cmd.build_temp = str(build_dir / "temp")
        cmd.parallel = os.cpu_count()
        cmd.ensure_finalized()
        cmd.run()

        built_pkg = out.joinpath(*PKG.split("/"))
        artifacts = sorted(
            p for p in built_pkg.iterdir() if p.suffix in (".so", ".pyd")
        )
        if len(artifacts) != len(names):
            raise RuntimeError(f"expected {len(names)} modules, built {len(artifacts)}")
        return artifacts
