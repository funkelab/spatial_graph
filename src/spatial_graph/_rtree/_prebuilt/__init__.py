"""Ahead-of-time compiled RTree modules, populated at build time by `setup.py`.

Empty in a plain source checkout: `_load_prebuilt` then finds nothing and every
tree is JIT-compiled, exactly as before prebuilding existed.
"""
