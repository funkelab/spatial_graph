History
=======

[Implementation from Joshua J Baker](https://github.com/tidwall/rtree.c), under
an MIT-style license (see LICENSE).

* modified to work with [witty](https://github.com/funkelab/witty)
* added kNN search

Data Structures
===============

R-Tree Itself
-------------

`rect`
`item`
    * `data` = node ID (type `item_data_t` defined in wrapper)
`node`
    * `rects`: list of `rect`s
    * `kind`:
        * `LEAF`: contains a list of `item`s, one for each `rect`
        * `BRANCH`: contains a list of `node`s, one for each `rect`

Priority Queue for kNN
----------------------

`element`
    * `distance`
    * `kind` (same enum as rtree kind):
        * `LEAF`: contains a `node*`
        * `BRANCH`: contains a `node*`
        * `ITEM`: contains an `item`
`priority_queue`
    * dynamically allocated array of `element`s

Implementation for Line Segments
================================

`item`:
    * `u`, `v`: IDs of endpoint nodes
    * `corner_mask`: (n-1)D bitmask for line segment corner in bounding box

`distance(coord_t point[], rect *bb, item *segment)`
    * compute distance of `point` to line `segment` bound by `bb`
`distance_bb(coord_t point[], rect *bb)`
    * previous implementation, computes distance to bounding box

`element`
    * `kind`
        * `LEAF`, `BRANCH` as before
        * `ITEM_BY_BB`: item, distance computed using bounding box
        * `ITEM`: item, distance computed exactly


Generalize for Arbitrary Items
==============================

User-provided "templates":

Level 0 (C)
-----------

`DIMS`
    * a `#define`, the number of dimensions

`KNN_USE_EXACT_DISTANCE`
    * an optional `#define`
    * if defined, the distance function below needs to be provided
    * if not defined, only distances to bounding boxes are used for the kNN search

`coord_t`
    * the scalar type of coordinates

`item_t`
    * a type (can be a `struct`) representing an item
    * keep small (e.g., only the ID of a node)

`distance(coord_t point[], rect *bb, item_t *item)`
    * a function that computes the exact distance of a point to the item
    * `point`: the query point
    * `bb`: the bounding box of the item (as given during insert)
    * `item`: the item to compute the distance for
    * if not given, should not attempt to compute exact distance and return
      item based on `distance_bb` right away

Level 1 (PYX)
-------------

This level holds glue code to bridge between C and Python.


### Definitions

`NUM_DIMS`
    * the number of dimensions
    * sets definition of `DIMS` used in Level 0

`KNN_USE_EXACT_DISTANCE`
    * as described above

### Data Types

`coord_t`
    * needs to be provided as:
        * C code: `typedef`
        * PYX code: `ctypedef`

`item_t`
    * needs to be provided as
        * C code: this is the `item_t` used in Level 0 (`typedef` or `struct`)
        * PYX code: this is just to wrap `item_t` to be used in PYX code (not user facing)
          ⇒ TODO: is that really needed?

`pyx_item_t`
    * semantically, the Python-wrapped version of `item_t` (user facing for
      insert and query)
    * has to be something that can be placed in a numpy array / memory view
        * this is part of the interface for efficient insert and query
        * similar to attributes, nodes, and edge lists in the graph, the
          interface uses numpy arrays
    * this means it is either a numpy type or an array of that, i.e.:
        * dtype    -> dtype[::1] for efficient insert/query
        * dtype[n] -> dtype[:, ::1] for efficient insert/query (where n is the size of the item)
    * needs to be provided as:
        * C code: `typedef`
        * PYX code: `ctypedef`

`NP_ITEM_DTYPE`
    * the `dtype` of `pyx_item_t` (e.g., `uint64`), used to create numpy arrays

`API_ITEMS_MEMVIEW_TYPE`
    * the type of a list of `pyx_item_t`s, either
        * dtype[::1] or
        * dtype[:, ::1]

### Strings

`ITEM_LENGTH`
    * `d` if an array type, empty string otherwise

`ITEMS_EXTRA_DIMS_0`
    * extra indices to index an element: `item[i, 0]` would be ", 0"

### Functions

`item_t pyx_to_c_item(pyx_item_t pyx_item, *)`
    * needs to be provided as:
        * C code: function implementation
        * PYX code: `cdef` function declaration
    ⇒ TODO: can be PYX function only?

`void copy_c_to_pyx_item(const item_t item, pyx_item_t *pyx_item)`
    * copy and convert C-item `item` to address of `pyx_item`
    * needs to be provided as:
        * C code: function implementation
        * PYX code: `cdef` function declaration
    ⇒ TODO: can be PYX function only?

`distance(coord_t point[], rect *bb, item_t *item)`
    * only C implementation needed
    * skip if `KNN_USE_EXACT_DISTANCE` is not set
