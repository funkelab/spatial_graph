// Copyright 2023 Joshua J Baker. All rights reserved.
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

#ifndef RTREE_H
#define RTREE_H

#include <stdlib.h>
#include "config.h"

// rtree_new returns a new rtree
//
// Returns NULL if the system is out of memory.
struct rtree *rtree_new(void);


// rtree_new returns a new rtree using a custom allocator
//
// Returns NULL if the system is out of memory.
struct rtree *rtree_new_with_allocator(void *(*malloc)(size_t), void (*free)(void*));

// rtree_free frees an rtree
void rtree_free(struct rtree *tr);

// rtree_clone makes an instant copy of the btree.
//
// This operation uses shadowing / copy-on-write.
struct rtree *rtree_clone(struct rtree *tr);

// rtree_insert inserts an item into the rtree.
//
// This operation performs a copy of the data that is pointed to in the second
// and third arguments. The R-tree expects a rectangle, which is two arrays of
// coord_ts. The first N values as the minimum corner of the rect, and the next
// N values as the maximum corner of the rect, where N is the number of
// dimensions.
//
// When inserting points, the max coordinates is optional (set to NULL).
//
// Returns false if the system is out of memory.
bool rtree_insert(struct rtree *tr, const coord_t *min, const coord_t *max, const item_t item);


// rtree_search searches the rtree and iterates over each item that intersect
// the provided rectangle.
//
// Returning false from the iter will stop the search.
void rtree_search(const struct rtree *tr, const coord_t *min, const coord_t *max,
	bool (*iter)(const coord_t *min, const coord_t *max, const item_t item, void *udata),
	void *udata);

// Find the nearest neighbors to the given query point.
//
// Returning false from the iter will stop the search.
bool rtree_nearest(struct rtree *tr, const coord_t *point,
	bool (*iter)(const item_t item, coord_t distance, void *udata),
	void *udata);

// rtree_scan iterates over every item in the rtree.
//
// Returning false from the iter will stop the scan.
void rtree_scan(const struct rtree *tr,
	bool (*iter)(const coord_t *min, const coord_t *max, const item_t item, void *udata),
	void *udata);

// rtree_count returns the number of items in the rtree.
size_t rtree_count(const struct rtree *tr);

// query the total bounding box of the rtree
void rtree_bb(const struct rtree *tr, coord_t* min, coord_t* max);

// rtree_delete deletes an item from the rtree.
//
// This searches the tree for an item that is contained within the provided
// rectangle, and perform a binary comparison of its data to the provided
// data. The first item that is found is deleted.
//
// Returns the number of deleted items (0 or 1) or -1 if an OOM error occured.
int rtree_delete(struct rtree *tr, const coord_t *min, const coord_t *max, const item_t item);

// rtree_delete_with_comparator deletes an item from the rtree.
// This searches the tree for an item that is contained within the provided
// rectangle, and perform a comparison of its data to the provided data using
// a compare function. The first item that is found is deleted.
//
// Returns false if the system is out of memory.
bool rtree_delete_with_comparator(struct rtree *tr, const coord_t *min,
	const coord_t *max, const item_t item,
	int (*compare)(const item_t a, const item_t b, void *udata),
	void *udata);

// rtree_opt_relaxed_atomics activates memory_order_relaxed for all atomic
// loads. This may increase performance for single-threaded programs.
// Optionally, define RTREE_NOATOMICS to disbale all atomics.
void rtree_opt_relaxed_atomics(struct rtree *tr);

#endif // RTREE_H
