// Copyright 2023 Joshua J Baker. All rights reserved.
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

#include <string.h>
#include <math.h>
#include "config.h"
#include "rtree.h"

// used for splits
#define MINITEMS_PERCENTAGE 10
#define MINITEMS ((MAXITEMS) * (MINITEMS_PERCENTAGE) / 100 + 1)

#ifndef RTREE_NOPATHHINT
#define USE_PATHHINT
#endif

#ifdef RTREE_MAXITEMS
#undef MAXITEMS
#define MAXITEMS RTREE_MAXITEMS
#endif

#ifdef RTREE_NOATOMICS
typedef int rc_t;
static int rc_load(rc_t *ptr, bool relaxed) {
	(void)relaxed; // nothing to do
	return *ptr;
}
static int rc_fetch_sub(rc_t *ptr, int val) {
	int rc = *ptr;
	*ptr -= val;
	return rc;
}
static int rc_fetch_add(rc_t *ptr, int val) {
	int rc = *ptr;
	*ptr += val;
	return rc;
}
#else
#include <stdatomic.h>
typedef atomic_int rc_t;
static int rc_load(rc_t *ptr, bool relaxed) {
	if (relaxed) {
		return atomic_load_explicit(ptr, memory_order_relaxed);
	} else {
		return atomic_load(ptr);
	}
}
static int rc_fetch_sub(rc_t *ptr, int delta) {
	return atomic_fetch_sub(ptr, delta);
}
static int rc_fetch_add(rc_t *ptr, int delta) {
	return atomic_fetch_add(ptr, delta);
}
#endif

#define abs(x) ((x) >= 0 ? (x) : -(x))

enum kind {
	LEAF = 1,
	BRANCH = 2,
	ITEM_BY_BB = 3, // for priority queue, item enqueued by bb distance
	ITEM = 4        // for priority queue, item enqueued with exact distance
};

struct rect {
	coord_t min[DIMS];
	coord_t max[DIMS];
};

struct node {
	rc_t rc;			// reference counter for copy-on-write
	enum kind kind;	 // LEAF or BRANCH
	int count;		  // number of rects
	struct rect rects[MAXITEMS];
	union {
		struct node *nodes[MAXITEMS];
		item_t items[MAXITEMS];
	};
};

// priority queue
struct element {
	coord_t distance;
	enum kind kind;
	union {
		struct node* node;  // if kind == LEAF or BRANCH
		struct {            // if kind == ITEM_BY_BB or ITEM
			item_t item;
			struct rect *rect;
		};
	};
};

struct priority_queue {
	size_t size;
	size_t capacity;
	struct element *elements;
};

struct priority_queue* priority_queue_new() {
	struct priority_queue *queue = (struct priority_queue*)malloc(sizeof(struct priority_queue));
	if (!queue)
		return NULL;
	queue->size = 0;
	queue->capacity = INITIAL_QUEUE_SIZE;
	queue->elements = (struct element*)malloc(sizeof(struct element) * queue->capacity);
	if (!queue->elements) {
		free(queue);
		return NULL;
	}
	return queue;
}

void priority_queue_free(struct priority_queue* queue) {
	if (queue->elements)
		free(queue->elements);
	free(queue);
}

void swap(struct element* a, struct element* b) {
	struct element tmp = *a;
	*a = *b;
	*b = tmp;
}

void heapify_up(struct priority_queue* queue, size_t index) {
	if (index == 0) return;
	size_t midpoint = (index - 1)/2;
	if (queue->elements[midpoint].distance > queue->elements[index].distance) {
		swap(&queue->elements[midpoint], &queue->elements[index]);
		heapify_up(queue, midpoint);
	}
}

void heapify_down(struct priority_queue* queue, size_t index) {
	size_t smallest = index;
	size_t left = 2*index + 1;
	size_t right = 2*index + 2;

	if (left < queue->size) {
		if (queue->elements[left].distance < queue->elements[smallest].distance) {
			smallest = left;
		}
	}
	if (right < queue->size) {
		if (queue->elements[right].distance < queue->elements[smallest].distance) {
			smallest = right;
		}
	}
	if (smallest != index) {
		swap(&queue->elements[index], &queue->elements[smallest]);
		heapify_down(queue, smallest);
	}
}

bool enqueue(struct priority_queue* queue, struct element element) {
	if (queue->size == queue->capacity) {
		queue->capacity *= 2;
		queue->elements = realloc(queue->elements, sizeof(struct element) * queue->capacity);
		if (!queue->elements)
			return false;
	}
	queue->elements[queue->size] = element;
	queue->size++;
	heapify_up(queue, queue->size - 1);
	return true;
}

struct element dequeue(struct priority_queue* queue) {

	struct element top = queue->elements[0];
	queue->elements[0] = queue->elements[--queue->size];
	heapify_down(queue, 0);
	// reclaim some memory when the queue is shrinking
	if (queue->size < queue->capacity/4) {
		queue->capacity /= 2;
		struct element *elements = realloc(queue->elements, sizeof(struct element) * queue->capacity);
		if (!elements) {
			queue->capacity *= 2;
		} else {
			queue->elements = elements;
		}
	}
	return top;
}

struct element peek(struct priority_queue* queue) {
	return queue->elements[0];
}
// end priority queue

struct rtree {
	struct rect rect;
	struct node *root;
	struct priority_queue *queue;
	size_t count;
	size_t height;
#ifdef USE_PATHHINT
	int path_hint[16];
#endif
	bool relaxed;
	void *(*malloc)(size_t);
	void (*free)(void *);
};

static inline coord_t min0(coord_t x, coord_t y) {
	return x < y ? x : y;
}

static inline coord_t max0(coord_t x, coord_t y) {
	return x > y ? x : y;
}

static bool feq(coord_t a, coord_t b) {
	return !(a < b || a > b);
}

#ifdef KNN_USE_EXACT_DISTANCE
inline coord_t distance(const coord_t point[], const struct rect *rect, const struct item_t item);
#endif

static struct node *node_new(struct rtree *tr, enum kind kind) {
	struct node *node = (struct node *)tr->malloc(sizeof(struct node));
	if (!node) return NULL;
	memset(node, 0, sizeof(struct node));
	node->kind = kind;
	return node;
}

static struct node *node_copy(struct rtree *tr, struct node *node) {
	struct node *node2 = (struct node *)tr->malloc(sizeof(struct node));
	if (!node2) return NULL;
	memcpy(node2, node, sizeof(struct node));
	node2->rc = 0;
	if (node2->kind == BRANCH) {
		for (int i = 0; i < node2->count; i++) {
			rc_fetch_add(&node2->nodes[i]->rc, 1);
		}
	}
	return node2;
}

static void node_free(struct rtree *tr, struct node *node) {
	if (rc_fetch_sub(&node->rc, 1) > 0) return;
	if (node->kind == BRANCH) {
		for (int i = 0; i < node->count; i++) {
			node_free(tr, node->nodes[i]);
		}
	} else {
	}
	tr->free(node);
}

#define cow_node_or(rnode, code) { \
	if (rc_load(&(rnode)->rc, tr->relaxed) > 0) { \
		struct node *node2 = node_copy(tr, (rnode)); \
		if (!node2) { code; } \
		rc_fetch_sub(&(rnode)->rc, 1); \
		(rnode) = node2; \
	} \
}

static void rect_expand(struct rect *rect, const struct rect *other) {
	for (int i = 0; i < DIMS; i++) {
		rect->min[i] = min0(rect->min[i], other->min[i]);
		rect->max[i] = max0(rect->max[i], other->max[i]);
	}
}

static coord_t rect_area(const struct rect *rect) {
	coord_t result = 1;
	for (int i = 0; i < DIMS; i++) {
		result *= (rect->max[i] - rect->min[i]);
	}
	return result;
}

// return the area of two rects expanded
static coord_t rect_unioned_area(const struct rect *rect,
	const struct rect *other)
{
	coord_t result = 1;
	for (int i = 0; i < DIMS; i++) {
		result *= (max0(rect->max[i], other->max[i]) -
				   min0(rect->min[i], other->min[i]));
	}
	return result;
}

static bool rect_contains(const struct rect *rect, const struct rect *other) {
	int bits = 0;
	for (int i = 0; i < DIMS; i++) {
		bits |= other->min[i] < rect->min[i];
		bits |= other->max[i] > rect->max[i];
	}
	return bits == 0;
}

static bool rect_contains_point(const struct rect *rect, const coord_t point[]) {
	int bits = 0;
	for (int i = 0; i < DIMS; i++) {
		bits |= point[i] < rect->min[i];
		bits |= point[i] > rect->max[i];
	}
	return bits == 0;
}

static bool rect_intersects(const struct rect *rect, const struct rect *other) {
	int bits = 0;
	for (int i = 0; i < DIMS; i++) {
		bits |= other->min[i] > rect->max[i];
		bits |= other->max[i] < rect->min[i];
	}
	return bits == 0;
}

static bool rect_onedge(const struct rect *rect, const struct rect *other) {
	for (int i = 0; i < DIMS; i++) {
		if (feq(rect->min[i], other->min[i]) ||
			feq(rect->max[i], other->max[i]))
		{
			return true;
		}
	}
	return false;
}

static bool rect_equals(const struct rect *rect, const struct rect *other) {
	for (int i = 0; i < DIMS; i++) {
		if (!feq(rect->min[i], other->min[i]) ||
			!feq(rect->max[i], other->max[i]))
		{
			return false;
		}
	}
	return true;
}

static bool rect_equals_bin(const struct rect *rect, const struct rect *other) {
	for (int i = 0; i < DIMS; i++) {
		if (rect->min[i] != other->min[i] ||
			rect->max[i] != other->max[i])
		{
			return false;
		}
	}
	return true;
}

static int rect_largest_axis(const struct rect *rect) {
	int axis = 0;
	coord_t nlength = rect->max[0] - rect->min[0];
	for (int i = 1; i < DIMS; i++) {
		coord_t length = rect->max[i] - rect->min[i];
		if (length > nlength) {
			nlength = length;
			axis = i;
		}
	}
	return axis;
}

// swap two rectangles
static void node_swap(struct node *node, int i, int j) {
	struct rect tmp = node->rects[i];
	node->rects[i] = node->rects[j];
	node->rects[j] = tmp;
	if (node->kind == LEAF) {
		item_t tmp = node->items[i];
		node->items[i] = node->items[j];
		node->items[j] = tmp;
	} else {
		struct node *tmp = node->nodes[i];
		node->nodes[i] = node->nodes[j];
		node->nodes[j] = tmp;
	}
}

struct rect4 {
	coord_t all[DIMS*2];
};

static void node_qsort(struct node *node, int s, int e, int index, bool rev) {
	int nrects = e - s;
	if (nrects < 2) {
		return;
	}
	int left = 0;
	int right = nrects-1;
	int pivot = nrects / 2;
	node_swap(node, s+pivot, s+right);
	struct rect4 *rects = (struct rect4 *)&node->rects[s];
	if (!rev) {
		for (int i = 0; i < nrects; i++) {
			if (rects[i].all[index] < rects[right].all[index]) {
				node_swap(node, s+i, s+left);
				left++;
			}
		}
	} else {
		for (int i = 0; i < nrects; i++) {
			if (rects[right].all[index] < rects[i].all[index]) {
				node_swap(node, s+i, s+left);
				left++;
			}
		}
	}
	node_swap(node, s+left, s+right);
	node_qsort(node, s, s+left, index, rev);
	node_qsort(node, s+left+1, e, index, rev);
}

// sort the node rectangles by the axis. used during splits
static void node_sort_by_axis(struct node *node, int axis, bool rev, bool max) {
	int by_index = max ? DIMS+axis : axis;
	node_qsort(node, 0, node->count, by_index, rev);
}

static void node_move_rect_at_index_into(struct node *from, int index,
	struct node *into)
{
	into->rects[into->count] = from->rects[index];
	from->rects[index] = from->rects[from->count-1];
	if (from->kind == LEAF) {
		into->items[into->count] = from->items[index];
		from->items[index] = from->items[from->count-1];
	} else {
		into->nodes[into->count] = from->nodes[index];
		from->nodes[index] = from->nodes[from->count-1];
	}
	from->count--;
	into->count++;
}

static bool node_split_largest_axis_edge_snap(struct rtree *tr,
	struct rect *rect, struct node *node, struct node **right_out)
{
	int axis = rect_largest_axis(rect);
	struct node *right = node_new(tr, node->kind);
	if (!right) {
		return false;
	}
	for (int i = 0; i < node->count; i++) {
		coord_t min_dist = node->rects[i].min[axis] - rect->min[axis];
		coord_t max_dist = rect->max[axis] - node->rects[i].max[axis];
		if (max_dist < min_dist) {
			// move to right
			node_move_rect_at_index_into(node, i, right);
			i--;
		}
	}
	// Make sure that both left and right nodes have at least
	// MINITEMS by moving items into underflowed nodes.
	if (node->count < MINITEMS) {
		// reverse sort by min axis
		node_sort_by_axis(right, axis, true, false);
		do {
			node_move_rect_at_index_into(right, right->count-1, node);
		} while (node->count < MINITEMS);
	} else if (right->count < MINITEMS) {
		// reverse sort by max axis
		node_sort_by_axis(node, axis, true, true);
		do {
			node_move_rect_at_index_into(node, node->count-1, right);
		} while (right->count < MINITEMS);
	}
	*right_out = right;
	return true;
}

static bool node_split(struct rtree *tr, struct rect *rect, struct node *node,
	struct node **right)
{
	return node_split_largest_axis_edge_snap(tr, rect, node, right);
}

static int node_choose_least_enlargement(const struct node *node,
	const struct rect *ir)
{
	int j = 0;
	coord_t jenlarge = INFINITY;
	for (int i = 0; i < node->count; i++) {
		// calculate the enlarged area
		coord_t uarea = rect_unioned_area(&node->rects[i], ir);
		coord_t area = rect_area(&node->rects[i]);
		coord_t enlarge = uarea - area;
		if (enlarge < jenlarge) {
			j = i;
			jenlarge = enlarge;
		}
	}
	return j;
}

static int node_choose(struct rtree *tr, const struct node *node,
	const struct rect *rect, int depth)
{
#ifdef USE_PATHHINT
	int h = tr->path_hint[depth];
	if (h < node->count) {
		if (rect_contains(&node->rects[h], rect)) {
			return h;
		}
	}
#endif
	// Take a quick look for the first node that contain the rect.
	for (int i = 0; i < node->count; i++) {
		if (rect_contains(&node->rects[i], rect)) {
#ifdef USE_PATHHINT
			tr->path_hint[depth] = i;
#endif
			return i;
		}
	}
	// Fallback to using che "choose least enlargment" algorithm.
	int i = node_choose_least_enlargement(node, rect);
#ifdef USE_PATHHINT
	tr->path_hint[depth] = i;
#endif
	return i;
}

static struct rect node_rect_calc(const struct node *node) {
	struct rect rect = node->rects[0];
	for (int i = 1; i < node->count; i++) {
		rect_expand(&rect, &node->rects[i]);
	}
	return rect;
}

// node_insert returns false if out of memory
static bool node_insert(struct rtree *tr, struct rect *nr, struct node *node,
	struct rect *ir, item_t item, int depth, bool *split)
{
	if (node->kind == LEAF) {
		if (node->count == MAXITEMS) {
			*split = true;
			return true;
		}
		int index = node->count;
		node->rects[index] = *ir;
		node->items[index] = item;
		node->count++;
		*split = false;
		return true;
	}
	// Choose a subtree for inserting the rectangle.
	int i = node_choose(tr, node, ir, depth);
	cow_node_or(node->nodes[i], return false);
	if (!node_insert(tr, &node->rects[i], node->nodes[i], ir, item, depth+1,
		split))
	{
		return false;
	}
	if (!*split) {
		rect_expand(&node->rects[i], ir);
		*split = false;
		return true;
	}
	// split the child node
	if (node->count == MAXITEMS) {
		*split = true;
		return true;
	}
	struct node *right;
	if (!node_split(tr, &node->rects[i], node->nodes[i], &right)) {
		return false;
	}
	node->rects[i] = node_rect_calc(node->nodes[i]);
	node->rects[node->count] = node_rect_calc(right);
	node->nodes[node->count] = right;
	node->count++;
	return node_insert(tr, nr, node, ir, item, depth, split);
}

struct rtree *rtree_new_with_allocator(void *(*_malloc)(size_t),
	void (*_free)(void*)
) {
	_malloc = _malloc ? _malloc : malloc;
	_free = _free ? _free : free;
	struct rtree *tr = (struct rtree *)_malloc(sizeof(struct rtree));
	if (!tr) return NULL;
	memset(tr, 0, sizeof(struct rtree));
	tr->malloc = _malloc;
	tr->free = _free;
	return tr;
}

struct rtree *rtree_new(void) {
	return rtree_new_with_allocator(NULL, NULL);
}

bool rtree_insert(struct rtree *tr, const coord_t *min,
	const coord_t *max, const item_t item)
{
	// copy input rect
	struct rect rect;
	memcpy(&rect.min[0], min, sizeof(coord_t)*DIMS);
	memcpy(&rect.max[0], max?max:min, sizeof(coord_t)*DIMS);

	while (1) {
		if (!tr->root) {
			struct node *new_root = node_new(tr, LEAF);
			if (!new_root) {
				break;
			}
			tr->root = new_root;
			tr->rect = rect;
			tr->height = 1;
		}
		bool split = false;
		cow_node_or(tr->root, break);
		if (!node_insert(tr, &tr->rect, tr->root, &rect, item, 0, &split)) {
			break;
		}
		if (!split) {
			rect_expand(&tr->rect, &rect);
			tr->count++;
			return true;
		}
		struct node *new_root = node_new(tr, BRANCH);
		if (!new_root) {
			break;
		}
		struct node *right;
		if (!node_split(tr, &tr->rect, tr->root, &right)) {
			tr->free(new_root);
			break;
		}
		new_root->rects[0] = node_rect_calc(tr->root);
		new_root->rects[1] = node_rect_calc(right);
		new_root->nodes[0] = tr->root;
		new_root->nodes[1] = right;
		tr->root = new_root;
		tr->root->count = 2;
		tr->height++;
	}
	// out of memory
	return false;
}

void rtree_free(struct rtree *tr) {
	if (tr->root) {
		node_free(tr, tr->root);
	}
	if (tr->queue) {
		priority_queue_free(tr->queue);
	}
	tr->free(tr);
}

static bool node_search(struct node *node, struct rect *rect,
	bool (*iter)(const coord_t *min, const coord_t *max, const item_t item,
		void *udata),
	void *udata)
{
	if (node->kind == LEAF) {
		for (int i = 0; i < node->count; i++) {
			if (rect_intersects(&node->rects[i], rect)) {
				if (!iter(node->rects[i].min, node->rects[i].max,
					node->items[i], udata))
				{
					return false;
				}
			}
		}
		return true;
	}
	for (int i = 0; i < node->count; i++) {
		if (rect_intersects(&node->rects[i], rect)) {
			if (!node_search(node->nodes[i], rect, iter, udata)) {
				return false;
			}
		}
	}
	return true;
}

void rtree_search(const struct rtree *tr, const coord_t min[],
	const coord_t max[],
	bool (*iter)(const coord_t min[], const coord_t max[], const item_t item,
		void *udata),
	void *udata)
{
	// copy input rect
	struct rect rect;
	memcpy(&rect.min[0], min, sizeof(coord_t)*DIMS);
	memcpy(&rect.max[0], max?max:min, sizeof(coord_t)*DIMS);

	if (tr->root) {
		node_search(tr->root, &rect, iter, udata);
	}
}

coord_t distance_bb(const coord_t point[], struct rect *rect) {

	coord_t dist2 = 0;

	for (int i = 0; i < DIMS; i++) {
		if (point[i] < rect->min[i]) {
			dist2 += pow(rect->min[i] - point[i], 2);
		} else if (point[i] > rect->max[i]) {
			dist2 += pow(point[i] - rect->max[i], 2);
		}
		// else: coordinate is within min and max, does not contribute to
		// distance
	}

	return dist2;
}

bool rtree_nearest(struct rtree *tr, const coord_t point[],
	bool (*iter)(const item_t item, coord_t distance, void *udata),
	void *udata) {

	if (!tr->root)
		return true;

	if (!tr->queue)
		tr->queue = priority_queue_new();
	else
		tr->queue->size = 0;

	struct element root = { .distance = 0.0, .kind = tr->root->kind, .node = tr->root };
	if (!enqueue(tr->queue, root)) {
		return false;
	}

	while (tr->queue->size > 0) {

		struct element next_element = dequeue(tr->queue);

		if (next_element.kind == ITEM) {
			// We found an ITEM with an exact distance that is the next closest
			// to the query point.

			// Report the item and stop searching if the user function returns
			// false:
			bool keep_going = iter(next_element.item, next_element.distance, udata);
			if (!keep_going) {
				return true;
			}

		} else if (next_element.kind == ITEM_BY_BB) {
			// We found an ITEM_BY_BB in the queue, whose bounding box is next
			// closest to the query point.

			// Here, we can calculate a more accurate distance than the one
			// used in the queue (e.g., distance to a line is poorly
			// approximated by distance to bounding box of line). If that
			// distance is larger than the next element on the queue, enqueue
			// the item again with kind ITEM and continue the while-loop.

#ifdef KNN_USE_EXACT_DISTANCE
			next_element.distance = distance(point, next_element.rect, next_element.item);
			if (next_element.distance > peek(tr->queue).distance) {
				next_element.kind = ITEM;
				if (!enqueue(tr->queue, next_element)) {
					return false;
				}
				continue;
			}
#endif

			// Report the item and stop searching if the user function returns
			// false:
			bool keep_going = iter(next_element.item, next_element.distance, udata);
			if (!keep_going) {
				return true;
			}

		} else if (next_element.kind == LEAF) {
			// We found a LEAF node in the queue, whose bounding box is next
			// closest to the query point. Add each item contained in that leaf
			// to the queue.

			struct node *leaf = next_element.node;
			for (int i = 0; i < leaf->count; i++) {

				struct element item_element = {
					.distance = distance_bb(point, &leaf->rects[i]),
					.kind = ITEM_BY_BB,
					.item = leaf->items[i],
					.rect = &leaf->rects[i]
				};
				if (!enqueue(tr->queue, item_element)) {
					return false;
				}
			}

		} else {  // next_element.kind == BRANCH
			// We found a BRANCH node in the queue, whose bounding box is next
			// closest to the query point. Add each child node (BRANCH or LEAF)
			// to the queue.

			struct node *branch = next_element.node;
			for (int i = 0; i < branch->count; i++) {

				struct element node_element = {
					.distance = distance_bb(point, &branch->rects[i]),
					.kind = branch->nodes[i]->kind,  // BRANCH or LEAF
					.node = branch->nodes[i]
				};
				if (!enqueue(tr->queue, node_element)) {
					return false;
				}
			}
		}
	}
	return true;
}

static bool node_scan(struct node *node,
	bool (*iter)(const coord_t *min, const coord_t *max, const item_t item,
		void *udata),
	void *udata)
{
	if (node->kind == LEAF) {
		for (int i = 0; i < node->count; i++) {
			if (!iter(node->rects[i].min, node->rects[i].max,
				node->items[i], udata))
			{
				return false;
			}
		}
		return true;
	}
	for (int i = 0; i < node->count; i++) {
		if (!node_scan(node->nodes[i], iter, udata)) {
			return false;
		}
	}
	return true;
}

void rtree_scan(const struct rtree *tr,
	bool (*iter)(const coord_t *min, const coord_t *max, const item_t item,
		void *udata),
	void *udata)
{
	if (tr->root) {
		node_scan(tr->root, iter, udata);
	}
}

size_t rtree_count(const struct rtree *tr) {
	return tr->count;
}

void rtree_bb(const struct rtree *tr, coord_t* min, coord_t* max) {
	memcpy(min, tr->rect.min, sizeof(coord_t)*DIMS);
	memcpy(max, tr->rect.max, sizeof(coord_t)*DIMS);
}

static bool node_delete(struct rtree *tr, struct rect *nr, struct node *node,
	struct rect *ir, item_t item, int depth, bool *removed, bool *shrunk,
	int (*compare)(const item_t a, const item_t b, void *udata),
	void *udata)
{
	*removed = false;
	*shrunk = false;
	if (node->kind == LEAF) {
		for (int i = 0; i < node->count; i++) {
			if (!rect_equals_bin(ir, &node->rects[i])) {
				// different bounding box, keep going
				continue;
			}
			if (!equal(node->items[i], item)) {
				// different content, keep going
				continue;
			}
			// Found the target item to delete.
			node->rects[i] = node->rects[node->count-1];
			node->items[i] = node->items[node->count-1];
			node->count--;
			if (rect_onedge(ir, nr)) {
				// The item rect was on the edge of the node rect.
				// We need to recalculate the node rect.
				*nr = node_rect_calc(node);
				// Notify the caller that we shrunk the rect.
				*shrunk = true;
			}
			*removed = true;
			return true;
		}
		return true;
	}
	int h = 0;
#ifdef USE_PATHHINT
	h = tr->path_hint[depth];
	if (h < node->count) {
		if (rect_contains(&node->rects[h], ir)) {
			cow_node_or(node->nodes[h], return false);
			if (!node_delete(tr, &node->rects[h], node->nodes[h], ir, item,
				depth+1,removed, shrunk, compare, udata))
			{
				return false;
			}
			if (*removed) {
				goto removed;
			}
		}
	}
	h = 0;
#endif
	for (; h < node->count; h++) {
		if (!rect_contains(&node->rects[h], ir)) {
			continue;
		}
		struct rect crect = node->rects[h];
		cow_node_or(node->nodes[h], return false);
		if (!node_delete(tr, &node->rects[h], node->nodes[h], ir, item, depth+1,
			removed, shrunk, compare, udata))
		{
			return false;
		}
		if (!*removed) {
			continue;
		}
	removed:
		if (node->nodes[h]->count == 0) {
			// underflow
			node_free(tr, node->nodes[h]);
			node->rects[h] = node->rects[node->count-1];
			node->nodes[h] = node->nodes[node->count-1];
			node->count--;
			*nr = node_rect_calc(node);
			*shrunk = true;
			return true;
		}
#ifdef USE_PATHHINT
		tr->path_hint[depth] = h;
#endif
		if (*shrunk) {
			*shrunk = !rect_equals(&node->rects[h], &crect);
			if (*shrunk) {
				*nr = node_rect_calc(node);
			}
		}
		return true;
	}
	return true;
}

// returns false if out of memory
static int rtree_delete0(struct rtree *tr, const coord_t *min,
	const coord_t *max, const item_t item,
	int (*compare)(const item_t a, const item_t b, void *udata),
	void *udata)
{
	// copy input rect
	struct rect rect;
	memcpy(&rect.min[0], min, sizeof(coord_t)*DIMS);
	memcpy(&rect.max[0], max?max:min, sizeof(coord_t)*DIMS);

	if (!tr->root) {
		return 0;
	}
	bool removed = false;
	bool shrunk = false;
	cow_node_or(tr->root, return false);
	if (!node_delete(tr, &tr->rect, tr->root, &rect, item, 0, &removed, &shrunk,
		compare, udata))
	{
		return -1; // OOM
	}
	if (!removed) {
		return 0;
	}
	tr->count--;
	if (tr->count == 0) {
		node_free(tr, tr->root);
		tr->root = NULL;
		memset(&tr->rect, 0, sizeof(struct rect));
		tr->height = 0;
	} else {
		while (tr->root->kind == BRANCH && tr->root->count == 1) {
			struct node *prev = tr->root;
			tr->root = tr->root->nodes[0];
			prev->count = 0;
			node_free(tr, prev);
			tr->height--;
		}
		if (shrunk) {
			tr->rect = node_rect_calc(tr->root);
		}
	}
	return 1;
}

int rtree_delete(struct rtree *tr, const coord_t *min, const coord_t *max,
	const item_t item)
{
	return rtree_delete0(tr, min, max, item, NULL, NULL);
}

int rtree_delete_with_comparator(struct rtree *tr, const coord_t *min,
	const coord_t *max, const item_t item,
	int (*compare)(const item_t a, const item_t b, void *udata),
	void *udata)
{
	return rtree_delete0(tr, min, max, item, compare, udata);
}

struct rtree *rtree_clone(struct rtree *tr) {
	if (!tr) return NULL;
	struct rtree *tr2 = tr->malloc(sizeof(struct rtree));
	if (!tr2) return NULL;
	memcpy(tr2, tr, sizeof(struct rtree));
	if (tr2->root) rc_fetch_add(&tr2->root->rc, 1);
	return tr2;
}

void rtree_opt_relaxed_atomics(struct rtree *tr) {
	tr->relaxed = true;
}

#ifdef TEST_PRIVATE_FUNCTIONS
#include "tests/priv_funcs.h"
#endif
