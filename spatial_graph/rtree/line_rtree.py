from .rtree import RTree


class LineRTree(RTree):
    pyx_item_t_declaration = f"""
    cdef struct item_t:
        item_base_t u
        item_base_t v
        bool corner_mask[NUM_DIMS]
"""

    c_item_t_declaration = f"""
typedef struct item_t {{
    item_base_t u;
    item_base_t v;
    bool corner_mask[NUM_DIMS];
}} item_t;
"""

    c_converter_functions = """
inline item_t convert_pyx_to_c_item(pyx_item_t *pyx_item, coord_t *from, coord_t *to) {
    item_t item;
    coord_t tmp;
    item.u = (*pyx_item)[0];
    item.v = (*pyx_item)[1];
    for (int d = 0; d < NUM_DIMS; d++) {
        item.corner_mask[d] = (from[d] < to[d]);
        if (!item.corner_mask[d]) {
            // swap coordinates to create bounding box
            tmp = from[d];
            from[d] = to[d];
            to[d] = tmp;
        }
    }
    return item;
}
inline void copy_c_to_pyx_item(const item_t c_item, pyx_item_t *pyx_item) {
    (*pyx_item)[0] = c_item.u;
    (*pyx_item)[1] = c_item.v;
}
"""

    c_distance_function = """
inline coord_t length2(const coord_t x[]) {
    coord_t length2 = 0;
    for (int d = 0; d < DIMS; d++) {
        length2 += pow(x[d], 2);
    }
    return length2;
}

inline coord_t point_segment_dist2(const coord_t point[], const coord_t from[], const coord_t to[]) {

    coord_t a[DIMS];
    coord_t b[DIMS];
    coord_t alpha = 0;

    for (int d = 0; d < DIMS; d++) {

        // subtract "from" from "to" and "point" to get "a" and "b"
        a[d] = to[d] - from[d];
        b[d] = point[d] - from[d];

        // compute dot product "alpha" of "a" and "b"
        alpha += a[d] * b[d];
    }

    // normalize dot product
    alpha /= length2(a);

    // clip at 0 and 1 (beginning and end of line segment)
    alpha = min0(1, max0(0, alpha));

    for (int d = 0; d < DIMS; d++) {

        // multiply "a" by "alpha" to obtain closest segment point to "b"
        a[d] *= alpha;

        // subtract "b" from "a" to get offset
        a[d] -= b[d];
    }

    // compute squared length of offset
    return length2(a);
}

inline coord_t distance(const coord_t point[], const struct rect *rect, const struct item_t item) {
    coord_t from[DIMS];
    coord_t to[DIMS];
    for (int d = 0; d < DIMS; d++) {
        if (item.corner_mask[d]) {
            from[d] = rect->min[d];
            to[d] = rect->max[d];
        } else {
            from[d] = rect->max[d];
            to[d] = rect->min[d];
        }
    }
    return point_segment_dist2(point, from, to);
}
"""

    def insert_lines(self, lines, froms, tos):
        """Insert a list of lines.

        Args:

            lines (`ndarray`, shape `(n, [m])`:

                Array containing the line identifiers (as passed as the
                `item_dtype` to the constructor). If the identifiers are an
                array of size `m`, the expected shape is `(n, m)` where `n` is
                the number of lines, otherwise the shape is just `(n,)`.

            froms (`ndarray`, shape `(n, d)`):

                The coordinates of the start of each line.

            tos (`ndarray`, shape `(n, d)`):

                The coordinates of the end of each line.
        """
        # we just forward to bb insert, "from" and "to" will be used to compute
        # the bounding box in our custom converter above
        return self.insert_bb_items(lines, froms, tos)
