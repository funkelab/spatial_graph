import numpy as np

from .rtree import RTree


class LineRTree(RTree):
    pyx_item_t_declaration = """
    cdef struct item_t:
        item_base_t u
        item_base_t v
        bool corner_mask[DIMS]
"""

    c_item_t_declaration = """
typedef struct item_t {
    item_base_t u;
    item_base_t v;
    bool corner_mask[DIMS];
} item_t;
"""

    c_converter_functions = """
inline item_t convert_pyx_to_c_item(pyx_item_t *pyx_item,
                                    coord_t *start, coord_t *end) {
    item_t item;
    coord_t tmp;
    item.u = (*pyx_item)[0];
    item.v = (*pyx_item)[1];
    for (int d = 0; d < DIMS; d++) {
        item.corner_mask[d] = (start[d] < end[d]);
        if (!item.corner_mask[d]) {
            // swap coordinates to create bounding box
            tmp = start[d];
            start[d] = end[d];
            end[d] = tmp;
        }
    }
    return item;
}
inline void copy_c_to_pyx_item(const item_t c_item, pyx_item_t *pyx_item) {
    (*pyx_item)[0] = c_item.u;
    (*pyx_item)[1] = c_item.v;
}
"""

    c_equal_function = """
inline int equal(const item_t a, const item_t b) {
    return (a.u == b.u && a.v == b.v);
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

inline coord_t point_segment_dist2(const coord_t point[], const coord_t start[],
                                   const coord_t end[]) {

    coord_t a[DIMS];
    coord_t b[DIMS];
    coord_t alpha = 0;

    for (int d = 0; d < DIMS; d++) {

        // subtract "start" from "end" and "point" to get "a" and "b"
        a[d] = end[d] - start[d];
        b[d] = point[d] - start[d];

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

extern inline coord_t distance(
    const coord_t point[], const struct rect *rect, const struct item_t item) {
    coord_t start[DIMS];
    coord_t end[DIMS];
    for (int d = 0; d < DIMS; d++) {
        if (item.corner_mask[d]) {
            start[d] = rect->min[d];
            end[d] = rect->max[d];
        } else {
            start[d] = rect->max[d];
            end[d] = rect->min[d];
        }
    }
    return point_segment_dist2(point, start, end);
}
"""

    def insert_line(self, line, start, end):
        """Convenience function to insert a single line. To insert multiple
        lines in bulk, please use the faster `insert_lines`.

        Parameters
        ----------
        line : np.ndarray
            The line identifier (as passed as the `item_dtype` to the
            constructor).
        start : np.ndarray
            The coordinates of the start of the line.
        end : np.ndarray
            The coordinates of the end of the line.
        """
        lines = np.array([line], dtype=self.item_dtype.base)
        starts = start[np.newaxis]
        ends = end[np.newaxis]
        return self._ctree.insert_bb_items(lines, starts, ends)

    def insert_lines(self, lines, starts, ends):
        """Insert a list of lines.

        Parameters
        ----------
        lines : np.ndarray, shape `(n, [m])`:
            Array containing the line identifiers (as passed as the
            `item_dtype` to the constructor). If the identifiers are an
            array of size `m`, the expected shape is `(n, m)` where `n` is
            the number of lines, otherwise the shape is just `(n,)`.

        starts : np.ndarray, shape `(n, d)`:
            The coordinates of the start of each line.

        ends : np.ndarray, shape `(n, d)`:
            The coordinates of the end of each line.
        """
        # we just forward to bb insert, "start" and "end" will be used to compute
        # the bounding box in our custom converter above
        return self.insert_bb_items(lines, starts, ends)
