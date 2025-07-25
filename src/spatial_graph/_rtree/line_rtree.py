import numpy as np

from .rtree import RTree


class LineRTree(RTree):
    c_item_t_declaration = """
typedef struct item_t {
    item_data_base_t u;
    item_data_base_t v;
    bool corner_mask[DIMS];
} item_t;
"""

    c_converter_functions = """
inline void item_to_item_data(
        const item_t& item,
        item_data_t *item_data) {

    (*item_data)[0] = item.u;
    (*item_data)[1] = item.v;
}
inline item_t item_data_to_item(
        item_data_base_t *item_data,
        coord_t *start,
        coord_t *end) {

    item_t item;
    item.u = item_data[0];
    item.v = item_data[1];
    for (unsigned int d = 0; d < DIMS; d++) {
        item.corner_mask[d] = (start[d] < end[d]);
        if (!item.corner_mask[d]) {
            // swap coordinates to create bounding box
            std::swap(start[d], end[d]);
        }
    }
    return item;
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

inline coord_t point_segment_dist2(
        const coord_t point[],
        const coord_t start[],
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
    alpha = std::min((coord_t)1, std::max((coord_t)0, alpha));

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
        const coord_t point[],
        const struct rect *rect,
        const struct item_t item) {

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
