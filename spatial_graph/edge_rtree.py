from .rtree import RTree


class EdgeRTree(RTree):

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

    def insert_edges(self, edges, froms, tos):
        """Insert a list of edges.

        Args:

            edges (`ndarray`, shape `(n, 2)`):

                Array containing the edges as `(u, v)` rows, where `u` and `v`
                are the IDs of the nodes.

            froms (`ndarray`, shape `(n, d)`):

                The coordinates of the "from" node `u` of the cooresponding edge.

            tos (`ndarray`, shape `(n, d)`):

                The coordinates of the "to" node `v` of the cooresponding edge.
        """
        # we just forward to bb insert, "from" and "to" will be used to compute
        # the bounding box in our custom converter above
        return self.insert_bb_items(edges, froms, tos)
