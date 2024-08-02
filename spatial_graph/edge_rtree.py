from .rtree import RTree


class EdgeRTree(RTree):
    # TODO: template for node dtype (not known here)
    # pyx_item_t_declaration = f"""
    # cdef struct item_t:
    # {node_dtype.to_pyxtype()} u
    # {node_dtype.to_pyxtype()} v
    # bool corner_mask[NUM_DIMS]
    # """

    # c_item_t_declaration = f"""
    # typedef struct item_t {{
    # {node_dtype.to_pyxtype()} u;
    # {node_dtype.to_pyxtype()} v;
    # bool corner_mask[NUM_DIMS];
    # }} item_t;
    # """

    c_function_implementations = """
inline item_t convert_pyx_to_c_item(pyx_item_t *pyx_item, coord_t *from, coord_t *to) {
    item_t item;
    item.u = (*pyx_item)[0];
    item.v = (*pyx_item)[1];
    for (int d = 0; d < NUM_DIMS; d++) {
        item.corner_mask[d] = (from[d] < to[d]);
    }
    return item;
}
inline void copy_c_to_pyx_item(const item_t c_item, pyx_item_t *pyx_item) {
    (*pyx_item)[0] = c_item.u;
    (*pyx_item)[1] = c_item.v;
}
        """
