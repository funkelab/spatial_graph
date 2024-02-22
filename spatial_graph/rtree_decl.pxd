
cdef extern from 'impl/rtree.h':

    ctypedef double coord_t
    ctypedef void* item_t

    rtree* rtree_new()

    bint rtree_insert(rtree* tr, const coord_t* min, const coord_t* max, const item_t data)

    size_t rtree_count(const rtree* tr)

    void rtree_search(const rtree* tr, const coord_t* min, const coord_t* max,
        bint (*iter)(const coord_t* min, const coord_t* max, const item_t data, void* udata),
        void *udata)

    bint rtree_delete(rtree* tr, const coord_t* min, const coord_t* max, const item_t data)

cdef extern from 'impl/rtree.c':

    cdef struct rtree:
        size_t count
