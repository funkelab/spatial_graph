cdef extern from 'impl/config.h':

    ctypedef unsigned long long item_data_t
    ctypedef double coord_t

cdef extern from 'impl/rtree.h':


    rtree* rtree_new()

    bint rtree_insert(rtree* tr, const coord_t* min, const coord_t* max, const item_data_t data)

    size_t rtree_count(const rtree* tr)

    void rtree_search(const rtree* tr, const coord_t* min, const coord_t* max,
        bint (*iter)(const coord_t* min, const coord_t* max, const item_data_t data, void* udata),
        void *udata)

    bint rtree_delete(rtree* tr, const coord_t* min, const coord_t* max, const item_data_t data)

cdef extern from 'impl/rtree.c':

    cdef struct rtree:
        size_t count
