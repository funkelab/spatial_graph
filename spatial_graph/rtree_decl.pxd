cdef extern from 'impl/rtree.h':
    rtree* rtree_new()
    bint rtree_insert(rtree* tr, const NUMTYPE* min, const NUMTYPE* max, const DATATYPE data)
    size_t rtree_count(const rtree* tr)
    bint rtree_delete(rtree* tr, const NUMTYPE* min, const NUMTYPE* max, const DATATYPE data)

cdef extern from 'impl/rtree.c':
    cdef struct rtree:
        size_t count
