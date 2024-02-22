cdef extern from 'impl/rtree.h':
    rtree* rtree_new()
    bint rtree_insert(rtree* tr, const double* min, const double* max, const void* data)
    size_t rtree_count(const rtree* tr)
    bint rtree_delete(rtree* tr, const double* min, const double* max, const void* data)

cdef extern from 'impl/rtree.c':
    cdef struct rtree:
        size_t count
