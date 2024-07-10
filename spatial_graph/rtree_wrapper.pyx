from libc.stdint cimport *
import numpy as np


ctypedef NODE_TYPE item_data_t
ctypedef COORD_TYPE coord_t
ctypedef int bool

cdef extern from *:
    """
    #define DIMS NUM_DIMS
    typedef NODE_TYPE item_data_t;
    typedef COORD_TYPE coord_t;

    #include "impl/rtree/rtree.h"
    #include "impl/rtree/rtree.c"

    """
    cdef struct rtree
    cdef rtree *rtree_new()
    cdef bool rtree_insert(
        rtree *tr,
        const coord_t *min,
        const coord_t *max,
        const item_data_t data)
    cdef void rtree_search(
        const rtree *tr,
        const coord_t *min,
        const coord_t *max,
        bool (*iter)(
            const coord_t *min,
            const coord_t *max,
            const item_data_t data,
            void *udata),
        void *udata)
    cdef bool rtree_delete(
        rtree *tr,
        const coord_t *min,
        const coord_t *max,
        const item_data_t data)
    cdef size_t rtree_count(const rtree *tr)

cdef bint count_iterator(
        const coord_t* bb_min,
        const coord_t* bb_max,
        const item_data_t item,
        void* udata
    ) noexcept:

    cdef size_t* count = <size_t*>udata
    count[0] = count[0] + 1
    return True


cdef bint search_iterator(
        const coord_t* bb_min,
        const coord_t* bb_max,
        const item_data_t item,
        void* udata
    ) noexcept:

    cdef search_results* results = <search_results*>udata
    results.data[results.size] = item
    results.size += 1
    return True


cdef struct search_results:
    size_t size
    item_data_t* data


# wrap a contiguous numpy buffer, this way we can pass it to a plain C function
cdef init_search_results_from_numpy(search_results* r, item_data_t[::1] data):
    r.size = 0
    r.data = &data[0]


cdef class RTree:

    cdef rtree* _rtree

    def __cinit__(self):
        self._rtree = rtree_new()

    def insert_point(self, id, coord_t[::1] point):

        rtree_insert(
            self._rtree,
            &point[0],
            NULL,
            <item_data_t>id)

    def insert_points(self, unsigned long[::1] ids, coord_t[:, ::1] points):

        for i in range(points.shape[0]):
            rtree_insert(
                self._rtree,
                &points[i, 0],
                NULL,
                <item_data_t>ids[i])

    def count(self, coord_t[::1] bb_min, coord_t[::1] bb_max):

        cdef size_t num = 0
        rtree_search(
            self._rtree,
            &bb_min[0],
            &bb_max[0],
            &count_iterator,
            &num)

        return num

    def search(self, coord_t[::1] bb_min, coord_t[::1] bb_max):

        cdef search_results results
        cdef size_t num_results = self.count(bb_min, bb_max)

        data_array = np.zeros((num_results,), dtype="NODE_DTYPE")
        if num_results == 0:
            return data_array
        init_search_results_from_numpy(&results, data_array)

        rtree_search(
            self._rtree,
            &bb_min[0],
            &bb_max[0],
            &search_iterator,
            &results)

        return data_array

    def delete(
            self,
            coord_t[::1] bb_min,
            coord_t[::1] bb_max,
            item_data_t item):

        cdef coord_t* bb_min_p = &bb_min[0]
        cdef coord_t* bb_max_p = &bb_min[0] if bb_min is None else NULL

        return rtree_delete(
            self._rtree,
            bb_min_p,
            bb_max_p,
            item)

    def __len__(self):

        return rtree_count(self._rtree)
