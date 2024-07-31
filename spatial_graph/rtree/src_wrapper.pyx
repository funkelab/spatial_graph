from libc.stdint cimport *
import numpy as np


ctypedef int bool

cdef extern from *:
    """
    #define DIMS NUM_DIMS

    C_DECLARATIONS

    #include "src/rtree.h"
    #include "src/rtree.c"

    C_FUNCTION_IMPLEMENTATIONS
    """

    PYX_DECLARATIONS

    cdef struct rtree
    cdef rtree *rtree_new()
    cdef void rtree_free(rtree *tr)
    cdef bool rtree_insert(
        rtree *tr,
        const coord_t *min,
        const coord_t *max,
        const item_t item)
    cdef void rtree_search(
        const rtree *tr,
        const coord_t *min,
        const coord_t *max,
        bool (*iter)(
            const coord_t *min,
            const coord_t *max,
            const item_t item,
            void *udata),
        void *udata)
    cdef bool rtree_nearest(
        rtree *tr,
        const coord_t *point,
        bool (*iter)(
            const item_t item,
            coord_t distance,
            void *udata),
        void *udata)
    cdef bool rtree_delete(
        rtree *tr,
        const coord_t *min,
        const coord_t *max,
        const item_t item)
    cdef size_t rtree_count(const rtree *tr)

cdef bint count_iterator(
        const coord_t* bb_min,
        const coord_t* bb_max,
        const item_t item,
        void* udata
    ) noexcept:

    cdef size_t* count = <size_t*>udata
    count[0] = count[0] + 1
    return True


cdef bint search_iterator(
        const coord_t* bb_min,
        const coord_t* bb_max,
        const item_t item,
        void* udata
    ) noexcept:

    cdef search_results* results = <search_results*>udata
    copy_c_to_pyx_item(item, &results.items[results.size])
    results.size += 1
    return True

cdef struct search_results:
    size_t size
    pyx_item_t* items

cdef bint nearest_iterator(
        const item_t item,
        coord_t distance,
        void* udata
    ) noexcept:

    cdef nearest_results* results = <nearest_results*>udata
    copy_c_to_pyx_item(item, &results.items[results.size])
    results.size += 1
    return results.size < results.max_size

cdef struct nearest_results:
    size_t size
    size_t max_size
    pyx_item_t* items

# wrap a contiguous numpy buffer, this way we can pass it to a plain C function
cdef init_search_results_from_numpy(search_results* r, pyx_item_t[::1] items):
    r.size = 0
    r.items = &items[0]
cdef init_nearest_results_from_numpy(nearest_results* r, pyx_item_t[::1] items):
    r.size = 0
    r.max_size = len(items)
    r.items = &items[0]


cdef class RTree:

    cdef rtree* _rtree

    def __cinit__(self):
        self._rtree = rtree_new()

    def __dealloc__(self):
        rtree_free(self._rtree)

    def insert_point_item(self, pyx_item_t item, coord_t[::1] point):

        rtree_insert(
            self._rtree,
            &point[0],
            NULL,
            convert_pyx_to_c_item(item, &point[0], NULL))

    def insert_point_items(self, pyx_item_t[::1] items, coord_t[:, ::1] points):

        for i in range(len(items)):
            rtree_insert(
                self._rtree,
                &points[i, 0],
                NULL,
                convert_pyx_to_c_item(items[i], &points[i, 0], NULL))

    def insert_bb_item(self, pyx_item_t item, coord_t[::1] bb_min, coord_t[::1] bb_max):

        rtree_insert(
            self._rtree,
            &bb_min[0],
            &bb_max[0],
            convert_pyx_to_c_item(item, &bb_min[0], &bb_max[0]))

    def insert_bb_items(self, pyx_item_t[::1] items, coord_t[:, ::1] bb_mins, coord_t[:, ::1] bb_maxs):

        for i in range(len(items)):
            rtree_insert(
                self._rtree,
                &bb_mins[i, 0],
                &bb_maxs[i, 0],
                convert_pyx_to_c_item(items[i], &bb_mins[i, 0], &bb_maxs[i, 0]))

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

        items = np.zeros((num_results,), dtype="BASE_ITEM_TYPE")
        if num_results == 0:
            return items
        init_search_results_from_numpy(&results, items)

        rtree_search(
            self._rtree,
            &bb_min[0],
            &bb_max[0],
            &search_iterator,
            &results)

        return items

    def nearest(self, coord_t[::1] point, size_t k):

        cdef nearest_results results

        items = np.zeros((k,), dtype="BASE_ITEM_TYPE")
        if k == 0:
            return items
        init_nearest_results_from_numpy(&results, items)

        all_good = rtree_nearest(
            self._rtree,
            &point[0],
            &nearest_iterator,
            &results)

        if not all_good:
            raise RuntimeError("RTree nearest neighbor search ran out of memory.")

        return items[:results.size]

    def delete(
            self,
            coord_t[::1] bb_min,
            coord_t[::1] bb_max,
            pyx_item_t item):

        cdef coord_t* bb_min_p = &bb_min[0]
        cdef coord_t* bb_max_p = &bb_max[0] if bb_max is not None else NULL

        return rtree_delete(
            self._rtree,
            bb_min_p,
            bb_max_p,
            convert_pyx_to_c_item(item, &bb_min[0], &bb_max[0]))

    def __len__(self):

        return rtree_count(self._rtree)
