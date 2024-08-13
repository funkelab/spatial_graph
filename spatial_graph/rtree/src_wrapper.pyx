from libc.stdint cimport *
import numpy as np


ctypedef int bool

cdef extern from *:
    """
    typedef int bool;
    #define false 0
    #define true 1

    C_MACROS
    #define DIMS NUM_DIMS

    C_DECLARATIONS
    typedef pyx_item_t* pyx_items_t;

    #include "src/rtree.h"
    #include "src/rtree.c"

    C_FUNCTION_IMPLEMENTATIONS
    """

    PYX_DECLARATIONS
    ctypedef pyx_item_t* pyx_items_t

    # PYX <-> C converters
    cdef item_t convert_pyx_to_c_item(pyx_item_t *pyx_item, coord_t *min, coord_t* max)
    cdef void copy_c_to_pyx_item(const item_t c_item, pyx_item_t *pyx_item)

    # rtree API
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


cdef pyx_items_t memview_to_pyx_items_t(API_ITEMS_MEMVIEW_TYPE items):
    # implementation depends on dimension of item
    return <pyx_items_t>&items[0 ITEMS_EXTRA_DIMS_0]


cdef bint count_iterator(
        const coord_t* bb_min,
        const coord_t* bb_max,
        const item_t item,
        void* udata
    ) noexcept:

    cdef size_t* count = <size_t*>udata
    count[0] = count[0] + 1
    return True


cdef struct search_results:
    size_t size
    pyx_items_t items


cdef init_search_results_from_memview(search_results* r, API_ITEMS_MEMVIEW_TYPE items):
    r.size = 0
    r.items = memview_to_pyx_items_t(items)


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


cdef struct nearest_results:
    size_t size
    size_t max_size
    pyx_items_t items


cdef init_nearest_results_from_memview(nearest_results* r, API_ITEMS_MEMVIEW_TYPE items):
    r.size = 0
    r.max_size = len(items)
    r.items = memview_to_pyx_items_t(items)


cdef bint nearest_iterator(
        const item_t item,
        coord_t distance,
        void* udata
    ) noexcept:

    cdef nearest_results* results = <nearest_results*>udata
    copy_c_to_pyx_item(item, &results.items[results.size])
    results.size += 1
    return results.size < results.max_size


cdef class RTree:

    cdef rtree* _rtree

    def __cinit__(self):
        self._rtree = rtree_new()

    def __dealloc__(self):
        rtree_free(self._rtree)

    def insert_point_items(self, API_ITEMS_MEMVIEW_TYPE items, coord_t[:, ::1] points):

        cdef pyx_items_t pyx_items = memview_to_pyx_items_t(items)

        for i in range(len(items)):
            rtree_insert(
                self._rtree,
                &points[i, 0],
                NULL,
                convert_pyx_to_c_item(&pyx_items[i], &points[i, 0], NULL))

    def insert_bb_items(self, API_ITEMS_MEMVIEW_TYPE items, coord_t[:, ::1] bb_mins, coord_t[:, ::1] bb_maxs):

        cdef pyx_items_t pyx_items = memview_to_pyx_items_t(items)

        for i in range(len(items)):
            rtree_insert(
                self._rtree,
                &bb_mins[i, 0],
                &bb_maxs[i, 0],
                convert_pyx_to_c_item(&pyx_items[i], &bb_mins[i, 0], &bb_maxs[i, 0]))

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

        items = np.zeros((num_results, ITEM_LENGTH), dtype="NP_ITEM_DTYPE")
        if num_results == 0:
            return items
        init_search_results_from_memview(&results, items)

        rtree_search(
            self._rtree,
            &bb_min[0],
            &bb_max[0],
            &search_iterator,
            &results)

        return items

    def nearest(self, coord_t[::1] point, size_t k):

        cdef nearest_results results

        items = np.zeros((k, ITEM_LENGTH), dtype="NP_ITEM_DTYPE")
        if k == 0:
            return items
        init_nearest_results_from_memview(&results, items)

        all_good = rtree_nearest(
            self._rtree,
            &point[0],
            &nearest_iterator,
            &results)

        if not all_good:
            raise RuntimeError("RTree nearest neighbor search ran out of memory.")

        return items[:results.size]

    def delete_items(
            self,
            coord_t[::1] bb_min,
            coord_t[::1] bb_max,
            API_ITEMS_MEMVIEW_TYPE items):

        cdef coord_t* bb_min_p = &bb_min[0]
        cdef coord_t* bb_max_p = &bb_max[0] if bb_max is not None else NULL

        cdef pyx_items_t pyx_items = memview_to_pyx_items_t(items)

        for i in range(len(items)):
            if not rtree_delete(
                    self._rtree,
                    bb_min_p,
                    bb_max_p,
                    convert_pyx_to_c_item(&pyx_items[i], &bb_min[0], &bb_max[0])):
                return False

        return True

    def __len__(self):

        return rtree_count(self._rtree)
