from libc.stdint cimport *
import numpy as np


ctypedef int bool

cdef extern from *:
    """
    typedef int bool;
    #define false 0
    #define true 1

    %if $c_distance_function
    #define KNN_USE_EXACT_DISTANCE
    %end if
    #define DIMS $dims

    typedef $coord_dtype.to_pyxtype() coord_t;
    typedef $item_dtype.base_c_type item_base_t;
    %if $item_dtype.is_array
    typedef item_base_t pyx_item_t[$item_dtype.size];
    %else
    typedef item_base_t pyx_item_t;
    %end if
    typedef pyx_item_t* pyx_items_t;

    %if $c_item_t_declaration
    $c_item_t_declaration
    %else
    %if $item_dtype.is_array
    typedef struct item_t {
        item_base_t data[$item_dtype.size];
    } item_t;
    %else
    typedef item_base_t item_t;
    %end if
    %end if

    %if $c_equal_function
    $c_equal_function
    %else
    inline bool equal(const item_t a, const item_t b) {
    %if $item_dtype.is_array
        return memcmp(&a, &b, sizeof(item_t));
    %else
        return a == b;
    %end if
    }
    %end if

    #include "src/rtree.h"
    #include "src/rtree.c"

    %if $c_converter_functions
    $c_converter_functions
    %else
    %if $item_dtype.is_array
    inline item_t convert_pyx_to_c_item(pyx_item_t *pyx_item, coord_t *min, coord_t *max) {
        item_t c_item;
        memcpy(&c_item, *pyx_item, sizeof(item_t));
        return c_item;
    }
    inline void copy_c_to_pyx_item(const item_t c_item, pyx_item_t *pyx_item) {
        memcpy(pyx_item, &c_item, sizeof(item_t));
    }
    %else
    // default PYX<->C converters, just casting
    inline item_t convert_pyx_to_c_item(pyx_item_t *pyx_item, coord_t *min, coord_t *max) {
        return (item_t)*pyx_item;
    }
    inline void copy_c_to_pyx_item(const item_t c_item, pyx_item_t *pyx_item) {
        memcpy(pyx_item, &c_item, sizeof(item_t));
    }
    %end if
    %end if

    %if $c_distance_function
    $c_distance_function
    %end if
    """
    cdef enum:
        DIMS = $dims
    ctypedef $coord_dtype.to_pyxtype() coord_t
    ctypedef $item_dtype.base_c_type item_base_t
    %if $item_dtype.is_array
    ctypedef item_base_t pyx_item_t[$item_dtype.size]
    %else
    ctypedef item_base_t pyx_item_t
    %end if
    ctypedef pyx_item_t* pyx_items_t

    %if $pyx_item_t_declaration
    $pyx_item_t_declaration
    %else
    %if $item_dtype.is_array
    # item_t can't be an array in rtree, arrays can't be assigned to (and this
    # is needed inside rtree). So we make item_t a struct with field `data` to
    # hold the array.
    cdef struct item_t:
        item_base_t data[$item_dtype.size]
    %else
    ctypedef item_base_t item_t
    %end if
    %end if

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
    cdef int rtree_delete(
        rtree *tr,
        const coord_t *min,
        const coord_t *max,
        const item_t item)
    cdef size_t rtree_count(const rtree *tr)
    cdef void rtree_bb(const rtree *tr, coord_t *min, coord_t *max)


cdef pyx_items_t memview_to_pyx_items_t($item_dtype.to_pyxtype(add_dim=True) items):
    # implementation depends on dimension of item
    %if $item_dtype.is_array
    return <pyx_items_t>&items[0, 0]
    %else
    return <pyx_items_t>&items[0]
    %end if


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


cdef init_search_results_from_memview(search_results* r, $item_dtype.to_pyxtype(add_dim=True) items):
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
    coord_t *distances


cdef init_nearest_results_from_memview(nearest_results* r,
                                       $item_dtype.to_pyxtype(add_dim=True) items,
                                       coord_t[::1] distances):
    r.size = 0
    r.max_size = len(items)
    r.items = memview_to_pyx_items_t(items)
    r.distances = &distances[0] if distances is not None else NULL


cdef bint nearest_iterator(
        const item_t item,
        coord_t distance,
        void* udata
    ) noexcept:

    cdef nearest_results* results = <nearest_results*>udata
    copy_c_to_pyx_item(item, &results.items[results.size])
    if results.distances != NULL:
        results.distances[results.size] = distance
    results.size += 1
    return results.size < results.max_size


cdef class RTree:

    cdef rtree* _rtree

    def __cinit__(self):
        self._rtree = rtree_new()

    def __dealloc__(self):
        rtree_free(self._rtree)

    def insert_point_items(
            self,
            $item_dtype.to_pyxtype(add_dim=True) items,
            coord_t[:, ::1] points
    ):

        cdef pyx_items_t pyx_items = memview_to_pyx_items_t(items)

        for i in range(len(items)):
            rtree_insert(
                self._rtree,
                &points[i, 0],
                NULL,
                convert_pyx_to_c_item(&pyx_items[i], &points[i, 0], NULL))

    def insert_bb_items(
            self,
            $item_dtype.to_pyxtype(add_dim=True) items,
            coord_t[:, ::1] bb_mins,
            coord_t[:, ::1] bb_maxs
    ):

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

    def bounding_box(self):
        bb_min = np.empty(($dims,), dtype="$coord_dtype.base")
        bb_max = np.empty(($dims,), dtype="$coord_dtype.base")
        cdef coord_t[::1] _bb_min = bb_min
        cdef coord_t[::1] _bb_max = bb_max
        rtree_bb(self._rtree, &_bb_min[0], &_bb_max[0])
        return (bb_min, bb_max)

    def search(self, coord_t[::1] bb_min, coord_t[::1] bb_max):

        cdef search_results results
        cdef size_t num_results = self.count(bb_min, bb_max)

        items = np.zeros((num_results, $item_dtype.size), dtype="$item_dtype.base")
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

    def nearest(self, coord_t[::1] point, size_t k, return_distances=False):

        cdef nearest_results results

        items = np.zeros((k, $item_dtype.size), dtype="$item_dtype.base")
        if return_distances:
            distances = np.zeros((k,), dtype="$coord_dtype.base")
        else:
            distances = None
        if k == 0:
            return items
        init_nearest_results_from_memview(&results, items, distances)

        all_good = rtree_nearest(
            self._rtree,
            &point[0],
            &nearest_iterator,
            &results)

        if not all_good:
            raise RuntimeError("RTree nearest neighbor search ran out of memory.")

        if return_distances:
            return items[:results.size], distances[:results.size]
        else:
            return items[:results.size]

    def delete_items(
            self,
            $item_dtype.to_pyxtype(add_dim=True) items,
            coord_t[:, ::1] bb_mins,
            coord_t[:, ::1] bb_maxs=None
        ):

        if bb_maxs is None:
            bb_maxs = bb_mins

        cdef pyx_items_t pyx_items = memview_to_pyx_items_t(items)

        total_deleted = 0
        for i in range(len(items)):
            num_deleted = rtree_delete(
                self._rtree,
                &bb_mins[i, 0],
                &bb_maxs[i, 0],
                convert_pyx_to_c_item(&pyx_items[i], &bb_mins[i, 0], &bb_maxs[i, 0]))
            if num_deleted == -1:
                raise RuntimeError("RTree delete ran out of memory.")
            # if num_deleted == 0:
                # print(f"Item {pyx_items[i]} not deleted!")
            total_deleted += num_deleted

        return total_deleted

    def __len__(self):

        return rtree_count(self._rtree)
