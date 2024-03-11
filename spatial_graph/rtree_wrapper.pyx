from . cimport rtree_decl as impl
import numpy as np


ctypedef double coord_t


cdef bint count_iterator(
        const coord_t* bb_min,
        const coord_t* bb_max,
        const impl.item_data_t item,
        void* udata
    ) noexcept:

    cdef size_t* count = <size_t*>udata
    count[0] = count[0] + 1
    return True


cdef bint search_iterator(
        const coord_t* bb_min,
        const coord_t* bb_max,
        const impl.item_data_t item,
        void* udata
    ) noexcept:

    cdef search_results* results = <search_results*>udata
    results.data[results.size] = item
    results.size += 1
    return True


cdef struct search_results:
    size_t size
    impl.item_data_t* data


# wrap a contiguous numpy buffer, this way we can pass it to a plain C function
cdef init_search_results_from_numpy(search_results* r, impl.item_data_t[::1] data):
    r.size = 0
    r.data = &data[0]


cdef class RTree:

    cdef impl.rtree* _rtree

    def __cinit__(self):
        self._rtree = impl.rtree_new()

    def insert_point(self, id, coord_t[::1] point):

        impl.rtree_insert(
            self._rtree,
            &point[0],
            NULL,
            <impl.item_data_t>id)

    def insert_points(self, unsigned long[::1] ids, coord_t[:, ::1] points):

        for i in range(points.shape[0]):
            impl.rtree_insert(
                self._rtree,
                &points[i, 0],
                NULL,
                <impl.item_data_t>ids[i])

    def count(self, coord_t[::1] bb_min, coord_t[::1] bb_max):

        cdef size_t num = 0
        impl.rtree_search(
            self._rtree,
            &bb_min[0],
            &bb_max[0],
            &count_iterator,
            &num)

        return num

    def search(self, coord_t[::1] bb_min, coord_t[::1] bb_max):

        cdef search_results results
        cdef size_t num_results = self.count(bb_min, bb_max)
        # TODO: initialize with node dtype equivalent
        data_array = np.zeros((num_results,), dtype=np.uint64)
        init_search_results_from_numpy(&results, data_array)

        impl.rtree_search(
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
            impl.item_data_t item):

        cdef coord_t* bb_min_p = &bb_min[0]
        cdef coord_t* bb_max_p = &bb_min[0] if bb_min is None else NULL

        return impl.rtree_delete(
            self._rtree,
            bb_min_p,
            bb_max_p,
            item)

    def __len__(self):

        return impl.rtree_count(self._rtree)
