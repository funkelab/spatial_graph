from . cimport rtree_decl as impl
from libc.stdio cimport printf

ctypedef double coord_t

cdef bint search_iterator(
    const coord_t* bb_min,
    const coord_t* bb_max,
    const impl.item_data_t item, void* udata) noexcept:

    print("Found one!")
    printf("ID: %llu, ", item)
    printf("pos: (%f, %f, %f)\n", bb_min[0], bb_min[1], bb_min[2])
    return True

cdef class RTree:

    cdef impl.rtree* _rtree

    def __cinit__(self):
        self._rtree = impl.rtree_new()

    def insert_points(self, coord_t[:, :] points):

        for i in range(points.shape[0]):
            impl.rtree_insert(
                self._rtree,
                &points[i, 0],
                NULL,
                <impl.item_data_t>i)

    def search(self, coord_t[:] bb_min, coord_t[:] bb_max):

        impl.rtree_search(
            self._rtree,
            &bb_min[0],
            &bb_max[0],
            &search_iterator,
            NULL)

    def __len__(self):

        return impl.rtree_count(self._rtree)
