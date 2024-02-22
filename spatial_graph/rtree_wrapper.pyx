from . cimport rtree_decl as impl

cdef class RTree:

    cdef impl.rtree* _rtree

    def __cinit__(self):
        self._rtree = impl.rtree_new()

    def insert_points(self, double[:, :] points):

        for i in range(points.shape[0]):
            impl.rtree_insert(
                self._rtree,
                &points[i, 0],
                NULL,
                &i)

    def __len__(self):

        return impl.rtree_count(self._rtree)
