from cython cimport view
from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdint cimport *
from libcpp.utility cimport pair
import numpy as np


cdef extern from *:
    """
    #include "src/graph_lite.h"

    // partial template instantiation of graph_lite::Graph as
    // UndirectedGraphTmpl
    template<typename NodeType, typename NodeData, typename EdgeData>
    class UndirectedGraphTmpl : public graph_lite::Graph<
        NodeType,
        NodeData,
        EdgeData,
        graph_lite::EdgeDirection::UNDIRECTED,
        graph_lite::MultiEdge::DISALLOWED,
        graph_lite::SelfLoop::DISALLOWED,
        graph_lite::Map::UNORDERED_MAP,
        graph_lite::Container::VEC
    > {};
    """

    cdef cppclass UndirectedGraphTmpl[NodeType, NodeData, EdgeData]:

        cppclass Iterator:
            NodeType operator*()
            Iterator operator++()
            bint operator==(Iterator)
            bint operator!=(Iterator)

        cppclass EdgePropIterWrap[ED]:
            ED& prop()

        cppclass NeighborsIterator:
            pair[NodeType, EdgePropIterWrap[EdgeData]] operator*()
            NeighborsIterator operator++()
            bint operator==(NeighborsIterator)
            bint operator!=(NeighborsIterator)

        int add_node_with_prop(NodeType& node, NodeData& prop)

        int add_edge_with_prop(NodeType& source, NodeType& target, EdgeData& prop)

        NodeData& node_prop[T](T& node)

        EdgeData& edge_prop[T](T& u, T& v)

        pair[NeighborsIterator, NeighborsIterator] neighbors(Iterator& node)
        pair[NeighborsIterator, NeighborsIterator] neighbors(NodeType& node)

        int remove_nodes(NodeType& node)

        int count_neighbors(NodeType& node)

        size_t size() const

        size_t num_edges() const

        Iterator begin()

        Iterator end()


NODE_TYPE_DECLARATION


NODE_DATA_DECLARATION


EDGE_DATA_DECLARATION


ctypedef UndirectedGraphTmpl[NodeType, NodeData, EdgeData] UndirectedGraphType
ctypedef UndirectedGraphType.Iterator NodeIterator
ctypedef UndirectedGraphType.NeighborsIterator NeighborsIterator


cdef class UndirectedGraph:

    cdef UndirectedGraphType _graph

    def add_node(self, NodeType node, NODE_DATA_ARGS):

NODE_DATA_ARRAY_POINTERS_SET
        self._graph.add_node_with_prop(
            node,
            NodeData(NODE_DATA_ARRAY_POINTERS_NAMES)
        )

    def add_nodes(self, NodeType[::1] nodes, NODE_DATA_ARRAY_ARGS):

NODE_DATA_ARRAYS_POINTERS_DEF
        for i in range(len(nodes)):
NODE_DATA_ARRAYS_POINTERS_SET
            self._graph.add_node_with_prop(
                nodes[i],
                NodeData(NODE_DATA_ARRAYS_POINTERS_NAMES)
            )

    def add_edge(self, NodeType[::1] edge, EDGE_DATA_ARGS):

EDGE_DATA_ARRAY_POINTERS_SET
        self._graph.add_edge_with_prop(
            edge[0], edge[1],
            EdgeData(EDGE_DATA_ARRAY_POINTERS_NAMES)
        )

    def add_edges(self, NodeType[:, :] edges, EDGE_DATA_ARRAY_ARGS):

EDGE_DATA_ARRAYS_POINTERS_DEF
        for i in range(edges.shape[0]):
EDGE_DATA_ARRAYS_POINTERS_SET
            self._graph.add_edge_with_prop(
                edges[i, 0], edges[i, 1],
                EdgeData(EDGE_DATA_ARRAYS_POINTERS_NAMES)
            )

    def nodes(self, bint data=False):

        cdef NodeIterator it = self._graph.begin()
        cdef NodeIterator end = self._graph.end()
        cdef NodeDataView node_data = NodeDataView()

        if data:
            while it != end:
                node_data.set_ptr(&self._graph.node_prop(it))
                yield deref(it), node_data
                inc(it)
        else:
            while it != end:
                yield deref(it)
                inc(it)

    def edges(self, node=None, bint data=False):

        if node is not None:
            yield from self._neighbors(<NodeType>node, data)
            return

        # iterate over all edges by iterating over all nodes u and their
        # neighbors v with u < v
        cdef NodeIterator node_it = self._graph.begin()
        cdef NodeIterator node_end = self._graph.end()
        cdef pair[NeighborsIterator, NeighborsIterator] view
        cdef NodeType u, v
        cdef EdgeDataView edge_data = EdgeDataView()

        while node_it != node_end:
            view = self._graph.neighbors(node_it)
            u = deref(node_it)
            it = view.first
            end = view.second
            while it != end:
                v = deref(it).first
                if u < v:
                    if data:
                        edge_data.set_ptr(&deref(it).second.prop())
                        yield (u, v), edge_data
                    else:
                        yield (u, v)
                inc(it)
            inc(node_it)

    # same as above, but for fast access to edges incident to an array of nodes
    def edges_by_nodes(self, NodeType[::1] nodes):

        # iterate over all edges by iterating over all nodes u and their
        # neighbors v with u < v
        cdef pair[NeighborsIterator, NeighborsIterator] view
        cdef NodeType u, v
        cdef Py_ssize_t i = 0

        num_edges = self._num_edges(nodes)
        data = np.empty(shape=(num_edges, 2), dtype="NODE_NPTYPE")
        cdef NodeType[:, ::1] edges = data

        for u in nodes:
            view = self._graph.neighbors(u)
            it = view.first
            end = view.second
            while it != end:
                v = deref(it).first
                if u < v:
                    edges[i, 0] = u
                    edges[i, 1] = v
                    i += 1
                inc(it)

        return data[:i]

    def nodes_data(self, NodeType[::1] nodes):
        cdef NodeDataView node_data = NodeDataView()
        for node in nodes:
            node_data.set_ptr(&self._graph.node_prop(node))
            yield node, node_data

NODE_DATA_BY_NAME

NODES_DATA_BY_NAME

    def edges_data(self, NodeType[::1] us, NodeType[::1] vs):
        cdef EdgeDataView edge_data = EdgeDataView()
        num_edges = len(us)
        for i in range(num_edges):
            edge_data.set_ptr(&self._graph.edge_prop(us[i], vs[i]))
            yield edge_data

EDGE_DATA_BY_NAME

EDGES_DATA_BY_NAME

    def remove_node(self, NodeType node):
        self._graph.remove_nodes(node)

    def count_neighbors(self, NodeType[:] nodes):
        num_nodes = len(nodes)
        cdef int[:] counts = view.array(
            shape=(num_nodes,),
            itemsize=sizeof(int),
            format="i")
        for i in range(num_nodes):
            counts[i] = self._graph.count_neighbors(nodes[i])
        return counts

    def _neighbors(self, NodeType node, bint data):

        cdef pair[NeighborsIterator, NeighborsIterator] view = self._graph.neighbors(node)
        cdef NeighborsIterator it = view.first
        cdef NeighborsIterator end = view.second
        cdef EdgeDataView edge_data = EdgeDataView()

        if data:
            while it != end:
                edge_data.set_ptr(&deref(it).second.prop())
                yield deref(it).first, edge_data
                inc(it)
        else:
            while it != end:
                yield deref(it).first
                inc(it)

    def __len__(self):
        return self._graph.size()

    def num_edges(self):
        return self._graph.num_edges()

    def _num_edges(self, NodeType[::1] nodes):
        return np.sum(self.count_neighbors(nodes))
