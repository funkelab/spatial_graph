from cython cimport view
from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdint cimport *
from libcpp.utility cimport pair
import numpy as np


cdef extern from *:
    """
    #include "impl/graphlite/graph_lite.h"

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

    def add_edge(self, NodeType u, NodeType v, EDGE_DATA_ARGS):

EDGE_DATA_ARRAY_POINTERS_SET
        self._graph.add_edge_with_prop(
            u, v,
            EdgeData(EDGE_DATA_ARRAY_POINTERS_NAMES)
        )

    def add_edges(self, NodeType[:, ::1] edges, EDGE_DATA_ARRAY_ARGS):

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

        if data:
            while it != end:
                yield deref(it), self._graph.node_prop(it)
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

        while node_it != node_end:
            view = self._graph.neighbors(node_it)
            u = deref(node_it)
            it = view.first
            end = view.second
            while it != end:
                v = deref(it).first
                if u < v:
                    if data:
                        yield (u, v), deref(it).second.prop()
                    else:
                        yield (u, v)
                inc(it)
            inc(node_it)

    def nodes_data(self, NodeType[::1] nodes):
        for node in nodes:
            yield node, self._graph.node_prop(node)

NODE_DATA_BY_NAME

NODES_DATA_BY_NAME

    def edge_data(self, NodeType u, NodeType v):
        return self._graph.edge_prop(u, v)

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

        if data:
            while it != end:
                yield deref(it).first, deref(it).second.prop()
                inc(it)
        else:
            while it != end:
                yield deref(it).first
                inc(it)

    def __len__(self):
        return self._graph.size()
