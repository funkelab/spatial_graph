// rtree configuration header

#ifndef __RTREE_CONFIG_H
#define __RTREE_CONFIG_H

// the maximal number of items per node in the rtree
#define MAXITEMS 64

// the initial number of items in the priority queue to find nearest neighbors
// (will keep doubling as needed to accommodate more)
#define INITIAL_QUEUE_SIZE 256

#endif // __RTREE_CONFIG_H
