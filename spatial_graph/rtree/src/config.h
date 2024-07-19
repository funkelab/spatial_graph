// rtree configuration header

#ifndef __RTREE_CONFIG_H
#define __RTREE_CONFIG_H

// the maximal number of items per node in the rtree
#define MAXITEMS 64

// the maximal number of items in the priority queue to find nearest neighbors
#define MAX_QUEUE_SIZE 1000000

// workaround to avoid cython warnings, replaces stdbool
typedef int bool;
#define false 0
#define true 1

#endif // __RTREE_CONFIG_H
