// rtree configuration header

#ifndef __RTREE_CONFIG_H
#define __RTREE_CONFIG_H

// number of spatial dimensions
#define DIMS 3

// user-defined data to store for each item (this is also used to identify items 
// for deletion)
typedef unsigned long long item_data_t;

// type of spatial coordinates
typedef double coord_t;

// the maximal number of items per node in the rtree
#define MAXITEMS 64

// workaround to avoid cython warnings, replaces stdbool
typedef int bool;
#define false 0
#define true 1

#endif // __RTREE_CONFIG_H
