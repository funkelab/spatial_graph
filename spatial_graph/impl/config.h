// rtree configuration header

#ifndef __RTREE_CONFIG_H
#define __RTREE_CONFIG_H

// user-defined data to store for each item (this is also used to identify items 
// for deletion)
#define DATATYPE void *

// number of spatial dimensions
#define DIMS 3

// type of spatial coordinates
#define NUMTYPE double

// the maximal number of items per node in the rtree
#define MAXITEMS 64

typedef int bool;
#define false 0
#define true 1

#endif // __RTREE_CONFIG_H
