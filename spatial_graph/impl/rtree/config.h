// rtree configuration header

#ifndef __RTREE_CONFIG_H
#define __RTREE_CONFIG_H

// the maximal number of items per node in the rtree
#define MAXITEMS 64

// workaround to avoid cython warnings, replaces stdbool
typedef int bool;
#define false 0
#define true 1

#endif // __RTREE_CONFIG_H
