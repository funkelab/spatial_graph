#include <array>
#include <vector>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

// number of spatial dimensions
#define DIMS $dims

/*********************
 * TYPE DECLARATIONS *
 *********************/

// the base type of coordinates
typedef $coord_dtype.base_c_type coord_t;

// the base type of item data
typedef $item_dtype.base_c_type item_data_base_t;

// the external representation of an item (referred to as item_data)
%if $item_dtype.is_array
	// for arrays
	typedef std::array<item_data_base_t, $item_dtype.size> item_data_t;
%else
	// for scalars
	typedef item_data_base_t item_data_t;
%end if

// the internal type of an item
%if $c_item_t_declaration
	// custom item_t declaration
	$c_item_t_declaration
%else
	// default item_t declaration
	%if $item_dtype.is_array
		// for arrays
		typedef struct item_t {
			item_data_base_t data[$item_dtype.size];
		} item_t;
	%else
		// for scalars
		typedef item_data_base_t item_t;
	%end if
%end if

// shape of arrays holding data for multiple items
%if $item_dtype.is_array
	using items_data_shape = nb::shape<-1, $item_dtype.size>;
%else
	using items_data_shape = nb::shape<-1>;
%end if

/*************
 * FUNCTIONS *
 *************/

/* CONVERTERS */

// item_t -> item_data_t and item_data_t -> item_t
%if $c_converter_functions
	// custom converter functions
	$c_converter_functions
%else
	%if $item_dtype.is_array
		// default converters for array item_t
		inline item_t item_data_to_item(
				item_data_base_t *item_data,
				coord_t *min,
				coord_t *max) {

			item_t item;
			memcpy(&item, item_data, sizeof(item_t));
			return item;
		}
		inline void item_to_item_data(
				const item_t& item,
				item_data_t *item_data) {

			memcpy(item_data, &item, sizeof(item_t));
		}
	%else
		// default converters for scalar item_t
		inline item_t item_data_to_item(
				item_data_base_t *item_data,
				coord_t *min,
				coord_t *max) {

			return (item_t)*item_data;
		}
		inline void item_to_item_data(
				const item_t& item,
				item_data_t *item_data) {

			*item_data = static_cast<item_data_base_t>(item);
		}
	%end if
%end if


/* COMPARISON */

%if $c_equal_function
	// custom comparison function
	$c_equal_function
%else
	// default comparison function
	inline bool equal(const item_t a, const item_t b) {
		%if $item_dtype.is_array
			// for arrays
			return memcmp(&a, &b, sizeof(item_t));
		%else
			// for scalars
			return a == b;
		%end if
	}
%end if

/*******************
 * RTREE C BACKEND *
 *******************/

/* DISTANCE */

%if $c_distance_function
	// if a custom distance function is used, use this instead of the default 
	// bounding box distance function for kNN search
	#define KNN_USE_EXACT_DISTANCE
%end if

extern "C"{

	#include "src/rtree.h"
	#include "src/rtree.c"

} // extern "C"

%if $c_distance_function
	// custom distance function for exact computation of distances to items
	$c_distance_function
%end if

/************
 * TYPEDEFS *
 ************/

using ItemsVec = std::vector<item_data_t>;
using DistancesVec = std::vector<coord_t>;

using Items = nb::ndarray<
	nb::numpy,
	item_data_base_t,
	items_data_shape,
	nb::c_contig
>;
using Distances = nb::ndarray<
	nb::numpy,
	coord_t,
	nb::shape<-1>,
	nb::c_contig
>;

using Point = nb::ndarray<
	nb::numpy,
	coord_t,
	nb::shape<DIMS>,
	nb::c_contig
>;
using Points = nb::ndarray<
	nb::numpy,
	coord_t,
	nb::shape<-1, DIMS>,
	nb::c_contig
>;

/***************
 * RTree CLASS *
 ***************/

// Keep a unique C++ RTree typename for each module, so that multiple RTrees can 
// coexist. The name in the resulting python module will still be just RTree.
#define RTree RTree_WITTY_MODULE_HASH

class RTree {

private:

	// kNN search results
	struct Results {
		ItemsVec items;
		DistancesVec distances;
		size_t k;
		bool return_distances;
	};

	rtree* _rtree;

public:

	RTree() {
		_rtree = rtree_new();
	}

	~RTree() {
		rtree_free(_rtree);
	}

	void insert_point_items(
		Items items,
		Points points) {

		for (size_t i = 0; i < items.shape(0); i++) {
			rtree_insert(
				_rtree,
				&points(i, 0),
				NULL,
				%if $item_dtype.is_array
					item_data_to_item(&items(i, 0), &points(i, 0), NULL)
				%else
					item_data_to_item(&items(i), &points(i, 0), NULL)
				%end if
			);
		}
	}

	void insert_bb_items(
		Items items,
		Points bb_mins,
		Points bb_maxs) {

		for (size_t i = 0; i < items.shape(0); i++) {
			rtree_insert(
				_rtree,
				&bb_mins(i, 0),
				&bb_maxs(i, 0),
				%if $item_dtype.is_array
					item_data_to_item(&items(i, 0), &bb_mins(i, 0), &bb_maxs(i, 0))
				%else
					item_data_to_item(&items(i), &bb_mins(i, 0), &bb_maxs(i, 0))
				%end if
			);
		}
	}

	size_t count(
			Point bb_min,
			Point bb_max) {

		auto count_iterator = [](
			const coord_t* bb_min,
			const coord_t* bb_max,
			const item_t item,
			void* udata) {

			size_t* count = (size_t*)udata;
			*count += 1;
			return true;
		};

		size_t num = 0;
		rtree_search(
			_rtree,
			bb_min.data(),
			bb_max.data(),
			count_iterator,
			&num);

		return num;
	}

	nb::tuple bounding_box() {

		coord_t* bb_min = new coord_t[DIMS];
		coord_t* bb_max = new coord_t[DIMS];
		rtree_bb(_rtree, bb_min, bb_max);

		nb::capsule bb_min_owner(bb_min, [](void* p) noexcept {
			delete[] (coord_t*)p;
		});
		nb::capsule bb_max_owner(bb_max, [](void* p) noexcept {
			delete[] (coord_t*)p;
		});

		return nb::make_tuple(
			nb::ndarray<nb::numpy, coord_t, nb::shape<DIMS>>(
				bb_min,
				{ DIMS },
				bb_min_owner),
			nb::ndarray<nb::numpy, coord_t, nb::shape<DIMS>>(
				bb_max,
				{ DIMS },
				bb_max_owner)
		);
	}

	Items search(
			Point bb_min,
			Point bb_max) {

		ItemsVec* results = new ItemsVec();
		auto search_iterator = [](
				const coord_t *bb_min,
				const coord_t *bb_max,
				const item_t item,
				void* results) {

			item_data_t item_data;
			item_to_item_data(item, &item_data);
			static_cast<ItemsVec*>(results)->push_back(item_data);
			return true;
		};

		rtree_search(
			_rtree,
			bb_min.data(),
			bb_max.data(),
			search_iterator,
			results);

		nb::capsule results_owner(results, [](void* p) noexcept {
			delete (ItemsVec*)p;
		});

		%if $item_dtype.is_array
			return Items(results->data(), { results->size(), $item_dtype.size }, results_owner);
		%else
			return Items(results->data(), { results->size() }, results_owner);
		%end if
	}

	Results* find_nearest(
			Point point,
			size_t k,
			bool return_distances) {

		Results* results = new Results();
		results->k = k;
		results->return_distances = return_distances;
		auto nearest_iterator = [](
			const item_t item,
			coord_t distance,
			void* results) {
			Results* r = static_cast<Results*>(results);
			item_data_t item_data;
			item_to_item_data(item, &item_data);
			r->items.push_back(item_data);
			if (r->return_distances)
				r->distances.push_back(distance);
			return r->items.size() < r->k;
		};

		bool all_good = rtree_nearest(
			_rtree,
			point.data(),
			nearest_iterator,
			results);

		if (!all_good)
			throw std::bad_alloc();

		return results;
	}

	Items nearest(
			Point point,
			size_t k) {

		Results* results = find_nearest(point, k, false);

		nb::capsule results_owner(results, [](void* p) noexcept {
			delete (Results*)p;
		});

		%if $item_dtype.is_array
			Items items(
				results->items.data(),
				{ results->items.size(), $item_dtype.size },
				results_owner);
		%else
			Items items(
				results->items.data(),
				{ results->items.size() },
				results_owner);
		%end if

		return items;
	}

	nb::tuple nearest_with_distances(
			Point point,
			size_t k) {

		Results* results = find_nearest(point, k, true);

		nb::capsule results_owner(results, [](void* p) noexcept {
			delete (Results*)p;
		});

		%if $item_dtype.is_array
			Items items(
				results->items.data(),
				{ results->items.size(), $item_dtype.size },
				results_owner);
		%else
			Items items(
				results->items.data(),
				{ results->items.size() },
				results_owner);
		%end if

		Distances distances(
			results->distances.data(),
			{ results->distances.size() },
			results_owner);

		return nb::make_tuple(items, distances);

	}

	size_t delete_items(
		Items items,
		Points bb_mins,
		Points bb_maxs) {

		size_t total_deleted = 0;
		for (size_t i = 0; i < items.shape(0); i++) {
			size_t num_deleted = rtree_delete(
				_rtree,
				&bb_mins(i, 0),
				&bb_maxs(i, 0),
				%if $item_dtype.is_array
					item_data_to_item(&items(i, 0), &bb_mins(i, 0), &bb_maxs(i, 0))
				%else
					item_data_to_item(&items(i), &bb_mins(i, 0), &bb_maxs(i, 0))
				%end if
			);

			if (num_deleted < 0)
				throw std::bad_alloc();

			total_deleted += num_deleted;
		}

		return total_deleted;
	}

	size_t __len__() {

		return rtree_count(_rtree);
	}
};

/*************************
 * NANOBIND REGISTRATION *
 *************************/

NB_MODULE(rtree, m) {
	nb::class_<RTree>(m, "RTree")
		.def(nb::init<>())
		.def("insert_point_items", &RTree::insert_point_items)
		.def("insert_bb_items", &RTree::insert_bb_items)
		.def("delete_items", &RTree::delete_items)
		.def("bounding_box", &RTree::bounding_box)
		.def("count", &RTree::count)
		.def("search", &RTree::search)
		.def("nearest", &RTree::nearest)
		.def("nearest_with_distances", &RTree::nearest_with_distances)
		.def("__len__", &RTree::__len__)
	;
}
