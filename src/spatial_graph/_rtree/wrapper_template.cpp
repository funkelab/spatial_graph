#include <vector>
#include <nanobind/ndarray.h>

#define DIMS $dims
%if $c_distance_function
#define KNN_USE_EXACT_DISTANCE
%end if

namespace nb = nanobind;

typedef $coord_dtype.base_c_type coord_t;
typedef $item_dtype.base_c_type item_base_t;

extern "C"{


	%if $item_dtype.is_array
	typedef item_base_t pyx_item_t[$item_dtype.size];
	%else
	typedef item_base_t pyx_item_t;
	%end if
	typedef pyx_item_t* pyx_items_t;

	%if $c_item_t_declaration
	$c_item_t_declaration
	%else
	%if $item_dtype.is_array
	typedef struct item_t {
		item_base_t data[$item_dtype.size];
	} item_t;
	%else
	typedef item_base_t item_t;
	%end if
	%end if

	%if $c_equal_function
	$c_equal_function
	%else
	inline bool equal(const item_t a, const item_t b) {
	%if $item_dtype.is_array
		return memcmp(&a, &b, sizeof(item_t));
	%else
		return a == b;
	%end if
	}
	%end if

	#include "src/rtree.h"
	#include "src/rtree.c"

	%if $c_converter_functions
	$c_converter_functions
	%else
	%if $item_dtype.is_array
	inline item_t convert_pyx_to_c_item(pyx_item_t *pyx_item, coord_t *min, coord_t *max) {
		item_t c_item;
		memcpy(&c_item, *pyx_item, sizeof(item_t));
		return c_item;
	}
	inline void copy_c_to_pyx_item(const item_t c_item, pyx_item_t *pyx_item) {
		memcpy(pyx_item, &c_item, sizeof(item_t));
	}
	%else
	// default PYX<->C converters, just casting
	inline item_t convert_pyx_to_c_item(pyx_item_t *pyx_item, coord_t *min, coord_t *max) {
		return (item_t)*pyx_item;
	}
	inline void copy_c_to_pyx_item(const item_t c_item, pyx_item_t *pyx_item) {
		memcpy(pyx_item, &c_item, sizeof(item_t));
	}
	%end if
	%end if

	%if $c_distance_function
	$c_distance_function
	%end if

} // extern "C"

/************
 * TYPEDEFS *
 ************/

using ItemsArray = nb::ndarray<
	nb::numpy,
	item_base_t,
	nb::shape<-1>,
	nb::c_contig
>;
using ItemsArrayObject = nb::detail::ndarray_object<
	nb::numpy,
	item_base_t,
	nb::shape<-1>,
	nb::c_contig
>;

/**************
 * CONVERTERS *
 **************/

// default implementation for scalar item_t
inline item_t create_rtree_item(item_base_t *item, coord_t *min, coord_t *max) {
	return (item_t)*item;
}

class RTree {

public:

	RTree() {
		_rtree = rtree_new();
	}

	~RTree() {
		rtree_free(_rtree);
	}

	void insert_point_items(
		nb::ndarray<item_base_t, nb::shape<-1>, nb::c_contig> items,
		nb::ndarray<coord_t, nb::shape<-1, DIMS>, nb::c_contig> points) {

		for (size_t i = 0; i < items.size(); i++) {
			rtree_insert(
				_rtree,
				&points(i, 0),
				NULL,
				create_rtree_item(&items(i), &points(i, 0), NULL)
			);
		}
	}

	void insert_bb_items(
		nb::ndarray<item_base_t, nb::shape<-1>, nb::c_contig> items,
		nb::ndarray<coord_t, nb::shape<-1, DIMS>, nb::c_contig> bb_mins,
		nb::ndarray<coord_t, nb::shape<-1, DIMS>, nb::c_contig> bb_maxs) {

		for (size_t i = 0; i < items.size(); i++) {
			rtree_insert(
				_rtree,
				&bb_mins(i, 0),
				&bb_maxs(i, 0),
				create_rtree_item(&items(i), &bb_mins(i, 0), &bb_maxs(i, 0))
			);
		}
	}

	size_t count(
			nb::ndarray<coord_t, nb::shape<DIMS>, nb::c_contig> bb_min,
			nb::ndarray<coord_t, nb::shape<DIMS>, nb::c_contig> bb_max) {

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

	typedef typename std::vector<item_t> Items;
	typedef typename std::vector<coord_t> Distances;

	ItemsArrayObject search(
			nb::ndarray<coord_t, nb::shape<DIMS>, nb::c_contig> bb_min,
			nb::ndarray<coord_t, nb::shape<DIMS>, nb::c_contig> bb_max) {

		Items results;
		auto search_iterator = [](
			const coord_t *bb_min,
			const coord_t *bb_max,
			const item_t item,
			void* results) {
			static_cast<Items*>(results)->push_back(item);
			return true;
		};

		rtree_search(
			_rtree,
			bb_min.data(),
			bb_max.data(),
			search_iterator,
			&results);

		return ItemsArray(results.data(), { results.size() }).cast();
	}

	ItemsArrayObject nearest(
			nb::ndarray<coord_t, nb::shape<DIMS>, nb::c_contig> point,
			size_t k,
			bool return_distances) {

		struct Results {
			Items items;
			Distances distances;
			size_t k;
			bool return_distances;
		};
		Results results;
		results.k = k;
		results.return_distances = return_distances;
		auto nearest_iterator = [](
			const item_t item,
			coord_t distance,
			void* results) {
			Results* r = static_cast<Results*>(results);
			r->items.push_back(item);
			if (r->return_distances)
				r->distances.push_back(distance);
			return r->items.size() < r->k;
		};

		bool all_good = rtree_nearest(
			_rtree,
			point.data(),
			nearest_iterator,
			&results);

		// TODO
		//if not all_good:
			//raise RuntimeError("RTree nearest neighbor search ran out of memory.")

		if (return_distances) {
			// TODO
			//return items[:results.size], distances[:results.size]
		} else {
			return ItemsArray(results.items.data(), { results.items.size() }).cast();
		}
	}

	size_t delete_items(
		ItemsArray items,
		nb::ndarray<coord_t, nb::shape<-1, DIMS>, nb::c_contig> bb_mins,
		nb::ndarray<coord_t, nb::shape<-1, DIMS>, nb::c_contig> bb_maxs) {

		// TODO
		//if bb_maxs is None:
			//bb_maxs = bb_mins

		//cdef pyx_items_t pyx_items = memview_to_pyx_items_t(items)

		size_t total_deleted = 0;
		for (size_t i = 0; i < items.size(); i++) {
			size_t num_deleted = rtree_delete(
				_rtree,
				&bb_mins(i, 0),
				&bb_maxs(i, 0),
				create_rtree_item(&items(i), &bb_mins(i, 0), &bb_maxs(i, 0))
			);
			// TODO
			//if (num_deleted == -1)
				//raise RuntimeError("RTree delete ran out of memory.")
			total_deleted += num_deleted;
		}

		return total_deleted;
	}

	size_t __len__() {

		return rtree_count(_rtree);
	}

private:

	rtree* _rtree;
};

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
		.def("__len__", &RTree::__len__)
	;
}
