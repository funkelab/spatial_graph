import numpy as np

import spatial_graph as sg


def test_search():
    rtree = sg.PointRTree("uint64", "double", 2)
    for i in range(100):
        rtree.insert_point_item(i, np.array([i, i], dtype="float64"))

    assert rtree.count(np.array([0.5, 0.5]), np.array([50.0, 50.0])) == 50
    points = rtree.search(np.array([0.5, 0.5]), np.array([50.0, 50.0]))
    assert len(points) == 50
    assert sorted(points) == sorted(range(1, 51))

    assert rtree.count(np.array([-100.0, -100.0]), np.array([100.0, 100.0])) == 100
    points = rtree.search(np.array([-100.0, -100.0]), np.array([100.0, 100.0]))
    assert len(points) == 100
    assert sorted(points) == sorted(range(100))


def test_delete():
    rtree = sg.PointRTree("uint64", "double", 2)
    for i in range(100):
        rtree.insert_point_item(i, np.array([i, i], dtype="float64"))

    for i in range(10):
        rtree.delete_item(i, np.array([i, i], dtype="float64"))

    assert rtree.count(np.array([-100.0, -100.0]), np.array([100.0, 100.0])) == 90
    points = rtree.search(np.array([-100.0, -100.0]), np.array([100.0, 100.0]))
    assert len(points) == 90
    assert sorted(points) == sorted(range(10, 100))


def test_nearest():
    rtree = sg.PointRTree("uint64", "double", 2)
    for i in range(100):
        rtree.insert_point_item(i, np.array([i, i], dtype="float64"))

    points = rtree.nearest(np.array([0.0, 0.0]), k=3)
    assert list(points) == [0, 1, 2]

    points = rtree.nearest(np.array([4.1, 4.1]), k=3)
    assert list(points) == [4, 5, 3]

    # ask for more neighbors than nodes
    points = rtree.nearest(np.array([0.0, 0.0]), k=1000)
    assert len(points) == 100
    assert list(points) == list(range(100))

    # ask an empty tree
    rtree = sg.PointRTree("uint64", "double", 3)
    points = rtree.nearest(np.array([0.0, 0.0, 0.0]), k=3)
    assert len(points) == 0

    # ask a very big tree
    all_points = np.random.random((10_000_000, 3)).astype("double")
    rtree.insert_point_items(
        np.arange(10_000_000, dtype="uint64"),
        all_points,
    )
    points = rtree.nearest(np.array([0.5, 0.5]), k=100_000)
    assert len(points) == 100_000

    # ensure that we find the right item in a big tree
    for i in np.random.randint(0, 10_000_000, size=(1000,)):
        points = rtree.nearest(all_points[i], k=1)
        assert points[0] == i


def test_array_item():
    rtree = sg.PointRTree("uint64[3]", "double", 2)
    for i in range(100):
        rtree.insert_point_item(
            np.array([i, i * 2, i * 3], dtype="uint64"),
            np.array([i, i], dtype="float64"),
        )


def test_line_rtree():
    line_rtree = sg.LineRTree("uint64[2]", "double", 2)

    line_rtree.insert_lines(
        np.array(
            [
                [0, 1],
                [10, 11],
            ],
            dtype="uint64",
        ),
        np.array(
            [
                [1.0, 1.0],
                [10.0, 10.0],
            ],
            dtype="double",
        ),
        np.array(
            [
                [0.0, 0.0],
                [11.0, 11.0],
            ],
            dtype="double",
        ),
    )

    lines = line_rtree.nearest(np.array([0.5, 0.5]), k=1)
    assert len(lines) == 1
    assert lines[0, 0] == 0
    assert lines[0, 1] == 1


def test_line_rtree_nearest():
    line_rtree = sg.LineRTree("uint64[2]", "double", 2)

    line_rtree.insert_lines(
        np.array(
            [
                [0, 1],
                [2, 3],
            ],
            dtype="uint64",
        ),
        np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
            ],
            dtype="double",
        ),
        np.array(
            [
                [1.0, 1.0],
                [1.0, 0.0],
            ],
            dtype="double",
        ),
    )

    # pick the correct line of two with the same bounding box:

    lines = line_rtree.nearest(np.array([0.6, 0.6]), k=1)
    assert len(lines) == 1
    assert lines[0, 0] == 0
    assert lines[0, 1] == 1

    lines = line_rtree.nearest(np.array([0.4, 0.6]), k=1)
    assert len(lines) == 1
    assert lines[0, 0] == 2
    assert lines[0, 1] == 3

    # pick the correct line that is closer to point, even though the bb is not
    # the closest one

    line_rtree = sg.LineRTree("uint64[2]", "double", 2)

    line_rtree.insert_lines(
        np.array(
            [
                [0, 1],
                [2, 3],
            ],
            dtype="uint64",
        ),
        np.array(
            [
                [0.0, 0.0],
                [0.0, 100.0],
            ],
            dtype="double",
        ),
        np.array(
            [
                [1.0, 1.0],
                [100.0, 0.0],
            ],
            dtype="double",
        ),
    )

    lines = line_rtree.nearest(np.array([2.0, 2.0]), k=1)
    assert len(lines) == 1
    assert lines[0, 0] == 0
    assert lines[0, 1] == 1

    # check that the distances are correct
    lines, distances = line_rtree.nearest(
        np.array([2.0, 2.0]), k=1, return_distances=True
    )
    assert len(lines) == 1
    assert len(distances) == 1
    assert lines[0, 0] == 0
    assert lines[0, 1] == 1
    np.testing.assert_almost_equal(distances[0], 2.0)

    lines, distances = line_rtree.nearest(
        np.array([0.5, 0.5]), k=1, return_distances=True
    )
    np.testing.assert_almost_equal(distances[0], 0.0)

    lines, distances = line_rtree.nearest(
        np.array([2.0, 0.0]), k=1, return_distances=True
    )
    np.testing.assert_almost_equal(distances[0], 2.0)

    lines, distances = line_rtree.nearest(
        np.array([1.0, 0.0]), k=1, return_distances=True
    )
    np.testing.assert_almost_equal(distances[0], 0.5)


def test_line_rtree_delete():
    line_rtree = sg.LineRTree("uint64[2]", "double", 2)

    line_rtree.insert_lines(
        np.array(
            [
                [0, 1],
                [2, 3],
            ],
            dtype="uint64",
        ),
        np.array(
            [
                [0.0, 0.0],
                [0.0, 100.0],
            ],
            dtype="double",
        ),
        np.array(
            [
                [1.0, 1.0],
                [100.0, 0.0],
            ],
            dtype="double",
        ),
    )

    # single item delete
    line_rtree.delete_item([0, 1], np.array([0.0, 0.0]), np.array([1.0, 1.0]))

    lines = line_rtree.nearest(np.array([0.5, 0.5]), k=1)
    assert len(lines) == 1
    assert lines[0, 0] == 2
    assert lines[0, 1] == 3

    line_rtree = sg.LineRTree("uint64[2]", "double", 2)

    np.random.seed(42)
    ids = np.random.randint(0, 1e10, size=(10_000, 2), dtype="uint64")
    starts = np.random.random((10_000, 2)).astype("double")
    ends = np.random.random((10_000, 2)).astype("double")

    line_rtree.insert_lines(ids, starts, ends)
    print(f"Line 100: {ids[100]}, {starts[100]}, {ends[100]}")

    assert line_rtree.count(np.array([0.0, 0.0]), np.array([1.0, 1.0])) == 10_000

    deleted = line_rtree.delete_items(ids[:1000], starts[:1000], ends[:1000])
    assert deleted == 1000

    assert line_rtree.count(np.array([0.0, 0.0]), np.array([1.0, 1.0])) == 9_000
