from spatial_graph import PointRTree, LineRTree
import numpy as np


def test_search():
    rtree = PointRTree("uint64", "double", 2)
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
    rtree = PointRTree("uint64", "double", 2)
    for i in range(100):
        rtree.insert_point_item(i, np.array([i, i], dtype="float64"))

    # for i in range(10):
    # assert rtree.delete(np.array([-100.0, -100.0]), np.array([100.0, 100.0]), i)
    for i in range(10):
        assert rtree.delete(
            np.array([i, i], dtype="float64"), np.array([i, i], dtype="float64"), i
        )

    assert rtree.count(np.array([-100.0, -100.0]), np.array([100.0, 100.0])) == 90
    points = rtree.search(np.array([-100.0, -100.0]), np.array([100.0, 100.0]))
    assert len(points) == 90
    assert sorted(points) == sorted(range(10, 100))


def test_nearest():
    rtree = PointRTree("uint64", "double", 2)
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
    rtree = PointRTree("uint64", "double", 2)
    points = rtree.nearest(np.array([0.0, 0.0]), k=3)
    assert len(points) == 0

    # ask a very big tree
    rtree.insert_point_items(
        np.arange(10_000_000, dtype="uint64"),
        np.random.random((10_000_000, 2)).astype("double"),
    )
    points = rtree.nearest(np.array([0.5, 0.5]), k=100_000)
    assert len(points) == 100_000


def test_array_item():
    rtree = PointRTree("uint64[3]", "double", 2)
    for i in range(100):
        rtree.insert_point_item(
            np.array([i, i * 2, i * 3], dtype="uint64"),
            np.array([i, i], dtype="float64"),
        )


def test_line_rtree():
    line_rtree = LineRTree("uint64[2]", "double", 2)

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
    line_rtree = LineRTree("uint64[2]", "double", 2)

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

    line_rtree = LineRTree("uint64[2]", "double", 2)

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
