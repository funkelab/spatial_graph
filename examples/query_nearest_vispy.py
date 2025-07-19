# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "pyopengl",
#     "pyqt6",
#     "vispy",
#     "spatial-graph",
# ]
# [tool.uv.sources]
# spatial-graph = { path = ".." }
# ///
from time import time

import numpy as np
from vispy import app, scene

from spatial_graph import SpatialGraph

canvas = scene.SceneCanvas(keys="interactive", size=(800, 600), show=True)
canvas.measure_fps()

view = canvas.central_widget.add_view()

camera = scene.cameras.FlyCamera(parent=view.scene, fov=60.0, name="Fly")
view.camera = camera

graph = SpatialGraph(
    ndims=3,
    node_dtype="uint64",
    node_attr_dtypes={"position": "double[3]"},
    edge_attr_dtypes={"score": "float32"},
    position_attr="position",
)
nodes = np.arange(100_000, dtype="uint64")
graph.add_nodes(nodes, position=np.random.random((100_000, 3)))

highlight_markers = scene.visuals.Markers(size=20.0, scaling="scene", spherical=True)
highlight_markers.parent = view.scene
node_markers = scene.visuals.Markers(
    pos=graph.node_attrs[nodes].position, size=10.0, scaling="scene", spherical=True
)
node_markers.parent = view.scene
axis_markers = scene.visuals.Markers(
    pos=np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    ),
    face_color=np.array(
        [
            [0.5, 0.5, 0.5],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    ),
    size=11,
    scaling="scene",
    spherical=True,
)
axis_markers.parent = view.scene

get_data_time = 0.0
set_data_time = 0.0
set_data_count = 0


@canvas.events.mouse_move.connect
def on_mouse_move(event):
    global get_data_time
    global set_data_time
    global set_data_count
    start = time()
    positions = graph.node_attrs[nodes].position
    get_data_time += time() - start
    start = time()
    node_markers.set_data(pos=positions, size=10)
    set_data_time += time() - start
    set_data_count += 1
    if set_data_count > 100:
        print(f"get_data(): {get_data_time / 100:.3f}s")
        print(f"set_data(): {set_data_time / 100:.3f}s")
        set_data_count = 0
        get_data_time = 0.0
        set_data_time = 0.0

    x_pos = event.pos[0]
    y_pos = event.pos[1]

    transform = node_markers.transforms.get_transform(map_from="scene", map_to="canvas")
    direction = transform.imap([x_pos, y_pos, 1.0])
    direction = -((direction / direction[3])[:3])
    direction /= np.linalg.norm(direction)
    camera_center = np.array(camera.center)
    z0_plane_intersection = (
        camera_center - (camera_center[2] / direction[2]) * direction
    )

    # get closest nodes to mouse position
    closest, distances = graph.query_nearest_nodes(
        z0_plane_intersection, k=10_000, return_distances=True
    )
    positions = graph.node_attrs[closest].position
    query = np.array([z0_plane_intersection])
    max_distance = distances.max()
    blend_coeffs = distances[:, np.newaxis] / max_distance
    colors = (
        np.tile([1.0, 0.5, 0.0], (len(positions), 1)) * (1.0 - blend_coeffs)
        + np.ones((len(positions), 3)) * blend_coeffs
    )
    highlight_markers.set_data(
        pos=np.concatenate((query, positions)),
        size=np.concatenate(([14], 10.0 + (1.0 - distances / max_distance) * 2)),
        face_color=np.concatenate(([[1.0, 0.5, 1.0]], colors)),
    )


if __name__ == "__main__":
    app.run()
