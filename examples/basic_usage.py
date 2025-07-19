# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "spatial-graph",
# ]
# [tool.uv.sources]
# spatial-graph = { path = ".." }
# ///

import numpy as np

import spatial_graph as sg


def main():
    print("=== Spatial Graph Basic Usage Example ===\n")

    # 1. Graph creation
    print("1. Creating a 3D spatial graph...")
    graph = sg.SpatialGraph(
        ndims=3,
        node_dtype="uint64",
        node_attr_dtypes={"position": "double[3]"},
        edge_attr_dtypes={"score": "float32"},
        position_attr="position",
    )
    print(f"   Created graph with {graph.ndims} dimensions")
    print(f"   Node dtype: {graph.node_dtype}")
    print(f"   Directed: {graph.directed}")
    print()

    # 2. Adding nodes
    print("2. Adding nodes with positions...")
    nodes = np.array([1, 2, 3, 4, 5], dtype="uint64")
    positions = np.array(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3],
            [0.4, 0.4, 0.4],
            [0.5, 0.5, 0.5],
        ],
        dtype="double",
    )

    graph.add_nodes(nodes, position=positions)
    print(f"   Added {len(nodes)} nodes")
    print(f"   Node IDs: {nodes}")
    print(f"   Positions shape: {positions.shape}")
    print()

    # 3. Adding edges
    print("3. Adding edges with scores...")
    edges = np.array([[1, 2], [3, 4], [5, 1]], dtype="uint64")
    scores = np.array([0.2, 0.3, 0.4], dtype="float32")

    graph.add_edges(edges, score=scores)
    print(f"   Added {len(edges)} edges")
    print(f"   Edges: {edges}")
    print(f"   Scores: {scores}")
    print()

    # 4. Query nodes in ROI
    print("4. Querying nodes in ROI...")
    roi = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    nodes_in_roi = graph.query_nodes_in_roi(roi)
    print(f"   ROI: {roi}")
    print(f"   Nodes in ROI: {nodes_in_roi}")
    print()

    # 5. Query edges in ROI
    print("5. Querying edges in ROI...")
    edges_in_roi = graph.query_edges_in_roi(roi)
    print(f"   Edges in ROI: {edges_in_roi}")
    print()

    # 6. Query nearest nodes
    print("6. Querying nearest nodes...")
    query_point = np.array([0.3, 0.3, 0.3])
    nearest_nodes = graph.query_nearest_nodes(query_point, k=3)
    print(f"   Query point: {query_point}")
    print(f"   3 nearest nodes: {nearest_nodes}")
    print()

    # 7. Query nearest edges
    print("7. Querying nearest edges...")
    nearest_edges = graph.query_nearest_edges(query_point, k=2)
    print(f"   2 nearest edges: {nearest_edges}")
    print()

    # 8. Access node attributes
    print("8. Accessing node attributes...")
    if len(nodes_in_roi) > 0:
        node_positions = graph.node_attrs[nodes_in_roi].position
        print("   Positions of nodes in ROI:")
        for i, (node_id, pos) in enumerate(zip(nodes_in_roi, node_positions)):
            print(f"     Node {node_id}: {pos}")
    print()

    # 9. Access edge attributes
    print("9. Accessing edge attributes...")
    if len(edges_in_roi) > 0:
        edge_scores = graph.edge_attrs[edges_in_roi].score
        print("   Scores of edges in ROI:")
        for i, (edge_idx, score) in enumerate(zip(edges_in_roi, edge_scores)):
            print(f"     Edge index {edge_idx}: score = {score}")
    print()

    # 10. Graph statistics before removal
    print("10. Graph statistics before node removal...")
    print(f"    Total nodes: {len(graph.nodes)}")
    print(f"    Total edges: {len(graph.edges)}")
    print()

    # 11. Remove some nodes
    print("11. Removing nodes...")
    nodes_to_remove = nodes[:2]  # Remove first 2 nodes
    print(f"    Removing nodes: {nodes_to_remove}")
    graph.remove_nodes(nodes_to_remove)
    print(f"    Nodes after removal: {len(graph.nodes)}")
    print(f"    Edges after removal: {len(graph.edges)}")
    print()

    # 12. Final query to show updated graph
    print("12. Final query on updated graph...")
    remaining_nodes = graph.query_nodes_in_roi(
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    )
    print(f"    Remaining nodes: {remaining_nodes}")

    if len(remaining_nodes) > 0:
        remaining_positions = graph.node_attrs[remaining_nodes].position
        print("    Remaining node positions:")
        for node_id, pos in zip(remaining_nodes, remaining_positions):
            print(f"      Node {node_id}: {pos}")

    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
