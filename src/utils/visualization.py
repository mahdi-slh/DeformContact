

import open3d as o3d
import numpy as np

def visualize_deformation_field(rest_graph, def_graph, contact_point, origin_point):
    # Extract node features (points) from the graph object
    rest_mesh_np = rest_graph.detach().numpy()
    def_mesh_np = def_graph.detach().numpy()
    contact_point_np = contact_point.detach().numpy()
    origin_point_np = origin_point.detach().numpy()

    # Calculate deformation magnitudes and find the point with the maximum deformation
    deformation_magnitudes = np.linalg.norm(def_mesh_np - rest_mesh_np, axis=1)
    max_deformation_index = np.argmax(deformation_magnitudes)
    max_deformation_point = rest_mesh_np[max_deformation_index]

    pcd_rest = o3d.geometry.PointCloud()
    pcd_def = o3d.geometry.PointCloud()
    pcd_rest.points = o3d.utility.Vector3dVector(rest_mesh_np)
    pcd_def.points = o3d.utility.Vector3dVector(def_mesh_np)

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(np.concatenate((rest_mesh_np, def_mesh_np)))
    n = len(rest_mesh_np)
    lineset.lines = o3d.utility.Vector2iVector([(i, i + n) for i in range(n)])

    # Create colliders as point clouds by converting the vertices of the triangle meshes to point clouds
    collider_contact_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.25)
    collider_contact_mesh.translate(contact_point_np)
    pcd_contact = o3d.geometry.PointCloud()
    pcd_contact.points = o3d.utility.Vector3dVector(np.array(collider_contact_mesh.vertices))
    pcd_contact.paint_uniform_color([0, 0, 1])  # Blue

    # Create the vector from the origin to the contact point
    vector_lineset = o3d.geometry.LineSet()
    vector_lineset.points = o3d.utility.Vector3dVector([origin_point_np, contact_point_np])
    vector_lineset.lines = o3d.utility.Vector2iVector([[0, 1]])
    vector_lineset.colors = o3d.utility.Vector3dVector([[0, 0, 0]])  # Black color for the vector, adjust as necessary

    pcd_rest.paint_uniform_color([0, 0.8, 0])
    pcd_def.paint_uniform_color([0.8, 0.8, 0])
    lineset.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5] for _ in range(n)])

    coor = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    print(contact_point_np, max_deformation_point)
    o3d.visualization.draw_geometries([pcd_rest, pcd_def, lineset, pcd_contact, coor, vector_lineset])



def visualize_merged_graphs(rest_graph, def_graph=None, collider_graph=None, pred_graph=None, translation=1.2):
    rest_points_np = rest_graph.pos.cpu().numpy()
    rest_edge_index_np = rest_graph.edge_index.t().cpu().numpy().astype(np.int32)

    rest_pcd = o3d.geometry.PointCloud()
    rest_pcd.points = o3d.utility.Vector3dVector(rest_points_np)
    rest_pcd.paint_uniform_color([0, 0.8, 0])  # Green for rest mesh

    rest_lines = o3d.geometry.LineSet()
    rest_lines.points = o3d.utility.Vector3dVector(rest_points_np)
    rest_lines.lines = o3d.utility.Vector2iVector(rest_edge_index_np)

    geometries = [rest_pcd, rest_lines]


    def_points_np = def_graph.pos.cpu().numpy() + [translation, 0, 0]
    def_edge_index_np = def_graph.edge_index.t().cpu().numpy().astype(np.int32)
    
    def_pcd = o3d.geometry.PointCloud()
    def_pcd.points = o3d.utility.Vector3dVector(def_points_np)
    def_pcd.paint_uniform_color([0.8, 0.8, 0])  # Yellow for deformed mesh
    
    def_lines = o3d.geometry.LineSet()
    def_lines.points = o3d.utility.Vector3dVector(def_points_np)
    def_lines.lines = o3d.utility.Vector2iVector(def_edge_index_np)
    
    geometries.extend([def_pcd, def_lines])


    collider_points_np = collider_graph.pos.cpu().numpy() + [translation, 0, 0]
    collider_edge_index_np = collider_graph.edge_index.t().cpu().numpy().astype(np.int32)
    
    collider_pcd = o3d.geometry.PointCloud()
    collider_pcd.points = o3d.utility.Vector3dVector(collider_points_np)
    collider_pcd.paint_uniform_color([0.5, 0.5, 0.8])  # Blue-ish for the collider mesh
    
    collider_lines = o3d.geometry.LineSet()
    collider_lines.points = o3d.utility.Vector3dVector(collider_points_np)
    collider_lines.lines = o3d.utility.Vector2iVector(collider_edge_index_np)
    
    geometries.extend([collider_pcd, collider_lines])

    if pred_graph:
        pred_points_np = pred_graph.pos.cpu().numpy() + [2*translation, 0, 0]
        pred_edge_index_np = pred_graph.edge_index.t().cpu().numpy().astype(np.int32)
        
        pred_pcd = o3d.geometry.PointCloud()
        pred_pcd.points = o3d.utility.Vector3dVector(pred_points_np)
        pred_pcd.paint_uniform_color([0.8, 0, 0])  # Red for predicted mesh
        
        pred_lines = o3d.geometry.LineSet()
        pred_lines.points = o3d.utility.Vector3dVector(pred_points_np)
        pred_lines.lines = o3d.utility.Vector2iVector(pred_edge_index_np)
        
        geometries.extend([pred_pcd, pred_lines])


        collider_points_np = collider_graph.clone().pos.cpu().numpy() + [2*translation, 0, 0]
        collider_edge_index_np = collider_graph.clone().edge_index.t().cpu().numpy().astype(np.int32)
        
        collider_pcd = o3d.geometry.PointCloud()
        collider_pcd.points = o3d.utility.Vector3dVector(collider_points_np)
        collider_pcd.paint_uniform_color([0.5, 0.5, 0.8])  # Blue-ish for the collider mesh
        
        collider_lines = o3d.geometry.LineSet()
        collider_lines.points = o3d.utility.Vector3dVector(collider_points_np)
        collider_lines.lines = o3d.utility.Vector2iVector(collider_edge_index_np)
        
        geometries.extend([collider_pcd, collider_lines])

    o3d.visualization.draw_geometries(geometries)

def visualize_meshes(mesh1, mesh2):
    """
    Visualize two meshes with different colors.

    Args:
    mesh1: First mesh (Open3D triangle mesh)
    mesh2: Second mesh (Open3D triangle mesh)
    """
    # Convert PyTorch tensors to numpy arrays
    mesh1_np = mesh1.detach().numpy()
    mesh2_np = mesh2.detach().numpy()

    # Create Open3D PointCloud objects
    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()

    pcd1.points = o3d.utility.Vector3dVector(mesh1_np)
    pcd2.points = o3d.utility.Vector3dVector(mesh2_np)

    # Set the colors
    pcd1.paint_uniform_color([1, 0, 0])  # Red for the first mesh
    pcd2.paint_uniform_color([0, 1, 0])  # Green for the second mesh

    # Visualize
    o3d.visualization.draw_geometries([pcd1, pcd2])
