

import open3d as o3d
import numpy as np
from utils.graph_utils import *

def visualize_deformation_field(soft_rest_graph, soft_def_graph, contact_point, origin_point):
    # Extract node features (points) from the graph object
    soft_rest_mesh_np = soft_rest_graph.detach().numpy()
    soft_def_mesh_np = soft_def_graph.detach().numpy()
    contact_point_np = contact_point.detach().numpy()
    origin_point_np = origin_point.detach().numpy()

    # Calculate deformation magnitudes and find the point with the maximum deformation
    deformation_magnitudes = np.linalg.norm(soft_def_mesh_np - soft_rest_mesh_np, axis=1)
    max_deformation_index = np.argmax(deformation_magnitudes)
    max_deformation_point = soft_rest_mesh_np[max_deformation_index]

    pcd_rest = o3d.geometry.PointCloud()
    pcd_def = o3d.geometry.PointCloud()
    pcd_rest.points = o3d.utility.Vector3dVector(soft_rest_mesh_np)
    pcd_def.points = o3d.utility.Vector3dVector(soft_def_mesh_np)

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(np.concatenate((soft_rest_mesh_np, soft_def_mesh_np)))
    n = len(soft_rest_mesh_np)
    lineset.lines = o3d.utility.Vector2iVector([(i, i + n) for i in range(n)])

    # Create rigids as point clouds by converting the vertices of the triangle meshes to point clouds
    rigid_contact_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.25)
    rigid_contact_mesh.translate(contact_point_np)
    pcd_contact = o3d.geometry.PointCloud()
    pcd_contact.points = o3d.utility.Vector3dVector(np.array(rigid_contact_mesh.vertices))
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



def visualize_merged_graphs(soft_rest_graph, soft_def_graph=None, rigid_graph=None, pred_graph=None, translation=1.2):
    soft_rest_points_np = soft_rest_graph.pos.cpu().numpy()
    soft_rest_edge_index_np = soft_rest_graph.edge_index.t().cpu().numpy().astype(np.int32)

    soft_rest_pcd = o3d.geometry.PointCloud()
    soft_rest_pcd.points = o3d.utility.Vector3dVector(soft_rest_points_np)
    soft_rest_pcd.paint_uniform_color([0, 0.8, 0])  # Green for rest mesh

    soft_rest_lines = o3d.geometry.LineSet()
    soft_rest_lines.points = o3d.utility.Vector3dVector(soft_rest_points_np)
    soft_rest_lines.lines = o3d.utility.Vector2iVector(soft_rest_edge_index_np)

    geometries = [soft_rest_pcd, soft_rest_lines]


    soft_def_points_np = soft_def_graph.pos.cpu().numpy() + [translation, 0, 0]
    soft_def_edge_index_np = soft_def_graph.edge_index.t().cpu().numpy().astype(np.int32)
    
    soft_def_pcd = o3d.geometry.PointCloud()
    soft_def_pcd.points = o3d.utility.Vector3dVector(soft_def_points_np)
    soft_def_pcd.paint_uniform_color([0.8, 0.8, 0])  # Yellow for deformed mesh
    
    soft_def_lines = o3d.geometry.LineSet()
    soft_def_lines.points = o3d.utility.Vector3dVector(soft_def_points_np)
    soft_def_lines.lines = o3d.utility.Vector2iVector(soft_def_edge_index_np)
    
    geometries.extend([soft_def_pcd, soft_def_lines])


    rigid_points_np = rigid_graph.pos.cpu().numpy() + [translation, 0, 0]
    rigid_edge_index_np = rigid_graph.edge_index.t().cpu().numpy().astype(np.int32)
    
    rigid_pcd = o3d.geometry.PointCloud()
    rigid_pcd.points = o3d.utility.Vector3dVector(rigid_points_np)
    rigid_pcd.paint_uniform_color([0.5, 0.5, 0.8])  # Blue-ish for the rigid mesh
    
    rigid_lines = o3d.geometry.LineSet()
    rigid_lines.points = o3d.utility.Vector3dVector(rigid_points_np)
    rigid_lines.lines = o3d.utility.Vector2iVector(rigid_edge_index_np)
    
    geometries.extend([rigid_pcd, rigid_lines])

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


        rigid_points_np = rigid_graph.clone().pos.cpu().numpy() + [2*translation, 0, 0]
        rigid_edge_index_np = rigid_graph.clone().edge_index.t().cpu().numpy().astype(np.int32)
        
        rigid_pcd = o3d.geometry.PointCloud()
        rigid_pcd.points = o3d.utility.Vector3dVector(rigid_points_np)
        rigid_pcd.paint_uniform_color([0.5, 0.5, 0.8])  # Blue-ish for the rigid mesh
        
        rigid_lines = o3d.geometry.LineSet()
        rigid_lines.points = o3d.utility.Vector3dVector(rigid_points_np)
        rigid_lines.lines = o3d.utility.Vector2iVector(rigid_edge_index_np)
        
        geometries.extend([rigid_pcd, rigid_lines])

    o3d.visualization.draw_geometries(geometries)


def map_deformation_to_color(deformation_values):
    min_val, max_val = np.min(deformation_values), np.max(deformation_values)
    normalized_values = (deformation_values - min_val) / (max_val - min_val)
    
    # Transition from white to black
    colors = np.ones((normalized_values.shape[0], 3)) - np.expand_dims(normalized_values, axis=1)
    
    return colors


def visualize_deformations_intensity(soft_rest_graph, soft_def_graph,deformation_intensities, translation=1.2):
    soft_rest_points_np = soft_rest_graph.pos.cpu().numpy()
    soft_rest_edge_index_np = soft_rest_graph.edge_index.t().cpu().numpy().astype(np.int32)
    
    # Compute deformation values
    deformation_values = deformation_intensities.cpu().numpy()
    deformation_colors = map_deformation_to_color(deformation_values)

    soft_rest_pcd = o3d.geometry.PointCloud()
    soft_rest_pcd.points = o3d.utility.Vector3dVector(soft_rest_points_np)
    soft_rest_pcd.colors = o3d.utility.Vector3dVector(deformation_colors)
    
    soft_rest_lines = o3d.geometry.LineSet()
    soft_rest_lines.points = o3d.utility.Vector3dVector(soft_rest_points_np)
    soft_rest_lines.lines = o3d.utility.Vector2iVector(soft_rest_edge_index_np)
    
    geometries = [soft_rest_pcd, soft_rest_lines]
    
    # Adding the deformed mesh visualization
    soft_def_points_np = soft_def_graph.pos.cpu().numpy() + [translation, 0, 0]
    soft_def_edge_index_np = soft_def_graph.edge_index.t().cpu().numpy().astype(np.int32)

    soft_def_pcd = o3d.geometry.PointCloud()
    soft_def_pcd.points = o3d.utility.Vector3dVector(soft_def_points_np)
    soft_def_pcd.colors = o3d.utility.Vector3dVector(deformation_colors)  # Use same deformation colors

    soft_def_lines = o3d.geometry.LineSet()
    soft_def_lines.points = o3d.utility.Vector3dVector(soft_def_points_np)
    soft_def_lines.lines = o3d.utility.Vector2iVector(soft_def_edge_index_np)

    geometries.extend([soft_def_pcd, soft_def_lines])  # Add deformed mesh geometries

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
