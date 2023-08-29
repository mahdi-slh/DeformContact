import open3d as o3d
import numpy as np

import open3d as o3d

import numpy as np
import open3d as o3d
import numpy as np
import open3d as o3d

def visualize_deformation_field(rest_mesh, def_mesh, contact_point, origin_point):
    rest_mesh_np = rest_mesh.detach().numpy()
    def_mesh_np = def_mesh.detach().numpy()
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

    sphere_radius = 0.02
    sphere_contact = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    sphere_contact.translate(contact_point_np)
    sphere_contact.paint_uniform_color([0, 0, 1])  # Blue for the contact sphere
    
    # Yellow sphere for the point with the biggest deformation field size
    sphere_deform = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    sphere_deform.translate(max_deformation_point)
    sphere_deform.paint_uniform_color([1, 1, 0])  # Yellow

    # Create the vector from the origin to the contact point
    vector_lineset = o3d.geometry.LineSet()
    vector_lineset.points = o3d.utility.Vector3dVector([origin_point_np, contact_point_np])
    vector_lineset.lines = o3d.utility.Vector2iVector([[0, 1]])
    vector_lineset.colors = o3d.utility.Vector3dVector([[0, 0, 0]])  # Black color for the vector, adjust as necessary

    pcd_rest.paint_uniform_color([0, 0.8,0])
    pcd_def.paint_uniform_color([0.8, 0.8, 0])
    lineset.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5] for _ in range(n)])

    coor = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    print(contact_point_np, max_deformation_point)
    o3d.visualization.draw_geometries([pcd_rest, pcd_def, lineset, sphere_contact, sphere_deform, coor, vector_lineset])


def visualize_merged_graphs(rest_points, rest_edge_index, def_points, def_edge_index, translation=1.2):
    """
    Visualize the rest and deformed graphs side-by-side using Open3D.
    
    Args:
    - rest_points (torch.Tensor): Tensor of shape [num_points, 3] representing the rest point cloud.
    - rest_edge_index (torch.Tensor): Tensor of shape [2, num_edges] defining the edge relations for rest graph.
    - def_points (torch.Tensor): Tensor of shape [num_points, 3] representing the deformed point cloud.
    - def_edge_index (torch.Tensor): Tensor of shape [2, num_edges] defining the edge relations for deformed graph.
    - translation (float): Distance to translate the deformed object for side-by-side visualization.
    """
    
    # Convert tensor to numpy
    rest_points_np = rest_points.cpu().numpy()
    rest_edge_index_np = rest_edge_index.t().cpu().numpy()
    
    def_points_np = def_points.cpu().numpy()
    def_edge_index_np = def_edge_index.t().cpu().numpy()

    # Translate deformed points for side-by-side visualization
    def_points_np += [translation, 0, 0]
    
    # Create Open3D point cloud objects
    rest_pcd = o3d.geometry.PointCloud()
    rest_pcd.points = o3d.utility.Vector3dVector(rest_points_np)
    rest_pcd.paint_uniform_color([0, 0.8, 0])  # Red for rest mesh
    
    def_pcd = o3d.geometry.PointCloud()
    def_pcd.points = o3d.utility.Vector3dVector(def_points_np)
    def_pcd.paint_uniform_color([0.8, 0.8, 0])  # Green for deformed mesh
    
    # Create lines for the edges
    rest_lines = o3d.geometry.LineSet()
    rest_lines.points = o3d.utility.Vector3dVector(rest_points_np)
    rest_lines.lines = o3d.utility.Vector2iVector(rest_edge_index_np)
    
    def_lines = o3d.geometry.LineSet()
    def_lines.points = o3d.utility.Vector3dVector(def_points_np)
    def_lines.lines = o3d.utility.Vector2iVector(def_edge_index_np)
    
    # Visualize
    o3d.visualization.draw_geometries([rest_pcd, def_pcd, rest_lines, def_lines])



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
