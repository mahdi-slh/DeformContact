import open3d as o3d
import numpy as np

def visualize_deformation_field(rest_mesh, def_mesh):
    """
    Visualize the deformation field of two meshes.

    Args:
    rest_mesh: Resting state mesh (PyTorch tensor)
    def_mesh: Deformed state mesh (PyTorch tensor)
    """
    # Convert PyTorch tensors to numpy arrays
    rest_mesh_np = rest_mesh.detach().numpy()
    def_mesh_np = def_mesh.detach().numpy()

    # Create Open3D PointCloud objects for both meshes
    pcd_rest = o3d.geometry.PointCloud()
    pcd_def = o3d.geometry.PointCloud()
    pcd_rest.points = o3d.utility.Vector3dVector(rest_mesh_np)
    pcd_def.points = o3d.utility.Vector3dVector(def_mesh_np)

    # Create a LineSet object for the deformation field
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(np.concatenate((rest_mesh_np, def_mesh_np)))

    # The lines are defined by pairs of indices pointing to the points
    # Here, we are connecting each point i in the resting mesh to the corresponding point i + n in the deformed mesh
    # where n is the number of points in the resting mesh
    n = len(rest_mesh_np)
    lineset.lines = o3d.utility.Vector2iVector([(i, i + n) for i in range(n)])

    # Set the colors
    pcd_rest.paint_uniform_color([1, 0, 0])  # Red for the resting mesh
    pcd_def.paint_uniform_color([0, 1, 0])  # Green for the deformed mesh
    lineset.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in range(n)])  # Blue for the lines

    # Visualize
    o3d.visualization.draw_geometries([pcd_rest, pcd_def, lineset])


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
