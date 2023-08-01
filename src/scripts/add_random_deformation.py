import open3d as o3d
import numpy as np


def apply_deformation(mesh, center, radius, sigma):
    # Get vertices as a numpy array
    vertices = np.asarray(mesh.vertices)

    # Compute the distances from the center to each vertex
    distances = np.linalg.norm(vertices - center, axis=1)

    # Compute the Gaussian deformation for each distance
    deformation = np.exp(-(distances**2)/(2*(sigma**2)))

    # Only apply deformation inside the sphere
    deformation[distances > radius] = 0

    # Apply the deformation to the vertices
    # Multiply by the deformation vector to have a radial deformation
    vertices += (vertices - center) * deformation[:, np.newaxis]

    # Assign the deformed vertices back to the mesh
    mesh.vertices = o3d.utility.Vector3dVector(vertices)


# Load the triangle mesh
mesh = o3d.io.read_triangle_mesh("datasets/everyday_deform/deformations/bag/01_rest.obj")


# Define the sphere parameters
sphere_radius = 0.3  # Adjust as needed
sigma = 0.3  # Adjust as needed

# Get the bounding box of the mesh
bbox = mesh.get_axis_aligned_bounding_box()

# Generate a random center for the sphere inside the bounding box
center = np.random.uniform(low=bbox.min_bound, high=bbox.max_bound)

# Apply the deformation
apply_deformation(mesh, center, sphere_radius, sigma)

# Visualize the deformed mesh
o3d.visualization.draw_geometries([mesh])

# Save the deformed mesh
o3d.io.write_triangle_mesh("datasets/everyday_deform/deformations/bag/01_def.obj", mesh)