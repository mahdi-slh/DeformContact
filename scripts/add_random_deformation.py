import open3d as o3d
import numpy as np
import json

frame_id = '00'

def apply_deformation(mesh, center, sigma_vector):
    vertices = np.asarray(mesh.vertices)
    differences = vertices - center
    deformation = np.exp(-((differences**2)/(2*(sigma_vector**2))))
    vertices += differences * deformation
    force = -2 * differences * deformation
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return vertices, np.sum(force, axis=0)

mesh_path = f"dataset/everyday_deform/bag/{frame_id}_rest.obj"
mesh = o3d.io.read_triangle_mesh(mesh_path)


vertex_indices = np.arange(len(mesh.vertices))
random_vertex_index = np.random.choice(vertex_indices)
center = np.asarray(mesh.vertices)[random_vertex_index]

sigma_vector = np.random.uniform(low=0, high=0.05, size=3)
deformed_vertices, force_vector = apply_deformation(mesh, center, sigma_vector)
closest_vertice = deformed_vertices[random_vertex_index]

data = {
    "contact_event": {
        "contact_position": {
            "x": closest_vertice[0],
            "y": closest_vertice[1],
            "z": closest_vertice[2]
        },
        "contact_type": "sharp",
        "timestamp": 0.0,
        "force_vector": {
            "fx": force_vector[0],
            "fy": force_vector[1],
            "fz": force_vector[2]
        }
    }
}

json_path = f"dataset/everyday_deform/bag/{frame_id}.json"
with open(json_path, 'w') as file:
    json.dump(data, file, indent=4)

o3d.visualization.draw_geometries([mesh])

mesh_out_path = f"dataset/everyday_deform/bag/{frame_id}_def.obj"
o3d.io.write_triangle_mesh(mesh_out_path, mesh)
