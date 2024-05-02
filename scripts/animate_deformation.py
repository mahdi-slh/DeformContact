import open3d as o3d
import numpy as np
import os
import shutil
import time
import pathlib
from tqdm import tqdm

def animate_meshes(initial_mesh, def_mesh, num_frames, extrapolation_frames):
    assert initial_mesh.has_vertices() and def_mesh.has_vertices(), "Meshes must have vertices"
    assert len(initial_mesh.vertices) == len(def_mesh.vertices), "Meshes must have the same number of vertices"
    
    initial_vertices = np.asarray(initial_mesh.vertices)
    def_vertices = np.asarray(def_mesh.vertices)
    
    interpolated_meshes = []
    
    total_frames = num_frames + extrapolation_frames
    
    for i in range(total_frames):
        if i < num_frames:
            alpha = i / (num_frames - 1)  
        else:
            alpha = 1 + (i - num_frames + 1) / extrapolation_frames 
        
        interpolated_vertices = (1 - alpha) * initial_vertices + alpha * def_vertices
        interpolated_mesh = o3d.geometry.TriangleMesh()
        interpolated_mesh.vertices = o3d.utility.Vector3dVector(interpolated_vertices)
        interpolated_mesh.triangles = initial_mesh.triangles
        interpolated_meshes.append(interpolated_mesh)
        
    return interpolated_meshes


        
def process_mesh(directory_path, file_name_deformed):
    file_name_resting = "InitialMesh.ply"
    

    parent_directory_path = pathlib.Path(directory_path).parent
    output_dir = os.path.join(parent_directory_path, "Cat_animate")
    os.makedirs(output_dir, exist_ok=True)

    json_file_name = file_name_deformed + ".json"
    shutil.copy(os.path.join(directory_path, json_file_name), os.path.join(output_dir, json_file_name))
    
    def_mesh = o3d.io.read_triangle_mesh(os.path.join(directory_path, file_name_deformed+'.ply'))
    initial_mesh = o3d.io.read_triangle_mesh(os.path.join(directory_path, file_name_resting))
    
    num_interpolation_frames = 5
    num_extrapolation_frames = 5
    
    mesh_list = animate_meshes(initial_mesh, def_mesh, num_interpolation_frames, num_extrapolation_frames)
    

    for idx, mesh in enumerate(mesh_list):
        file_name = f"{file_name_deformed}_{idx}.ply"
        o3d.io.write_triangle_mesh(os.path.join(output_dir, file_name), mesh)

if __name__ == "__main__":
    directory_path = "dataset/everyday_deform/Cat/"
    all_files = [f for f in os.listdir(directory_path) if f.startswith("2023") and f.endswith(".ply")]
    
    for file in tqdm(all_files, desc="Processing files"):
        file_name_deformed = os.path.splitext(file)[0]
        process_mesh(directory_path, file_name_deformed)