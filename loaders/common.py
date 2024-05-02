import open3d as o3d
import torch
import os
import numpy as np    

def _feature_rigid(meta_data,pos_enc): 

    force_vector = meta_data['force_vector']
    force_scaler = meta_data['force']



    vector_broadcasted = force_vector.repeat(pos_enc.shape[0], 1)

    force_scaler_t = torch.tensor(force_scaler, dtype=torch.float32)
    scaler_broadcasted = force_scaler_t.repeat(pos_enc.shape[0], 1)

    features = torch.cat([vector_broadcasted,scaler_broadcasted, pos_enc,], dim=1)
    return features

def _unity_to_open3d(vector_or_position):
    x, y, z = vector_or_position
    return [z,-x, y]

def _create_rigid_pointcloud(contact_point_np,rigid_radius):
    rigid_contact_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=rigid_radius)
    rigid_contact_mesh.translate(contact_point_np)
    pcd_contact = o3d.geometry.PointCloud()
    pcd_contact.points = o3d.utility.Vector3dVector(np.array(rigid_contact_mesh.vertices))
    return rigid_contact_mesh


def _load_deformed_mesh(sample_path, meta_data,root_dir):
    soft_def_mesh = o3d.io.read_triangle_mesh(os.path.join(root_dir, sample_path + ".ply"))
    soft_def_mesh.translate(meta_data['object_rigid_pos'].detach().cpu().numpy())
    return soft_def_mesh
 

def _sample_nearest(rigid_mesh, soft_def_mesh, soft_rest_mesh, n_points, use_center_only=True):

    center = np.mean(np.asarray(rigid_mesh.vertices), axis=0)

    pcd_rest = o3d.geometry.PointCloud()
    pcd_rest.points = o3d.utility.Vector3dVector(np.asarray(soft_rest_mesh.vertices))
    kdtree = o3d.geometry.KDTreeFlann(pcd_rest)

    _, idx, _ = kdtree.search_knn_vector_3d(center, n_points)
    idx_set = set(idx) 

    soft_rest_vertices_np = np.asarray(pcd_rest.points)[idx]
    soft_def_vertices_np = np.asarray(soft_def_mesh.vertices)[idx]


    soft_def_triangles = np.asarray(soft_def_mesh.triangles)
    sampled_triangles = [tri for tri in soft_def_triangles if all(v in idx_set for v in tri)]


    index_map = {v: i for i, v in enumerate(idx)}
    sampled_triangles_adjusted = np.array([[index_map[v] for v in tri] for tri in sampled_triangles])

    sampled_soft_rest_mesh = o3d.geometry.TriangleMesh()
    sampled_soft_rest_mesh.vertices = o3d.utility.Vector3dVector(soft_rest_vertices_np)
    sampled_soft_rest_mesh.triangles = o3d.utility.Vector3iVector(sampled_triangles_adjusted)

    sampled_soft_def_mesh = o3d.geometry.TriangleMesh()
    sampled_soft_def_mesh.vertices = o3d.utility.Vector3dVector(soft_def_vertices_np)
    sampled_soft_def_mesh.triangles = o3d.utility.Vector3iVector(sampled_triangles_adjusted)

    return sampled_soft_rest_mesh, sampled_soft_def_mesh
