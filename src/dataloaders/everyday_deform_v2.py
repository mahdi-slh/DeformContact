import os
import numpy as np
from torch.utils.data import Dataset
import open3d as o3d

from utils.pointcloud_utils import fps_points,construct_graph
import torch
from torch_geometric.data import Data
import json
from utils.graph_utils import compute_deformation_using_diff_coords
from utils.pos_encoding import to_log_freq

class EverydayDeformDataset(Dataset):
    def __init__(self, root_dir, obj_list, n_points, graph_method, radius=None, k=None, split='train'):
        self.root_dir = root_dir

        # Collect all deformation samples
        self.samples = []
        for obj in obj_list:
            # Exclude 'InitialMesh.ply' from the deformed meshes list
            obj_samples = [os.path.join(obj, f[:-4]) for f in os.listdir(os.path.join(root_dir, obj)) if f.endswith('.ply') and f != 'InitialMesh.ply']
            self.samples.extend(obj_samples)

        # Order the samples for deterministic split
        self.samples.sort()

        # Split the dataset
        num_samples = len(self.samples)
        train_samples = int(0.8 * num_samples)
        if split == 'train':
            self.samples = self.samples[:train_samples]
        elif split == 'val':
            self.samples = self.samples[train_samples:]

        # Read 'InitialMesh.ply' once and compute FPS indices
        example_obj = obj_list[0]  # Using the first object as an example to get the path for 'InitialMesh.ply'
        soft_rest_mesh = o3d.io.read_triangle_mesh(os.path.join(self.root_dir, example_obj, 'InitialMesh.ply'))
        self.soft_rest_mesh_np = np.asarray(soft_rest_mesh.vertices)

        # Sample indices using FPS
        self.n_points = n_points
        self.radius = radius
        self.k = k
        self.graph_method=graph_method
        self.sampled_indices = fps_points(self.soft_rest_mesh_np, n_points, return_indices=True)[1]
        self.rigid_radius = 0.25


    def __len__(self):
        return len(self.samples)//10

    def _load_deformed_mesh(self, sample_path, meta_data):
        soft_def_mesh = o3d.io.read_triangle_mesh(os.path.join(self.root_dir, sample_path + ".ply"))
        soft_def_mesh.translate(meta_data['object_rigid_pos'].detach().cpu().numpy())
        return np.asarray(soft_def_mesh.vertices)

    def _sample_points(self, soft_def_mesh_np):
        sampled_soft_rest_mesh_np = self.soft_rest_mesh_np[self.sampled_indices]
        sampled_soft_def_mesh_np = soft_def_mesh_np[self.sampled_indices]
        return torch.tensor(sampled_soft_rest_mesh_np, dtype=torch.float32), torch.tensor(sampled_soft_def_mesh_np, dtype=torch.float32)

    def _construct_graph(self, mesh_tensor):
        if self.graph_method == 'knn':
            return construct_graph(mesh_tensor, k=self.k)
        elif self.graph_method == 'radius':
            return construct_graph(mesh_tensor, radius=self.radius)

    def _create_rigid_pointcloud(self, contact_point_np):
        rigid_contact_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=self.rigid_radius)
        rigid_contact_mesh.translate(contact_point_np)
        pcd_contact = o3d.geometry.PointCloud()
        pcd_contact.points = o3d.utility.Vector3dVector(np.array(rigid_contact_mesh.vertices))
        return rigid_contact_mesh

    def _feature_rigid(self, rigid_contact_mesh, vector_lineset):
        vertices = np.asarray(rigid_contact_mesh.vertices)
        triangles = np.asarray(rigid_contact_mesh.triangles)
        vertices_t = torch.tensor(vertices, dtype=torch.float32)

        edge_indices = [[triangle[i], triangle[(i+1)%3]] for triangle in triangles for i in range(3)]
        edge_indices_t = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

        vector_lineset_np = np.asarray(vector_lineset.points)
        vector_diff = vector_lineset_np[1] - vector_lineset_np[0]
        vector_diff_t = torch.tensor(vector_diff, dtype=torch.float32)
        vector_broadcasted = vector_diff_t.repeat(vertices_t.shape[0], 1)

        features = torch.cat([vertices_t, vector_broadcasted], dim=1)
        
        return Data(x=features, edge_index=edge_indices_t, pos=vertices_t)
    
    def _feature_pos_enc(self, soft_contact_mesh):
        features = to_log_freq(soft_contact_mesh, 3, 1)
        
        return features

    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        obj_name = os.path.basename(os.path.dirname(sample_path))

        meta_data = self._read_meta_data(sample_path)
        
        soft_def_mesh_np = self._load_deformed_mesh(sample_path, meta_data)-meta_data['object_rigid_pos'].numpy()
        

        sampled_soft_rest_mesh_t, sampled_soft_def_mesh_t = self._sample_points(soft_def_mesh_np)
        if self.graph_method =='knn':
            soft_rest_edge_index = construct_graph(sampled_soft_rest_mesh_t,k=self.k)
        elif self.graph_method =='radius':
            soft_rest_edge_index = construct_graph(sampled_soft_rest_mesh_t,radius=self.radius)
        
        feat_soft_rest_mesh_t = self._feature_pos_enc(sampled_soft_rest_mesh_t)

        soft_def_edge_index = soft_rest_edge_index.clone()

        soft_rest_graph = Data(x=feat_soft_rest_mesh_t, edge_index=soft_rest_edge_index, pos=sampled_soft_rest_mesh_t) 
        soft_def_graph = Data(x=sampled_soft_def_mesh_t, edge_index=soft_def_edge_index, pos=sampled_soft_def_mesh_t)

        meta_data['deform_intensity'] = compute_deformation_using_diff_coords(soft_rest_graph,soft_def_graph)


        contact_point_np = meta_data['deformer_collision_position'].detach().numpy()
        origin_point_np = meta_data['deformer_origin'].detach().numpy()

        vector_lineset = o3d.geometry.LineSet()
        vector_lineset.points = o3d.utility.Vector3dVector([origin_point_np, contact_point_np])
        vector_lineset.lines = o3d.utility.Vector2iVector([[0, 1]])

        rigid_contact_mesh = self._create_rigid_pointcloud(contact_point_np)
        rigid_contact_mesh.translate(-meta_data['object_rigid_pos'].numpy())
        rigid_graph = self._feature_rigid(rigid_contact_mesh, vector_lineset)

        return obj_name, soft_rest_graph, soft_def_graph, meta_data, rigid_graph

        

    def _unity_to_open3d(self, vector_or_position):
        x, y, z = vector_or_position
        return [z,-x, y]

    def _read_meta_data(self, sample_path):
        with open(os.path.join(self.root_dir, sample_path + ".json"), "r") as f:
            json_data = json.load(f)[0]  # Assuming the first object in the array

        force_vector = torch.tensor(
            self._unity_to_open3d([
                json_data['forceDirectionX'],
                json_data['forceDirectionY'],
                json_data['forceDirectionZ']
            ]), dtype=torch.float32)

        object_rigid_pos = torch.tensor(
            self._unity_to_open3d([
                json_data['objectWorldPosX'],
                json_data['objectWorldPosY'],
                json_data['objectWorldPosZ']
            ]), dtype=torch.float32)
        

        contact_position = torch.tensor(
            self._unity_to_open3d([
                json_data['collisionPositionX'] - json_data['objectWorldPosX'],
                json_data['collisionPositionY'] - json_data['objectWorldPosY'],
                json_data['collisionPositionZ'] - json_data['objectWorldPosZ']
            ]), dtype=torch.float32)

        velocity = torch.tensor(
            self._unity_to_open3d([
                json_data['velocityX'],
                json_data['velocityY'],
                json_data['velocityZ']
            ]), dtype=torch.float32)

        angular_velocity = torch.tensor(
            self._unity_to_open3d([
                json_data['angularVelocityX'],
                json_data['angularVelocityY'],
                json_data['angularVelocityZ']
            ]), dtype=torch.float32)

        inertia_tensor_position = torch.tensor(
            self._unity_to_open3d([
                json_data['inertiaTensorPositionX'],
                json_data['inertiaTensorPositionY'],
                json_data['inertiaTensorPositionZ']
            ]), dtype=torch.float32)

        inertia_tensor_rotation = torch.tensor([
            json_data['inertiaTensorRotationX'],
            json_data['inertiaTensorRotationY'],
            json_data['inertiaTensorRotationZ']
        ], dtype=torch.float32)

        deformer_origin = torch.tensor(
            self._unity_to_open3d([
                json_data['deformerOriginX'],
                json_data['deformerOriginY'],
                json_data['deformerOriginZ']
            ]), dtype=torch.float32)

        deformer_collision_position = torch.tensor(
            self._unity_to_open3d([
                json_data['deformerCollisionPositionX'],
                json_data['deformerCollisionPositionY'],
                json_data['deformerCollisionPositionZ']
            ]), dtype=torch.float32)

        return {
            'force': json_data['force'],
            'force_vector': force_vector,
            'contact_position': contact_position,
            'collision_impulse': json_data['collisionImpulse'],
            'mass': json_data['mass'],
            'velocity': velocity,
            'angular_velocity': angular_velocity,
            'inertia_tensor_position': inertia_tensor_position,
            'inertia_tensor_rotation': inertia_tensor_rotation,
            'gravity_enabled': json_data['gravity_enabled'],
            'deformer_origin': deformer_origin,
            'deformer_collision_position': deformer_collision_position,
            'object_rigid_pos':object_rigid_pos
        }
