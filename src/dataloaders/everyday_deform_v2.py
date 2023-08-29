import torch
from torch.utils.data import Dataset
import open3d as o3d
import numpy as np
import json
import os
from utils.pointcloud_utils import fps_points,construct_graph



class EverydayDeformDataset(Dataset):
    
    def __init__(self, root_dir, obj_list, n_points,graph_method, radius=None,k=None):
        self.root_dir = root_dir

        # Collect all deformation samples
        self.samples = []
        for obj in obj_list:
            # Exclude 'InitialMesh.ply' from the deformed meshes list
            obj_samples = [os.path.join(obj, f[:-4]) for f in os.listdir(os.path.join(root_dir, obj)) if f.endswith('.ply') and f != 'InitialMesh.ply']
            self.samples.extend(obj_samples)

        # Read 'InitialMesh.ply' once and compute FPS indices
        example_obj = obj_list[0]  # Using the first object as an example to get the path for 'InitialMesh.ply'
        rest_mesh = o3d.io.read_triangle_mesh(os.path.join(self.root_dir, example_obj, 'InitialMesh.ply'))
        self.rest_mesh_np = np.asarray(rest_mesh.vertices)

        # Sample indices using FPS
        self.n_points = n_points
        self.radius = radius
        self.k = k
        self.graph_method=graph_method
        self.sampled_indices = fps_points(self.rest_mesh_np, n_points, return_indices=True)[1]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        obj_name = os.path.basename(os.path.dirname(sample_path))
        sample_id = os.path.basename(sample_path)

        # Load the meta data
        meta_data = self._read_meta_data(sample_path)

        # Load the deformed mesh
        def_mesh = o3d.io.read_triangle_mesh(os.path.join(self.root_dir, sample_path + ".ply"))
        def_mesh.translate(meta_data['object_rigid_pos'].detach().cpu().numpy())
        def_mesh_np = np.asarray(def_mesh.vertices)

        # Use the pre-computed FPS indices to get corresponding points from both meshes
        sampled_rest_mesh_np = self.rest_mesh_np[self.sampled_indices]
        sampled_def_mesh_np = def_mesh_np[self.sampled_indices]

        # Convert the sampled points to torch tensors
        sampled_rest_mesh_t = torch.tensor(sampled_rest_mesh_np, dtype=torch.float32)
        sampled_def_mesh_t = torch.tensor(sampled_def_mesh_np, dtype=torch.float32)

        # Convert the tensors to KNN graphs for the resting mesh
        if self.graph_method =='knn':

            rest_edge_index = construct_graph(sampled_rest_mesh_t,k=self.k)
        elif self.graph_method =='radius':
            rest_edge_index = construct_graph(sampled_rest_mesh_t,radius=self.radius)


        # Use the same edge indices for the deformed mesh
        def_edge_index = rest_edge_index.clone()

        

        return obj_name, (sampled_rest_mesh_t, rest_edge_index), (sampled_def_mesh_t, def_edge_index), meta_data


    

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

    # ... rest of the class ...
