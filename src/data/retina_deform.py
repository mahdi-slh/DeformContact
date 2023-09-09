import os
from torch.utils.data import Dataset
import open3d as o3d
from utils.pointcloud_utils import construct_graph
import torch
import json
from utils.graph_utils import mesh_to_graph
import copy
from data.common import _unity_to_open3d,_create_rigid_pointcloud,_feature_rigid,_load_deformed_mesh,_sample_nearest

class RetinaDeformDataset(Dataset):
    def __init__(self, root_dir, n_points, graph_method, radius=None, k=None, split='train'):
        self.root_dir = root_dir

        self.samples = [os.path.join( f[:-4]) for f in os.listdir(os.path.join(root_dir)) if f.endswith('.ply') and f != 'InitialMesh.ply' and f != 'Rigid.ply']
        self.samples.sort()

        num_samples = len(self.samples)
        train_samples = int(0.8 * num_samples)
        if split == 'train':
            self.samples = self.samples[:train_samples]
        elif split == 'val':
            self.samples = self.samples[train_samples:]

        self.rigid_mesh = o3d.io.read_triangle_mesh(os.path.join(self.root_dir, 'Rigid.ply'))
        self.soft_rest_mesh =  o3d.io.read_triangle_mesh(os.path.join(self.root_dir, 'InitialMesh.ply'))
        self.n_points = n_points
        self.radius = radius
        self.k = k
        self.graph_method = graph_method
        self.rigid_radius = 0.25

    def __len__(self):
        return len(self.samples)

    def _construct_graph(self, mesh_tensor):
        if self.graph_method == 'knn':
            return construct_graph(mesh_tensor, k=self.k)
        elif self.graph_method == 'radius':
            return construct_graph(mesh_tensor, radius=self.radius)

    def __getitem__(self, idx):
        rigid_mesh = copy.deepcopy(self.rigid_mesh)
        
        sample_path = self.samples[idx]
        obj_name = os.path.basename(os.path.dirname(sample_path))
        meta_data = self._read_meta_data(sample_path)
        rigid_mesh.translate(meta_data['rigid_pos'])
        rigid_graph = mesh_to_graph(rigid_mesh)
        rigid_graph.x = _feature_rigid(meta_data, rigid_graph.x)
        soft_def_mesh = _load_deformed_mesh(sample_path, meta_data,self.root_dir)
        soft_rest_mesh = copy.deepcopy(self.soft_rest_mesh)
        # soft_rest_mesh.translate(meta_data['object_rigid_pos'].detach().cpu().numpy())
        sampled_soft_rest_mesh_t, sampled_soft_def_mesh_t = _sample_nearest(rigid_mesh, soft_def_mesh, soft_rest_mesh,self.n_points)
        soft_rest_graph = mesh_to_graph(sampled_soft_rest_mesh_t)
        soft_def_graph = mesh_to_graph(sampled_soft_def_mesh_t)
        return obj_name, soft_rest_graph, soft_def_graph, meta_data, rigid_graph

    def _read_meta_data(self, sample_path):
        with open(os.path.join(self.root_dir, sample_path + ".json"), "r") as f:
            json_data = json.load(f)[0]

        force_vector = torch.tensor(_unity_to_open3d([
            json_data['forceDirectionX'],
            json_data['forceDirectionY'],
            json_data['forceDirectionZ']
        ]), dtype=torch.float32)

        rigid_pos = torch.tensor(_unity_to_open3d([
            json_data['deformerCollisionPositionX'],
            json_data['deformerCollisionPositionY'],
            json_data['deformerCollisionPositionZ']
        ]), dtype=torch.float32)

        object_rigid_pos = torch.tensor(_unity_to_open3d([
            json_data['objectWorldPosX'],
            json_data['objectWorldPosY'],
            json_data['objectWorldPosZ']
        ]), dtype=torch.float32)

        contact_position = torch.tensor(_unity_to_open3d([
            json_data['collisionPositionX'] - json_data['objectWorldPosX'],
            json_data['collisionPositionY'] - json_data['objectWorldPosY'],
            json_data['collisionPositionZ'] - json_data['objectWorldPosZ']
        ]), dtype=torch.float32)

        velocity = torch.tensor(_unity_to_open3d([
            json_data['velocityX'],
            json_data['velocityY'],
            json_data['velocityZ']
        ]), dtype=torch.float32)

        angular_velocity = torch.tensor([
            json_data['angularVelocityX'],
            json_data['angularVelocityY'],
            json_data['angularVelocityZ']
        ], dtype=torch.float32)

        inertia_tensor_position = torch.tensor(_unity_to_open3d([
            json_data['inertiaTensorPositionX'],
            json_data['inertiaTensorPositionY'],
            json_data['inertiaTensorPositionZ']
        ]), dtype=torch.float32)

        inertia_tensor_rotation = torch.tensor([
            json_data['inertiaTensorRotationX'],
            json_data['inertiaTensorRotationY'],
            json_data['inertiaTensorRotationZ']
        ], dtype=torch.float32)

        deformer_origin = torch.tensor(_unity_to_open3d([
            json_data['deformerInitialPositionX'],
            json_data['deformerInitialPositionY'],
            json_data['deformerInitialPositionZ']
        ]), dtype=torch.float32)

        deformer_collision_position = torch.tensor(_unity_to_open3d([
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
            'rigid_pos': rigid_pos,
            'object_rigid_pos':object_rigid_pos
        }
