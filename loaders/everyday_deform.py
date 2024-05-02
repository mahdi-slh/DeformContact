import os
from torch.utils.data import Dataset
import open3d as o3d
from utils.pointcloud_utils import construct_graph
import torch
import json
from utils.graph_utils import mesh_to_graph
import copy
from loaders.common import *
from loaders.common import _unity_to_open3d,_create_rigid_pointcloud,_feature_rigid,_load_deformed_mesh,_sample_nearest

class EverydayDeformDataset(Dataset):
    def __init__(self, root_dir, obj_list, n_points, graph_method,sphere_radius,force_max, neigbor_radius=None, neigbor_k=None, split='train'):
        self.root_dir = root_dir

        self.samples = []
        self.soft_rest_mesh = {}
        for obj in obj_list:
            obj_samples = [os.path.join(obj, f[:-4]) for f in os.listdir(os.path.join(root_dir, obj)) if f.endswith('.ply') and f != 'InitialMesh.ply']
            obj_samples.sort()
            num_samples = len(obj_samples)
            train_samples = int(0.8 * num_samples)
            if split == 'train':
                obj_split = obj_samples[:train_samples]
            elif split == 'val':
                obj_split = obj_samples[train_samples:]
            self.samples.extend(obj_split)

            soft_rest_mesh = o3d.io.read_triangle_mesh(os.path.join(self.root_dir, obj, 'InitialMesh.ply'))
            self.soft_rest_mesh[obj] = soft_rest_mesh

        self.n_points = n_points
        self.neigbor_radius = neigbor_radius
        self.neigbor_k = neigbor_k
        self.force_max = force_max
        self.graph_method = graph_method
        self.rigid_radius = sphere_radius

    def __len__(self):
        return len(self.samples)

    def _construct_graph(self, mesh_tensor):
        if self.graph_method == 'knn':
            return construct_graph(mesh_tensor, k=self.neigbor_k)
        elif self.graph_method == 'radius':
            return construct_graph(mesh_tensor, radius=self.neigbor_radius)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        obj_name = os.path.basename(os.path.dirname(sample_path))

        meta_data = self._read_meta_data(sample_path)
        contact_point_np = meta_data['deformer_collision_position'].detach().numpy()
        rigid_mesh = _create_rigid_pointcloud(contact_point_np,self.rigid_radius)
        rigid_graph = mesh_to_graph(rigid_mesh)
        rigid_graph.x = _feature_rigid(meta_data, rigid_graph.x)
        soft_def_mesh = _load_deformed_mesh(sample_path, meta_data,self.root_dir)
        soft_rest_mesh = copy.deepcopy(self.soft_rest_mesh[obj_name])
        soft_rest_mesh.translate(meta_data['object_rigid_pos'].detach().cpu().numpy())
        if self.n_points ==-1:
            sampled_soft_rest_mesh_o3d, sampled_soft_def_mesh_o3d = soft_rest_mesh, soft_def_mesh
        else:
            sampled_soft_rest_mesh_o3d, sampled_soft_def_mesh_o3d = _sample_nearest(rigid_mesh, soft_def_mesh, soft_rest_mesh,self.n_points)
        soft_rest_graph = mesh_to_graph(sampled_soft_rest_mesh_o3d)
        soft_def_graph = mesh_to_graph(sampled_soft_def_mesh_o3d)

        meta_data['rigid_mesh'] = rigid_mesh
        meta_data['soft_rest_mesh'] = soft_rest_mesh
        
        meta_data['sample_path'] = sample_path

        return obj_name, soft_rest_graph, soft_def_graph, meta_data, rigid_graph

    def _read_meta_data(self, sample_path):
        with open(os.path.join(self.root_dir, sample_path + ".json"), "r") as f:
            json_data = json.load(f)[0]

        force_vector = torch.tensor(_unity_to_open3d([
            json_data['forceDirectionX'],
            json_data['forceDirectionY'],
            json_data['forceDirectionZ']
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
            json_data['deformerOriginX'],
            json_data['deformerOriginY'],
            json_data['deformerOriginZ']
        ]), dtype=torch.float32)

        deformer_collision_position = torch.tensor(_unity_to_open3d([
            json_data['deformerCollisionPositionX'],
            json_data['deformerCollisionPositionY'],
            json_data['deformerCollisionPositionZ']
        ]), dtype=torch.float32)

        return {
            'force': json_data['force']/self.force_max,
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
            'object_rigid_pos': object_rigid_pos
        }
