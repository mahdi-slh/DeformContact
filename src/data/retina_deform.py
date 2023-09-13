import os
from torch.utils.data import Dataset
import open3d as o3d
from utils.pointcloud_utils import construct_graph
import torch
import json
from utils.graph_utils import mesh_to_graph
import copy
from data.common import _unity_to_open3d,_create_rigid_pointcloud,_feature_rigid,_load_deformed_mesh,_sample_nearest
import numpy as np

class RetinaDeformDataset(Dataset):
    def __init__(self, root_dir, n_points, graph_method,force_max, neigbor_radius=None, neigbor_k=None , split='train'):
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
        self.neigbor_radius = neigbor_radius
        self.neigbor_k = neigbor_k
        self.graph_method = graph_method
        self.force_max = force_max

    def __len__(self):
        return len(self.samples)

    def _construct_graph(self, mesh_tensor):
        if self.graph_method == 'knn':
            return construct_graph(mesh_tensor, k=self.neigbor_k)
        elif self.graph_method == 'radius':
            return construct_graph(mesh_tensor, radius=self.neigbor_radius)

    def create_cylinder_between_points(self,a, b):
        # Create a cylinder mesh with specified height and radius
        cylinder_height = 15# np.linalg.norm(np.array(b) - np.array(a))
        mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.5, height=cylinder_height)

        # Calculate the direction vector from point a to point b
        direction = np.array(b) - np.array(a)
        direction_normalized = direction / np.linalg.norm(direction)

        # Calculate the rotation matrix
        # Cylinder is initially aligned with y-axis
        default_cylinder_dir = np.array([0, 1, 0])
        axis = np.cross(default_cylinder_dir, direction_normalized)
        axis_normalized = axis / np.linalg.norm(axis)
        angle = np.arcsin(-np.dot(default_cylinder_dir, direction_normalized))
        rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_normalized * angle)

        center = np.array(b) -direction_normalized*(cylinder_height/2)
        transform = np.identity(4)
        transform[:3, :3] = rot
        transform[:3, 3] = center

        mesh_cylinder.transform(transform)
        return mesh_cylinder
    def __getitem__(self, idx):
        rigid_mesh = copy.deepcopy(self.rigid_mesh)
        rigid_mesh_temp = copy.deepcopy(self.rigid_mesh)
        
        sample_path = self.samples[idx]
        obj_name = os.path.basename(os.path.dirname(sample_path))
        meta_data = self._read_meta_data(sample_path)
        
        # R = rigid_mesh.get_rotation_matrix_from_xyz(np.pi*meta_data['object_rigid_rot']/180)
        # rigid_mesh.rotate(R)
        # rigid_mesh.translate(meta_data['rigid_pos'])

        # rigid_mesh_temp = copy.deepcopy(self.rigid_mesh)
        # rigid_mesh_temp.translate(meta_data['rigid_pos'])
        # rigid_mesh_temp.rotate(R)

        # rigid_mesh.translate(meta_data['tip_collision_position'])
        # rigid_mesh.translate(meta_data['collision_position'])

        # head = o3d.geometry.TriangleMesh.create_sphere(radius=1)
        # head.translate(meta_data['deformer_origin'])
        # tip = o3d.geometry.TriangleMesh.create_sphere(radius=1)
        # tip.translate(meta_data['tip_collision_position'])
        rigid_mesh= self.create_cylinder_between_points(meta_data['deformer_origin'],meta_data['tip_collision_position'])
        
        
        
        soft_def_mesh = _load_deformed_mesh(sample_path, meta_data,self.root_dir)
        soft_rest_mesh = copy.deepcopy(self.soft_rest_mesh)
        
        # soft_rest_mesh.translate(meta_data['object_rigid_pos'].detach().cpu().numpy())
        sampled_soft_rest_mesh_o3d, sampled_soft_def_mesh_o3d = _sample_nearest(rigid_mesh, soft_def_mesh, soft_rest_mesh,self.n_points)
        max_bound = np.maximum(np.max(np.abs(sampled_soft_def_mesh_o3d.get_min_bound())),np.max(np.abs(sampled_soft_def_mesh_o3d.get_max_bound())))
        # scale_ratio = 1/max_bound
        center = np.array([0.0, 0.0, 0.0])
        sampled_soft_rest_mesh_o3d.scale(1/max_bound,center)
        sampled_soft_def_mesh_o3d.scale(1/max_bound,center)
        rigid_mesh.scale(1/max_bound,center)


        rigid_graph = mesh_to_graph(rigid_mesh)
        rigid_graph.x = _feature_rigid(meta_data, rigid_graph.x)
        soft_rest_graph = mesh_to_graph(sampled_soft_rest_mesh_o3d)
        soft_def_graph = mesh_to_graph(sampled_soft_def_mesh_o3d)
        return obj_name, soft_rest_graph, soft_def_graph, meta_data, rigid_graph

    def _read_meta_data(self, sample_path):
        with open(os.path.join(self.root_dir, sample_path + ".json"), "r") as f:
            json_data = json.load(f)[0]

        force_vector = torch.tensor(_unity_to_open3d([
            json_data['forceDirectionX'],
            json_data['forceDirectionY'],
            json_data['forceDirectionZ'],
        ]), dtype=torch.float32)

        collision_position = torch.tensor(_unity_to_open3d([
            json_data['collisionPositionX'],
            json_data['collisionPositionY'],
            json_data['collisionPositionZ']
        ]), dtype=torch.float32)

        rigid_rot = torch.tensor(_unity_to_open3d([
            json_data['deformerInitialRotationX'],
            json_data['deformerInitialRotationY'],
            json_data['deformerInitialRotationZ']
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
        
        
        deformer_tip_collision_position = torch.tensor(_unity_to_open3d([
            json_data['deformerTipCollisionPositionX'],
            json_data['deformerTipCollisionPositionY'],
            json_data['deformerTipCollisionPositionZ']
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
            'collision_position': collision_position,
            'object_rigid_pos':object_rigid_pos,
            'object_rigid_rot':rigid_rot,
            'tip_collision_position':deformer_tip_collision_position
        }
