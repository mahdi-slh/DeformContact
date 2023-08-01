import torch
from torch.utils.data import Dataset
import open3d as o3d
import numpy as np
import json
import os

class EverydayDeformDataset(Dataset):
    
    def __init__(self, object_classes, root_dir):
        self.root_dir = root_dir

        # Loop over the object classes and collect all deformation samples
        self.samples = []
        for obj_class in object_classes:
            # Get all rest files for this object class
            rest_files = [f for f in os.listdir(os.path.join(root_dir,'deformations', obj_class)) if f.endswith('_rest.obj')]

            for rest_file in rest_files:
                # Get the sample id by removing the '_rest.obj' suffix
                sample_id = rest_file[:-9]

                # Add a tuple with the object class and sample id to the samples list
                self.samples.append((obj_class, sample_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obj_class, sample_id = self.samples[idx]

        # Load the resting and deformed meshes
        rest_mesh = o3d.io.read_triangle_mesh(os.path.join(self.root_dir,'deformations', obj_class, sample_id + "_rest.obj"))
        def_mesh = o3d.io.read_triangle_mesh(os.path.join(self.root_dir,'deformations', obj_class, sample_id + "_def.obj"))

        rest_mesh_t = torch.tensor(np.asarray(rest_mesh.vertices), dtype=torch.float32)
        def_mesh_t = torch.tensor(np.asarray(def_mesh.vertices), dtype=torch.float32)
        

        # Load the meta data
        with open(os.path.join(self.root_dir,'deformations', obj_class, sample_id + ".json"), "r") as f:
            meta_data = json.load(f)

        # Unpack the nested metadata and convert it to tensors
        contact_event = meta_data['contact_event']
        contact_position = torch.tensor([contact_event['contact_position'][axis] for axis in ('x', 'y', 'z')], dtype=torch.float32)
        contact_type = contact_event['contact_type']
        timestamp = torch.tensor(contact_event['timestamp'], dtype=torch.float32)
        force_vector = torch.tensor([contact_event['force_vector'][axis] for axis in ('fx', 'fy', 'fz')], dtype=torch.float32)
        meta_data = {'contact_position': contact_position, 'contact_type': contact_type, 'timestamp': timestamp, 'force_vector': force_vector}

        return rest_mesh_t, def_mesh_t, meta_data
