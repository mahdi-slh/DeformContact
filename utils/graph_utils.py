import torch
from torch_geometric.data import Data
import numpy as np
from utils.pos_encoding import to_log_freq


def mesh_to_graph(mesh,encode=True):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    vertices_t = torch.tensor(vertices, dtype=torch.float32)

    edge_indices = [[triangle[i], triangle[(i + 1) % 3]] for triangle in triangles for i in range(3)]
    edge_indices_t = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    if encode:
        feat = to_log_freq(vertices_t, 3, 1)
    else:
        feat = vertices_t

    return Data(x=feat, edge_index=edge_indices_t, pos=vertices_t)
