import open3d as o3d
import numpy as np
from scipy import spatial
import torch_geometric.nn as pyg_nn


def construct_graph(point_cloud, k=None, radius=None):

    if radius is not None:
        edge_index = pyg_nn.radius_graph(point_cloud, r=radius, batch=None, loop=False)
    else:
        edge_index = pyg_nn.knn_graph(point_cloud, k=k, batch=None, loop=False)
    return edge_index

