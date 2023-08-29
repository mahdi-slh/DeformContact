import open3d as o3d
import numpy as np
from scipy import spatial
import torch_geometric.nn as pyg_nn


def fps_points(points_np, num_samples, return_indices=False):

    seeds = [np.array([0.0, 0.0, 0.0])]
    nn_index = spatial.cKDTree(seeds)
    dists, _ = nn_index.query(points_np, k=1)
    inds = []
    for i in range(num_samples):
        new_idx = np.argmax(dists)
        sample = points_np[new_idx]
        seeds.append(points_np[new_idx])

        dists[new_idx] = -1
        dists = np.minimum(dists, np.linalg.norm(points_np - sample, axis=1))

        inds.append(new_idx)
    seeds.pop(0)
    seeds = np.array(seeds)

    if return_indices:
        return seeds, inds
    else:
        return seeds


def fps_points_random(points_np, num_samples, return_indices=False):

    rand_seeds = np.random.choice(points_np.shape[0], num_samples // 2)
    seeds = [points_np[id, :] for id in rand_seeds]
    nn_index = spatial.cKDTree(seeds)
    dists, _ = nn_index.query(points_np, k=1)
    inds = list(rand_seeds)
    for i in range(num_samples // 2):
        new_idx = np.argmax(dists)
        sample = points_np[new_idx]
        seeds.append(points_np[new_idx])

        dists[new_idx] = -1
        dists = np.minimum(dists, np.linalg.norm(points_np - sample, axis=1))

        inds.append(new_idx)
    # seeds.pop(0)
    seeds = np.array(seeds)

    if return_indices:
        return seeds, inds
    else:
        return seeds

def construct_graph(point_cloud, k=None, radius=None):
    """Construct a graph from a given point cloud using PyTorch Geometric.
    
    Args:
    - point_cloud (torch.Tensor): Tensor of shape [num_points, num_dims] representing the point cloud.
    - k (int): Number of neighbors for KNN graph construction. Default is 5.
    - radius (float, optional): Radius value for radius ball graph construction. If provided, the function will
                                return a radius ball graph instead of a KNN graph.
    
    Returns:
    - edge_index (torch.Tensor): Tensor of shape [2, num_edges] defining the edge relations.
    """
    if radius is not None:
        edge_index = pyg_nn.radius_graph(point_cloud, r=radius, batch=None, loop=False)
    else:
        edge_index = pyg_nn.knn_graph(point_cloud, k=k, batch=None, loop=False)
    return edge_index

