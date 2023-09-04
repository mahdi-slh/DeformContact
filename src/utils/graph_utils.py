import torch

def compute_differential_coordinates(graph):
    """
    Compute the differential coordinates for a graph's nodes.
    """
    # Create an index for the start of the edges for each node
    idx = torch.zeros(graph.pos.size(0), dtype=torch.long)
    idx[graph.edge_index[0]] = torch.arange(graph.edge_index.size(1))

    # Compute the average of the neighbors' positions
    avg_neighbors = torch.zeros_like(graph.pos)
    count_neighbors = torch.zeros(graph.pos.size(0), 1)
    avg_neighbors.index_add_(0, graph.edge_index[1], graph.pos[graph.edge_index[0]])
    count_neighbors.index_add_(0, graph.edge_index[1], torch.ones(graph.edge_index[0].size(0), 1))
    avg_neighbors /= count_neighbors

    # Compute differential coordinate
    diff_coords = graph.pos - avg_neighbors
    return diff_coords

def compute_deformation_using_diff_coords(soft_rest_graph, soft_def_graph):
    """
    Compute deformation values for each node in the graph using differential coordinates.
    """
    soft_rest_diff_coords = compute_differential_coordinates(soft_rest_graph)
    soft_def_diff_coords = compute_differential_coordinates(soft_def_graph)
    
    # Compute deformation values based on differential coordinates difference
    deformation_values = torch.norm(soft_rest_diff_coords - soft_def_diff_coords, dim=1)

    # Create an index for the start of the edges for each node
    idx = torch.zeros(soft_rest_graph.pos.size(0), dtype=torch.long)
    idx[soft_rest_graph.edge_index[0]] = torch.arange(soft_rest_graph.edge_index.size(1))

    # To get the deformation in the vicinity, average over neighbors
    adjusted_deformation_values = torch.zeros_like(deformation_values)
    count_neighbors = torch.zeros(soft_rest_graph.pos.size(0))
    adjusted_deformation_values.index_add_(0, soft_rest_graph.edge_index[1], deformation_values[soft_rest_graph.edge_index[0]])
    count_neighbors.index_add_(0, soft_rest_graph.edge_index[1], torch.ones(soft_rest_graph.edge_index[0].size(0)))
    adjusted_deformation_values /= count_neighbors

    # Normalize deformation values between 0 and 1
    min_val = adjusted_deformation_values.min()
    max_val = adjusted_deformation_values.max()
    normalized_deformation_values = (adjusted_deformation_values - min_val) / (max_val - min_val)

    return normalized_deformation_values
