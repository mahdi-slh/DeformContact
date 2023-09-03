import torch

def compute_differential_coordinates(graph):
    """
    Compute the differential coordinates for a graph's nodes.
    """
    diff_coords = []
    
    # Iterate over each node
    for i in range(graph.x.size(0)):
        # Neighbors of node i
        neighbors = graph.edge_index[:, graph.edge_index[0] == i][1]
        
        # Compute the average of the neighbors' positions
        avg_neighbors = graph.x[neighbors].mean(dim=0)
        
        # Compute differential coordinate
        diff_coord = graph.x[i] - avg_neighbors
        diff_coords.append(diff_coord)
        
    return torch.stack(diff_coords, dim=0)

def compute_deformation_using_diff_coords(rest_graph, def_graph):
    """
    Compute deformation values for each node in the graph using differential coordinates.
    """
    rest_diff_coords = compute_differential_coordinates(rest_graph)
    def_diff_coords = compute_differential_coordinates(def_graph)
    
    # Compute deformation values based on differential coordinates difference
    deformation_values = torch.norm(rest_diff_coords - def_diff_coords, dim=1)
    
    # Create a container for adjusted deformation values
    adjusted_deformation_values = torch.zeros_like(deformation_values)

    # To get the deformation in the vicinity, average over neighbors
    for i in range(rest_graph.x.size(0)):
        # Neighbors of node i
        neighbors = rest_graph.edge_index[:, rest_graph.edge_index[0] == i][1]
        adjusted_deformation_values[i] = deformation_values[neighbors].mean()

    # Normalize deformation values between 0 and 1
    min_val = adjusted_deformation_values.min()
    max_val = adjusted_deformation_values.max()
    normalized_deformation_values = (adjusted_deformation_values - min_val) / (max_val - min_val)

    return normalized_deformation_values