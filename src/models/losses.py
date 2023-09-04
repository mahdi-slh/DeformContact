import torch.nn as nn

class GradientConsistencyLoss(nn.Module):
    def __init__(self):
        super(GradientConsistencyLoss, self).__init__()

    def forward(self, pred_graphs_batched, soft_rest_graphs_batched):
        node_diffs = pred_graphs_batched.pos - soft_rest_graphs_batched.pos
        node_pos_rest = soft_rest_graphs_batched.pos
        node_pos_pred = pred_graphs_batched.pos

        edge_diffs_rest = node_pos_rest[soft_rest_graphs_batched.edge_index[1]] - node_pos_rest[soft_rest_graphs_batched.edge_index[0]]
        edge_diffs_pred = node_pos_pred[pred_graphs_batched.edge_index[1]] - node_pos_pred[pred_graphs_batched.edge_index[0]]

        cross_shape_diffs = edge_diffs_rest - edge_diffs_pred

        loss = cross_shape_diffs.norm(p=2, dim=-1).sum() / len(soft_rest_graphs_batched.edge_index[0])  # Scale by the number of edges

        return loss
    

class DeformableDistance(nn.Module):
    def __init__(self):
        super(DeformableDistance, self).__init__()

    def forward(self, pred_graphs_batched, soft_rest_graphs_batched, deform_intensity):
        # Compute the distance between nodes in the predicted and rest graphs
        node_diffs = (pred_graphs_batched.pos - soft_rest_graphs_batched.pos).norm(p=2, dim=-1)
        
        # Normalize the deformation intensities to [0, 1]
        min_int = deform_intensity.min()
        max_int = deform_intensity.max()
        normalized_intensity = (deform_intensity - min_int) / (max_int - min_int + 1e-9)
        
        # Add a small constant to ensure every region still has some influence
        weights = normalized_intensity + 0.01
        
        # Weight the node differences by the normalized deformation intensity
        weighted_diffs = node_diffs * weights.view(node_diffs.shape)
        
        # Compute the average loss across all nodes
        loss = weighted_diffs.sum() / len(soft_rest_graphs_batched.pos)
        
        return loss