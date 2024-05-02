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
    
