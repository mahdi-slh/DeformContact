import torch.nn as nn
from torch_geometric.nn import GATConv, knn
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean  # Added import for scatter_mean

class GraphNet(nn.Module):
    def __init__(self, input_dims, hidden_dim, output_dim):
        super(GraphNet, self).__init__()

        # Layers for resting graph
        self.conv1_resting = GATConv(input_dims[0], hidden_dim)
        self.conv2_resting = GATConv(hidden_dim, hidden_dim)
        
        # Layers for collider graph
        self.conv1_collider = GATConv(input_dims[1], hidden_dim)
        self.conv2_collider = GATConv(hidden_dim, hidden_dim)

        # Decoder
        self.decoder = nn.Linear(hidden_dim * 2, output_dim)  # *2 because we will concatenate features

    def forward(self, graph_resting, graph_collider):
        # For resting graph
        x_resting = F.relu(self.conv1_resting(graph_resting.x, graph_resting.edge_index))
        x_resting = F.dropout(x_resting, training=self.training)
        x_resting = F.relu(self.conv2_resting(x_resting, graph_resting.edge_index))
        
        # For collider graph
        x_collider = F.relu(self.conv1_collider(graph_collider.x, graph_collider.edge_index))
        x_collider = F.dropout(x_collider, training=self.training)
        x_collider = F.relu(self.conv2_collider(x_collider, graph_collider.edge_index))
        
        # Using KNN to establish interaction between x_resting and x_collider
        edge_index = knn(x_resting, x_collider, k=3)
        col, row = edge_index

        # Use scatter_mean for aggregation of features
        pooled_features = scatter_mean(x_collider[col], row, dim=0, dim_size=x_resting.size(0))
        
        # Concatenate original x_resting features with pooled features from x_collider
        x_combined = torch.cat([x_resting, pooled_features], dim=-1)
        
        # Pass through the decoder to get the deformed positions
        x_out = self.decoder(x_combined)
        
        # Building a graph with deformed positions
        deformed_graph = graph_resting.clone()
        deformed_graph.pos = x_out

        return deformed_graph
