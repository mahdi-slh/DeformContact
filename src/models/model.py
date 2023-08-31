import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, TAGConv, knn
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean

class GraphNet(nn.Module):
    def __init__(self, input_dims, hidden_dim, output_dim, num_layers=2, dropout_rate=0.5, knn_k=3, backbone="GATConv"):
        super(GraphNet, self).__init__()

        self.num_layers = num_layers
        self.backbone = backbone

        # Initialize layer lists for resting and collider graphs
        self.conv_layers_resting = nn.ModuleList()
        self.conv_layers_collider = nn.ModuleList()

        self.dropout_rate = dropout_rate
        self.knn_k = knn_k

        # Choose the appropriate convolution layer based on the selected backbone
        conv_layer = GATConv if self.backbone == "GATConv" else GCNConv if self.backbone == "GCNConv" else TAGConv

        # Layers for resting graph
        for _ in range(self.num_layers):
            self.conv_layers_resting.append(conv_layer(input_dims[0], hidden_dim))
            input_dims[0] = hidden_dim  # Update input dimensions for subsequent layers

        # Layers for collider graph
        for _ in range(self.num_layers):
            self.conv_layers_collider.append(conv_layer(input_dims[1], hidden_dim))
            input_dims[1] = hidden_dim  # Update input dimensions for subsequent layers

        # Decoder
        self.decoder = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, graph_resting, graph_collider):
        # For resting graph
        x_resting = graph_resting.x
        for conv in self.conv_layers_resting:
            x_resting = F.relu(conv(x_resting, graph_resting.edge_index))
            x_resting = F.dropout(x_resting, p=self.dropout_rate, training=self.training)
        
        # For collider graph
        x_collider = graph_collider.x
        for conv in self.conv_layers_collider:
            x_collider = F.relu(conv(x_collider, graph_collider.edge_index))
            x_collider = F.dropout(x_collider, p=self.dropout_rate, training=self.training)
        
        # Using KNN to establish interaction between x_resting and x_collider
        edge_index = knn(x_resting, x_collider, k=self.knn_k)
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
