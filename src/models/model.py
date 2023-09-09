import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, TAGConv, knn
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean,scatter_max


class MultiHeadAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([nn.Linear(feature_dim, feature_dim) for _ in range(num_heads)])
        
    def forward(self, x_resting, x_rigid):
        outputs = []
        for head in self.attention_heads:
            scores = torch.mm(head(x_resting), head(x_rigid).transpose(0, 1))
            attn_weights = F.softmax(scores, dim=-1)
            output = torch.mm(attn_weights, x_rigid)
            outputs.append(output)
        
        # Concatenate outputs from all attention heads
        return torch.cat(outputs, dim=-1)
    
class GraphNet(nn.Module):
    def __init__(self, input_dims, hidden_dim, output_dim, encoder_layers, decoder_layers, dropout_rate, knn_k, backbone, num_heads=4):
        super(GraphNet, self).__init__()

        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers  # Specify the number of decoder layers
        self.backbone = backbone

        # Initialize layer lists for resting and rigid graphs
        self.conv_layers_resting = nn.ModuleList()
        self.conv_layers_rigid = nn.ModuleList()

        self.dropout_rate = dropout_rate
        self.knn_k = knn_k

        # Choose the appropriate convolution layer based on the selected backbone
        conv_layer = GATConv if self.backbone == "GATConv" else GCNConv if self.backbone == "GCNConv" else TAGConv

        # Make a copy of input_dims to avoid overwriting values
        input_dims_resting = input_dims.copy()
        input_dims_rigid = input_dims.copy()

        # Layers for resting graph
        for _ in range(self.encoder_layers):
            self.conv_layers_resting.append(conv_layer(input_dims_resting[0], hidden_dim))
            input_dims_resting[0] = hidden_dim  # Update input dimensions for subsequent layers

        # Layers for rigid graph
        for _ in range(self.encoder_layers):
            self.conv_layers_rigid.append(conv_layer(input_dims_rigid[1], hidden_dim))
            input_dims_rigid[1] = hidden_dim  # Update input dimensions for subsequent layers

        # Decoder
        decoder = []
        input_dim_decoder = hidden_dim * (num_heads+1)
        for _ in range(self.decoder_layers):
            decoder.append(nn.Linear(input_dim_decoder, hidden_dim))
            # decoder.append(nn.BatchNorm1d(hidden_dim))  # Add BatchNorm
            decoder.append(nn.ReLU())  # Add activation function (e.g., ReLU)
            decoder.append(nn.Dropout(self.dropout_rate))
            input_dim_decoder = hidden_dim
        decoder.append(nn.Linear(hidden_dim, output_dim))
        self.decoder = nn.Sequential(*decoder)
        self.multihead_attention = MultiHeadAttention(hidden_dim, num_heads=num_heads)

    def forward(self, graph_resting, graph_rigid):
        # For resting graph
        x_resting = graph_resting.x
        for conv in self.conv_layers_resting:
            x_resting = F.relu(conv(x_resting, graph_resting.edge_index))
            x_resting = F.dropout(x_resting, p=self.dropout_rate, training=self.training)

        # For rigid graph
        x_rigid = graph_rigid.x
        for conv in self.conv_layers_rigid:
            x_rigid = F.relu(conv(x_rigid, graph_rigid.edge_index))
            x_rigid = F.dropout(x_rigid, p=self.dropout_rate, training=self.training)

        # Using KNN to establish interaction between x_resting and x_rigid
        # edge_index = knn(x_resting, x_rigid, k=self.knn_k)
        # col, row = edge_index

        # Use scatter_mean for aggregation of features
        # pooled_features = scatter_mean(x_rigid[col], row, dim=0, dim_size=x_resting.size(0))
        pooled_features = self.multihead_attention(x_resting, x_rigid)


        # Concatenate original x_resting features with pooled features from x_rigid
        # x_combined = torch.cat([x_resting, pooled_features], dim=-1)
        x_combined = torch.cat([x_resting, pooled_features], dim=-1)


        # Pass through the decoder to get the deformed positions
        x_out = self.decoder(x_combined)

        # Building a graph with deformed positions
        deformed_graph = graph_resting.clone()
        deformed_graph.pos = x_out

        return deformed_graph



# class GraphNet(nn.Module):
#     def __init__(self, input_dims, hidden_dim, output_dim, encoder_layers=2, dropout_rate=0.5, knn_k=3, backbone="GATConv"):
#         super(GraphNet, self).__init__()

#         self.encoder_layers = encoder_layers
#         self.backbone = backbone

#         # Initialize layer lists for resting and rigid graphs
#         self.conv_layers_resting = nn.ModuleList()
#         self.conv_layers_rigid = nn.ModuleList()

#         self.dropout_rate = dropout_rate
#         self.knn_k = knn_k

#         # Choose the appropriate convolution layer based on the selected backbone
#         conv_layer = GATConv if self.backbone == "GATConv" else GCNConv if self.backbone == "GCNConv" else TAGConv

#         # Layers for resting graph
#         for _ in range(self.encoder_layers):
#             self.conv_layers_resting.append(conv_layer(input_dims[0], hidden_dim))
#             input_dims[0] = hidden_dim  # Update input dimensions for subsequent layers

#         # Layers for rigid graph
#         for _ in range(self.encoder_layers):
#             self.conv_layers_rigid.append(conv_layer(input_dims[1], hidden_dim))
#             input_dims[1] = hidden_dim  # Update input dimensions for subsequent layers

#         # Decoder
#         self.decoder = nn.Linear(hidden_dim * 2, output_dim)

#     def add_noise(self,data, noise_factor=0.5):
#         noisy_data = data + noise_factor * torch.randn(data.size()).to(data.device)
#         return noisy_data


#     def forward(self, graph_resting, graph_rigid, noise_factor=0.2):
#         # Inject noise
#         graph_resting_noisy = graph_resting.clone()
#         graph_resting_noisy.x = self.add_noise(graph_resting.x, noise_factor)
        
#         graph_rigid_noisy = graph_rigid.clone()
#         graph_rigid_noisy.x = self.add_noise(graph_rigid.x, noise_factor)
#         # For resting graph
#         x_resting = graph_resting_noisy.x
#         for conv in self.conv_layers_resting:
#             x_resting = F.relu(conv(x_resting, graph_resting.edge_index))
#             x_resting = F.dropout(x_resting, p=self.dropout_rate, training=self.training)
        
#         # For rigid graph
#         x_rigid = graph_rigid_noisy.x
#         for conv in self.conv_layers_rigid:
#             x_rigid = F.relu(conv(x_rigid, graph_rigid.edge_index))
#             x_rigid = F.dropout(x_rigid, p=self.dropout_rate, training=self.training)
        
#         # Using KNN to establish interaction between x_resting and x_rigid
#         edge_index = knn(x_resting, x_rigid, k=self.knn_k)
#         col, row = edge_index

#         # Use scatter_mean for aggregation of features
#         pooled_features = scatter_mean(x_rigid[col], row, dim=0, dim_size=x_resting.size(0))
        
#         # Concatenate original x_resting features with pooled features from x_rigid
#         x_combined = torch.cat([x_resting, pooled_features], dim=-1)
        
#         # Pass through the decoder to get the deformed positions
#         x_out = self.decoder(x_combined)
        
#         # Building a graph with deformed positions
#         deformed_graph = graph_resting.clone()
#         deformed_graph.pos += x_out

#         return deformed_graph
