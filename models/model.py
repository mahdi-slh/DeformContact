import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, TAGConv, knn
import torch
import torch.nn.functional as F


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

        return torch.cat(outputs, dim=-1)
    
class GraphNet(nn.Module):
    def __init__(self, input_dims, hidden_dim, output_dim, encoder_layers, decoder_layers, dropout_rate, knn_k, backbone,use_mha, num_mha_heads,mode):
        super(GraphNet, self).__init__()

        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.backbone = backbone

        self.conv_layers_resting = nn.ModuleList()
        self.conv_layers_rigid = nn.ModuleList()
        self.use_mha = use_mha

        self.dropout_rate = dropout_rate
        self.knn_k = knn_k
        self.mode = mode

        conv_layer = GATConv if self.backbone == "GATConv" else GCNConv if self.backbone == "GCNConv" else TAGConv

        input_dims_resting = input_dims.copy()
        input_dims_rigid = input_dims.copy()

        for _ in range(self.encoder_layers):
            self.conv_layers_resting.append(conv_layer(input_dims_resting[0], hidden_dim))
            input_dims_resting[0] = hidden_dim 

        for _ in range(self.encoder_layers):
            self.conv_layers_rigid.append(conv_layer(input_dims_rigid[1], hidden_dim))
            input_dims_rigid[1] = hidden_dim  

        # Decoder
        decoder = []
        if self.use_mha:
            input_dim_decoder = hidden_dim * (num_mha_heads+1)
        else:
            input_dim_decoder = hidden_dim *2
        for _ in range(self.decoder_layers):
            decoder.append(nn.Linear(input_dim_decoder, hidden_dim))
            decoder.append(nn.ReLU())  
            decoder.append(nn.Dropout(self.dropout_rate))
            input_dim_decoder = hidden_dim
        decoder.append(nn.Linear(hidden_dim, output_dim))
        self.decoder = nn.Sequential(*decoder)
        self.multihead_attention = MultiHeadAttention(hidden_dim, num_heads=num_mha_heads)

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

        

        pooled_features = self.multihead_attention(x_resting, x_rigid)

        x_combined = torch.cat([x_resting, pooled_features], dim=-1)


        # Pass through the decoder to get the deformed positions
        x_out = self.decoder(x_combined)

        # Building a graph with deformed positions
        deformed_graph = graph_resting.clone()
        if  self.mode == "res":
            deformed_graph.pos += x_out
        elif self.mode == "rec":
            deformed_graph.pos = x_out

        return deformed_graph
