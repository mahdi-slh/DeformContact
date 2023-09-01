import torch
from torch.utils.data import DataLoader
from dataloaders.everyday_deform_v2 import EverydayDeformDataset
from torch_geometric.data import Batch
from utils.visualization import *
from dataloaders.collate import collate_fn
from configs.config import Config
from models.model import GraphNet
from models.loss import GradientConsistencyLoss
import torch.nn as nn
import os
import json

if __name__ == "__main__":
    log_dir = input("Enter the path of the log directory (e.g.2023-08-30_14-30-45/): ")

    # Load configuration from saved file
    with open(os.path.join('logs/',log_dir, 'config.json'), 'r') as f:
        saved_config = json.load(f)
    
    config = Config()  # Initialize a default config
    for key, value in saved_config.items():
        setattr(config, key, value)  # Override with saved values


    # Load dataset and dataloader
    val_dataset = EverydayDeformDataset(root_dir=config.dataset["root_dir"], 
        obj_list=config.dataset["obj_list"], 
        n_points=config.dataset["n_points"],
        graph_method=config.dataset["graph_method"],
        radius=config.dataset["radius"],
        k=config.dataset["k"], split='val')

    dataloader = DataLoader(
        val_dataset,
        batch_size=config.dataloader["batch_size"],
        shuffle=False,  # typically, you don't shuffle in evaluation
        collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize and load model
    model = GraphNet(input_dims=config.network["input_dims"],
                     hidden_dim=config.network["hidden_dim"],
                     output_dim=config.network["output_dim"],
                     num_layers=config.network["num_layers"],  # Add the num_layers parameter
                     dropout_rate=config.network["dropout_rate"],  # Add the dropout_rate parameter
                     knn_k=config.network["knn_k"],  # Add the knn_k parameter
                     backbone=config.network["backbone"]).to(device)  # Add the backbone parameter
    model.load_state_dict(torch.load(os.path.join('logs',log_dir, 'model_weights.pth')))
    model.eval()

    criterion_mse = nn.MSELoss()
    criterion_grad = GradientConsistencyLoss()
    lambda_gradient = config.training["lambda_gradient"]

    total_loss = 0.0
    with torch.no_grad():  # disable gradient computation during evaluation
        for batch_idx, (obj_name, rest_graphs, def_graphs, meta_data, collider_graphs) in enumerate(dataloader):
            rest_graphs_batched = Batch.from_data_list(rest_graphs).to(device)
            collider_graphs_batched = Batch.from_data_list(collider_graphs).to(device)
            def_graphs_batched = Batch.from_data_list(def_graphs).to(device)

            # Get model predictions
            predictions = model(rest_graphs_batched, collider_graphs_batched)

            # Compute the loss
            loss_mse = criterion_mse(predictions.pos, def_graphs_batched.pos)
            loss_consistency = criterion_grad(predictions, def_graphs_batched)
            loss = loss_mse + lambda_gradient * loss_consistency
            total_loss += loss.item()

            # Visualization
            visualize_deformation_field(rest_graphs[0].pos.cpu(), predictions[0].pos.cpu(), meta_data['deformer_collision_position'][0], meta_data['deformer_origin'][0])
            visualize_merged_graphs(rest_graphs[0], def_graphs_batched[0], collider_graphs[0],predictions[0])

        avg_loss = total_loss / len(dataloader)
        print(f"Average Loss on Eval Dataset: {avg_loss}")
