import torch
from torch.utils.data import DataLoader
from dataloaders.everyday_deform_v2 import EverydayDeformDataset
from torch_geometric.data import Batch
from utils.visualization import *
from dataloaders.collate import collate_fn
from configs.config import Config
from models.model import GraphNet
from models.losses import GradientConsistencyLoss
import torch.nn as nn
import os
import json
import wandb
import yaml 
from utils.graph_utils import *

if __name__ == "__main__":
    run_id = input("Enter the wandb run ID (e.g. mahdi-slh/GeoForce-src/runs/0pbafsok): ")
    model_path = "model_weights.pth"
    config_path = "config.yaml"
    log_dir = f'./wandb/{run_id}/logs'
    
    # Download model weights and config using wandb API
    run = wandb.Api().run(run_id)
    run.file(model_path).download(replace=True, root=log_dir)
    run.file(config_path).download(replace=True, root=log_dir)
    
    # Load configuration from saved file
    with open(os.path.join(log_dir, config_path), 'r') as f:
        saved_config = yaml.safe_load(f)
    
    config = Config()  # Initialize a default config
    for key, value in saved_config.items():
        setattr(config, key, value)  # Override with saved values

    # Load dataset and dataloader
    val_dataset = EverydayDeformDataset(root_dir=config.dataset.root_dir, 
        obj_list=config.dataset.obj_list, 
        n_points=config.dataset.n_points,
        graph_method=config.dataset.graph_method,
        radius=config.dataset.radius,
        k=config.dataset.k, split='val')

    dataloader_val = DataLoader(
        val_dataset, 
        batch_size=config.dataloader.batch_size, 
        shuffle=config.dataloader.shuffle,
        collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphNet(input_dims=config.network.input_dims,
                    hidden_dim=config.network.hidden_dim,
                    output_dim=config.network.output_dim,
                    num_layers=config.network.num_layers,
                    dropout_rate=config.network.dropout_rate,
                    knn_k=config.network.knn_k,
                    backbone=config.network.backbone).to(device)
    
    model.load_state_dict(torch.load(os.path.join(log_dir, 'model_weights.pth')))

    model.eval()

    criterion_mse = nn.MSELoss()
    criterion_grad = GradientConsistencyLoss()
    lambda_gradient = config.training.lambda_gradient

    total_loss = 0.0
    with torch.no_grad():  # disable gradient computation during evaluation
        for batch_idx, (obj_name, rest_graphs, def_graphs, meta_data, collider_graphs) in enumerate(dataloader_val):
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
            for indx in range(config.dataloader.batch_size):
                # deformation_per_node = compute_deformation_using_diff_coords(rest_graphs[indx],  def_graphs_batched[indx])
                # print(deformation_per_node)

                visualize_deformations_intensity(rest_graphs[indx], def_graphs_batched[indx],meta_data['deform_intensity'][indx])
                visualize_deformation_field(rest_graphs[indx].pos.cpu(), predictions[indx].pos.cpu(), meta_data['deformer_collision_position'][indx], meta_data['deformer_origin'][indx])
                visualize_merged_graphs(rest_graphs[indx], def_graphs_batched[indx], collider_graphs[indx],predictions[indx])
                

        avg_loss = total_loss / len(dataloader_val)
        print(f"Average Loss on Eval Dataset: {avg_loss}")
