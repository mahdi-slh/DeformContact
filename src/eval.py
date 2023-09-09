import torch

from torch_geometric.data import Batch
from utils.visualization import *
from configs.config import Config
from models.model_loader import load_model
from models.losses import GradientConsistencyLoss
from data.dataset_loader import load_dataset
import wandb
import json
import os
import torch.nn as nn
from utils.graph_utils import *


def eval():

    run_id = input("Enter the wandb run ID (e.g. mahdi-slh/GeoContact/runs/0pbafsok): ")
    model_path = "model_weights.pth"
    config_path = "retina.json"

    config = Config("configs/retina.json")
    
    log_dir = f'./wandb/{run_id}/logs'
    
    run = wandb.Api().run(run_id)
    run.file(model_path).download(replace=True, root=log_dir)
    # run.file(config_path).download(replace=True, root=log_dir)
    
    # with open(os.path.join(log_dir, config_path), 'r') as f:
    #     saved_config = json.load(f) 
    
    # config = Config(saved_config)

    _,dataloader_val = load_dataset(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model= load_model(config).to(device)
    
    model.load_state_dict(torch.load(os.path.join(log_dir, 'model_weights.pth')))

    model.eval()

    criterion_mse = nn.MSELoss()
    criterion_grad = GradientConsistencyLoss()
    lambda_gradient = config.training.lambda_gradient

    total_loss = 0.0
    with torch.no_grad():  # disable gradient computation during evaluation
        for batch_idx, (obj_name, soft_rest_graphs, soft_def_graphs, meta_data, rigid_graphs) in enumerate(dataloader_val):
            soft_rest_graphs_batched = Batch.from_data_list(soft_rest_graphs).to(device)
            rigid_graphs_batched = Batch.from_data_list(rigid_graphs).to(device)
            soft_def_graphs_batched = Batch.from_data_list(soft_def_graphs).to(device)

            # Get model predictions
            predictions = model(soft_rest_graphs_batched, rigid_graphs_batched)

            # Compute the loss
            loss_mse = criterion_mse(predictions.pos, soft_def_graphs_batched.pos)
            loss_consistency = criterion_grad(predictions, soft_def_graphs_batched)
            loss = loss_mse + lambda_gradient * loss_consistency
            total_loss += loss.item()

            # Visualization
            for indx in range(config.dataloader.batch_size):
                visualize_deformations_normals_colors(soft_rest_graphs[indx], soft_def_graphs_batched[indx])
                visualize_deformation_field(soft_rest_graphs[indx].pos.cpu(), predictions[indx].pos.cpu(),rigid_graphs[indx].pos.cpu(), meta_data['deformer_collision_position'][indx], meta_data['deformer_origin'][indx])
                visualize_merged_graphs(soft_rest_graphs[indx], soft_def_graphs_batched[indx], rigid_graphs[indx],predictions[indx])
                

        avg_loss = total_loss / len(dataloader_val)
        print(f"Average Loss on Eval Dataset: {avg_loss}")

        
if __name__ == "__main__":
    
    eval()