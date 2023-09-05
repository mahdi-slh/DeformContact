import torch
import os
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloaders.everyday_deform import EverydayDeformDataset
from torch_geometric.data import Batch
from dataloaders.collate import collate_fn
from configs.config import Config
from models.model import GraphNet
from models.losses import GradientConsistencyLoss,DeformableDistance

import wandb
import yaml

def train(config):

    train_dataset = EverydayDeformDataset(root_dir=config.dataset.root_dir, 
        obj_list=config.dataset.obj_list, 
        n_points=config.dataset.n_points,
        graph_method=config.dataset.graph_method,
        radius=config.dataset.radius,
        k=config.dataset.k, split='train')
    
    val_dataset = EverydayDeformDataset(root_dir=config.dataset.root_dir, 
        obj_list=config.dataset.obj_list, 
        n_points=config.dataset.n_points,
        graph_method=config.dataset.graph_method,
        radius=config.dataset.radius,
        k=config.dataset.k, split='val')

    dataloader_train = DataLoader(
    train_dataset, 
    batch_size=config.dataloader.batch_size, 
    shuffle=config.dataloader.shuffle, 
    collate_fn=collate_fn
    )

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
                    encoder_layers=config.network.encoder_layers,
                    decoder_layers=config.network.decoder_layers,
                    dropout_rate=config.network.dropout_rate,
                    knn_k=config.network.knn_k,
                    backbone=config.network.backbone).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
    criterion_mse = nn.MSELoss()
    criterion_grad = GradientConsistencyLoss()
    criterion_def = DeformableDistance()
    lambda_gradient = config.training.lambda_gradient

    for epoch in range(config.training.n_epochs):
        total_tr_loss = 0
        model.train()
        for batch_idx, (obj_name, soft_rest_graphs, soft_def_graphs, meta_data, rigid_graphs) in enumerate(dataloader_train):
            soft_rest_graphs_batched = Batch.from_data_list(soft_rest_graphs)
            rigid_graphs_batched = Batch.from_data_list(rigid_graphs)
            soft_def_graphs_batched = Batch.from_data_list(soft_def_graphs)


            soft_rest_graphs_batched, rigid_graphs_batched, soft_def_graphs_batched = soft_rest_graphs_batched.to(device), rigid_graphs_batched.to(device), soft_def_graphs_batched.to(device)

            predictions = model(soft_rest_graphs_batched, rigid_graphs_batched)
            predictions.pos = predictions.pos  - soft_rest_graphs_batched.pos
            soft_def_graphs_batched.pos = soft_def_graphs_batched.pos  - soft_rest_graphs_batched.pos

            loss_mse = criterion_mse(predictions.pos, soft_def_graphs_batched.pos)
            loss_consistency = criterion_grad(predictions, soft_def_graphs_batched)
            loss_deformable = criterion_def(predictions, soft_def_graphs_batched, meta_data['deform_intensity'].to(device))
            lambda_deformable = config.training.lambda_deformable  
            tr_loss = loss_mse# + lambda_gradient * loss_consistency #+lambda_deformable * loss_deformable
            # if epoch<50:
            #     tr_loss = loss_mse + lambda_gradient * loss_consistency
            # else:
            #     tr_loss = loss_deformable	
            wandb.log({
                "tr_mse_loss": loss_mse.item(),
                "tr_consistency_loss": loss_consistency.item(),
                "tr_deformable_loss": loss_deformable.item(),
                "tr_loss": tr_loss.item()
            })
            total_tr_loss += tr_loss.item()
            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()  
        total_val_loss = 0.0
        with torch.no_grad():  
            for batch_idx, (obj_name, soft_rest_graphs, soft_def_graphs, meta_data, rigid_graphs) in enumerate(dataloader_val):
                soft_rest_graphs_batched = Batch.from_data_list(soft_rest_graphs)
                rigid_graphs_batched = Batch.from_data_list(rigid_graphs)
                soft_def_graphs_batched = Batch.from_data_list(soft_def_graphs)

                soft_rest_graphs_batched, rigid_graphs_batched, soft_def_graphs_batched = soft_rest_graphs_batched.to(device), rigid_graphs_batched.to(device), soft_def_graphs_batched.to(device)

                predictions = model(soft_rest_graphs_batched, rigid_graphs_batched)
                loss_mse = criterion_mse(predictions.pos, soft_def_graphs_batched.pos)
                loss_consistency = criterion_grad(predictions, soft_def_graphs_batched)
                loss_deformable = criterion_def(predictions, soft_def_graphs_batched, meta_data['deform_intensity'].to(device))
                loss_val = loss_mse + lambda_gradient * loss_consistency + lambda_deformable * loss_deformable


                total_val_loss += loss_val.item()

        avg_val_loss = total_val_loss / len(dataloader_val)
        avg_tr_loss = total_tr_loss / len(dataloader_train)
        print(f"Epoch {epoch+1}/{config.training.n_epochs} - Training Loss: {avg_tr_loss} - Validation Loss: {avg_val_loss}")
        
        # Logging validation loss to wandb
        wandb.log({"validation_loss": avg_val_loss})

        # Save the model after each epoch
        model_save_path = os.path.join(wandb.run.dir, 'model_weights.pth')
        config_save_path = os.path.join(wandb.run.dir, 'config.json')

        torch.save(model.state_dict(), model_save_path)

        # Save the configuration file to the same directory
        config.save(config_save_path)


        # Save both files to the WandB run directory
        wandb.save(model_save_path)
        wandb.save(config_save_path)

    wandb.finish()

if __name__ == "__main__":
    config = Config()
    wandb.init(project="GeoContact")
    train(config)
