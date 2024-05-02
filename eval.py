import torch

from torch_geometric.data import Batch
from utils.visualization import *
from configs.config import Config
from models.model_loader import load_model
from models.losses import GradientConsistencyLoss
from loaders.dataset_loader import load_dataset
import wandb
import os
import torch.nn as nn
from utils.graph_utils import *

visualize = False


import time


def eval_runtime():
    run_id = input("Enter the wandb run ID (e.g. mahdi-slh/DeformContact/runs/XXXXXXX): ")
    model_file_name = "model_weights.pth"
    config_file_name = "config.json"

    log_dir = f"./wandb/{run_id}/logs"

    run = wandb.Api().run(run_id)
    run.file(model_file_name).download(replace=True, root=log_dir)
    run.file(config_file_name).download(replace=True, root=log_dir)

    config = Config(os.path.join(log_dir, config_file_name))
    _, dataloader_val = load_dataset(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config).to(device)
    model.load_state_dict(torch.load(os.path.join(log_dir, "model_weights.pth")))
    model.eval()

    total_data_load_time = 0.0
    total_inference_time = 0.0

    with torch.no_grad():
        for batch_idx, (
            obj_name,
            soft_rest_graphs,
            soft_def_graphs,
            meta_data,
            rigid_graphs,
        ) in enumerate(dataloader_val):

            start_time = time.time()

            soft_rest_graphs_batched = Batch.from_data_list(soft_rest_graphs).to(device)
            rigid_graphs_batched = Batch.from_data_list(rigid_graphs).to(device)

            end_time = time.time()
            total_data_load_time += end_time - start_time

            start_time = time.time()

            predictions = model(soft_rest_graphs_batched, rigid_graphs_batched)

            end_time = time.time()
            total_inference_time += end_time - start_time

    avg_data_load_time = total_data_load_time / len(dataloader_val)
    avg_inference_time = total_inference_time / len(dataloader_val)

    print(f"Average Data Loading Time per Batch: {avg_data_load_time:.6f} seconds")
    print(f"Average Inference Time per Batch: {avg_inference_time:.6f} seconds")


def eval():
    run_id = input("Enter the wandb run ID (e.g. mahdi-slh/DeformContact/runs/XXXXXXX): ")
    model_file_name = "model_weights.pth"
    config_file_name = "config.json"

    log_dir = f"./wandb/{run_id}/logs"

    run = wandb.Api().run(run_id)
    run.file(model_file_name).download(replace=True, root=log_dir)
    run.file(config_file_name).download(replace=True, root=log_dir)

    config = Config(os.path.join(log_dir, config_file_name))
    _, dataloader_val = load_dataset(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config).to(device)
    model.load_state_dict(torch.load(os.path.join(log_dir, "model_weights.pth")))
    model.eval()

    criterion_mse = nn.MSELoss()
    criterion_grad = GradientConsistencyLoss()
    criterion_mae = nn.L1Loss()

    total_mae, total_mse, total_consistency = 0.0, 0.0, 0.0
    errors = []

    with torch.no_grad():
        for batch_idx, (
            obj_name,
            soft_rest_graphs,
            soft_def_graphs,
            meta_data,
            rigid_graphs,
        ) in enumerate(dataloader_val):
            soft_rest_graphs_batched = Batch.from_data_list(soft_rest_graphs).to(device)
            rigid_graphs_batched = Batch.from_data_list(rigid_graphs).to(device)
            soft_def_graphs_batched = Batch.from_data_list(soft_def_graphs).to(device)

            predictions = model(soft_rest_graphs_batched, rigid_graphs_batched)

            loss_mse = criterion_mse(predictions.pos, soft_def_graphs_batched.pos)
            loss_mae = criterion_mae(predictions.pos, soft_def_graphs_batched.pos)

            loss_consistency = criterion_grad(predictions, soft_def_graphs_batched)
            errors.append((predictions.pos - soft_def_graphs_batched.pos).cpu().numpy())

            total_mse += loss_mse.item()
            total_consistency += loss_consistency.item()
            total_mae += loss_mae.item()

            if visualize:
                for indx in range(config.dataloader.batch_size):

                    output_folder = "./outputs/{}".format(config.dataset.obj_list[0])
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)

                    rigid_mesh = meta_data["rigid_mesh"][indx]
                    rigid_mesh_path = os.path.join(
                        "./outputs/", meta_data["sample_path"][indx] + "_rigid.obj"
                    )
                    o3d.io.write_triangle_mesh(rigid_mesh_path, rigid_mesh)

                    soft_mesh = meta_data["soft_rest_mesh"][indx]

                    # For resting
                    soft_mesh.vertices = o3d.utility.Vector3dVector(
                        soft_rest_graphs[indx].pos.cpu().numpy()
                    )
                    resting_mesh_path = os.path.join(
                        "./outputs/", meta_data["sample_path"][indx] + "_resting.obj"
                    )
                    o3d.io.write_triangle_mesh(resting_mesh_path, soft_mesh)

                    # For gt
                    soft_mesh.vertices = o3d.utility.Vector3dVector(
                        soft_def_graphs_batched[indx].pos.cpu().numpy()
                    )
                    gt_mesh_path = os.path.join(
                        "./outputs/", meta_data["sample_path"][indx] + "_gt.obj"
                    )
                    o3d.io.write_triangle_mesh(gt_mesh_path, soft_mesh)

                    # For prediction
                    soft_mesh.vertices = o3d.utility.Vector3dVector(
                        predictions[indx].pos.cpu().numpy()
                    )
                    pred_mesh_path = os.path.join(
                        "./outputs/", meta_data["sample_path"][indx] + "_pred.obj"
                    )
                    o3d.io.write_triangle_mesh(pred_mesh_path, soft_mesh)
                    # visualize_deformations_normals_colors(soft_rest_graphs[indx])
                    # visualize_deformations_normals_colors(soft_def_graphs_batched[indx])
                    # visualize_deformations_normals_colors(predictions[indx])

                    # visualize_deformations_normals_colors(soft_rest_graphs[indx], soft_def_graphs_batched[indx])
                    # visualize_deformation_field(soft_rest_graphs[indx].pos.cpu(), predictions[indx].pos.cpu(),rigid_graphs[indx].pos.cpu(), meta_data['force_vector'][indx])
                    # visualize_merged_graphs(soft_rest_graphs[indx], soft_def_graphs_batched[indx], rigid_graphs[indx],predictions[indx])

        avg_mae = loss_mae / len(dataloader_val)
        avg_mse = total_mse / len(dataloader_val)
        errors = np.concatenate(errors)
        variance_of_error = np.var(errors)
        avg_consistency = total_consistency / len(dataloader_val)
        obj_name = config.dataset.obj_list[0]
        print(f"Average MSE Error for {obj_name}: {avg_mse}")
        print(f"Average Consistency Error for {obj_name}: {avg_consistency}")
        print(f"Average MAE Error for {obj_name}: {avg_mae}")
        print(f"Variance of MSE Error for {obj_name}: {variance_of_error}")


if __name__ == "__main__":

    eval()
    # eval_runtime()
