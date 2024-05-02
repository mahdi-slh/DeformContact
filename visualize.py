from utils.visualization import *

from data.dataset_loader import load_dataset
from configs.config import Config
from torch_geometric.data import Batch

if __name__ == "__main__":
    config = Config("configs/everyday.json")

    _, dataloader_val = load_dataset(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

            for indx in range(config.dataloader.batch_size):
                visualize_deformations_normals_colors(
                    soft_rest_graphs[indx], soft_def_graphs_batched[indx]
                )
                visualize_deformation_field(
                    soft_rest_graphs[indx].pos.cpu(),
                    soft_def_graphs_batched[indx].pos.cpu(),
                    rigid_graphs[indx].pos.cpu(),
                    meta_data["force_vector"][indx],
                )
                visualize_merged_graphs(
                    soft_rest_graphs[indx],
                    soft_def_graphs_batched[indx],
                    rigid_graphs[indx],
                    soft_def_graphs_batched[indx],
                )
