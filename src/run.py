from torch.utils.data import DataLoader
from dataloaders.everyday_deform_v2 import EverydayDeformDataset
from visualize import *
import json



if __name__ == "__main__":
    with open("src/configs/default.json", "r") as f:
        config = json.load(f)

    dataset_config = config["dataset"]
    dataloader_config = config["dataloader"]
    visualization_config = config["visualization"]

    dataset = EverydayDeformDataset(
        root_dir = dataset_config["root_dir"], 
        obj_list = dataset_config["obj_list"], 
        n_points = dataset_config["n_points"],
        graph_method = dataset_config["graph_method"],
        radius = dataset_config["radius"],
        k = dataset_config["k"]
    )

    dataloader = DataLoader(dataset, batch_size=dataloader_config["batch_size"], shuffle=dataloader_config["shuffle"])

    for obj_name, (rest_points, rest_edge_index), (def_points, def_edge_index), meta_data in dataloader:
        visualize_deformation_field(rest_points[0], def_points[0], meta_data['deformer_collision_position'][0], meta_data['deformer_origin'][0])
        visualize_merged_graphs(rest_points[0], rest_edge_index[0], def_points[0], def_edge_index[0])
        



