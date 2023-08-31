from torch.utils.data import DataLoader
from dataloaders.everyday_deform_v2 import EverydayDeformDataset
from utils.visualization import *
from utils.collate import collate_fn

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


    dataloader = DataLoader(dataset, batch_size=dataloader_config["batch_size"], shuffle=dataloader_config["shuffle"], collate_fn=collate_fn)

    for obj_name, rest_graphs, def_graphs, meta_data, collider_graph in dataloader:


        visualize_deformation_field(rest_graphs[0].pos,def_graphs[0].pos, meta_data['deformer_collision_position'][0], meta_data['deformer_origin'][0])
        visualize_merged_graphs(rest_graphs[0],def_graphs[0],collider_graph[0])
