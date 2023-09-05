from torch.utils.data import DataLoader
from dataloaders.everyday_deform import EverydayDeformDataset
from utils.visualization import *
from dataloaders.collate import collate_fn


import json

if __name__ == "__main__":
    with open("configs/default.json", "r") as f:
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

    for obj_name, soft_rest_graphs, soft_def_graphs, meta_data, rigid_graph in dataloader:


        visualize_deformation_field(soft_rest_graphs[0].pos,soft_def_graphs[0].pos,rigid_graph[0].pos, meta_data['deformer_collision_position'][0], meta_data['deformer_origin'][0])
        visualize_merged_graphs(soft_rest_graphs[0],soft_def_graphs[0],rigid_graph[0])
