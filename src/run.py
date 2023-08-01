from torch.utils.data import DataLoader

from dataloaders.everyday_deform import EverydayDeformDataset
from visualize import *


if __name__ == "__main__":
    dataset = EverydayDeformDataset(["bag"], "datasets/everyday_deform/")

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for rest_mesh, def_mesh, meta_data in dataloader:
        print(rest_mesh.shape, def_mesh.shape, meta_data)
        visualize_deformation_field(rest_mesh[1], def_mesh[1])
        
