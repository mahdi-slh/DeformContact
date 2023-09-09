from data.everyday_deform import EverydayDeformDataset
from data.retina_deform import RetinaDeformDataset
from torch.utils.data import DataLoader
from data.collate import collate_fn

def load_dataset(config):
    if config.dataset.name == "retina":
        train_dataset = RetinaDeformDataset(root_dir=config.dataset.root_dir, 
        n_points=config.dataset.n_points,
        graph_method=config.dataset.graph_method,
        radius=config.dataset.radius,
        k=config.dataset.k, split='train')
    
        val_dataset = RetinaDeformDataset(root_dir=config.dataset.root_dir, 
        n_points=config.dataset.n_points,
        graph_method=config.dataset.graph_method,
        radius=config.dataset.radius,
        k=config.dataset.k, split='val')


    elif config.dataset.name == "everyday":
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
    else:
        raise ValueError(f"Unknown dataset name: {config.dataset.name}")
    

    dataloader_val = DataLoader(
    val_dataset, 
    batch_size=config.dataloader.batch_size, 
    shuffle=False,
    collate_fn=collate_fn
    )

    dataloader_train = DataLoader(
    train_dataset, 
    batch_size=config.dataloader.batch_size, 
    shuffle=config.dataloader.shuffle,
    collate_fn=collate_fn
    )

    return dataloader_train,dataloader_val

    
