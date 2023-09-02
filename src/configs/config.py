import json

class Config:
    def __init__(self, updates=None):
        self.dataset = DatasetConfig()  # Initialize nested DatasetConfig class
        self.visualization = VisualizationConfig()  # Initialize nested VisualizationConfig class
        self.dataloader = DataLoaderConfig()  # Initialize nested DataLoaderConfig class
        self.training = TrainingConfig()  # Initialize nested TrainingConfig class
        self.network = NetworkConfig()  # Initialize nested NetworkConfig class

        with open('src/configs/default.json', 'r') as f:
            defaults = json.load(f)
        if defaults:
            self._load_defaults(defaults)
        if updates:
            self._apply_updates(updates)

    def _load_defaults(self, defaults):
        for key, value in defaults.items():
            if hasattr(self, key):
                for sub_key, sub_value in value.items():
                    if hasattr(getattr(self, key), sub_key):
                        setattr(getattr(self, key), sub_key, sub_value)

    def _apply_updates(self, updates):
        for key, value in updates.items():
            if hasattr(self, key):
                for sub_key, sub_value in value.items():
                    if hasattr(getattr(self, key), sub_key):
                        setattr(getattr(self, key), sub_key, sub_value)

class DatasetConfig:
    def __init__(self):
        self.root_dir = "datasets/everyday_deform/deformations/"
        self.obj_list = ["Box"]
        self.n_points = 1024
        self.radius = 0.2
        self.k = 7
        self.graph_method = "knn"

class VisualizationConfig:
    def __init__(self):
        self.collider_radius_contact = 0.1
        self.collider_radius_deform = 0.01
        self.colors = {
            "contact_collider": [0, 0, 1],
            "deform_collider": [1, 1, 0],
            "rest_pcd": [1, 0, 0],
            "def_pcd": [0, 1, 0],
            "lineset": [0.5, 0.5, 0.5],
            "vector": [0, 0, 0]
        }

class DataLoaderConfig:
    def __init__(self):
        self.batch_size = 4
        self.shuffle = True

class TrainingConfig:
    def __init__(self):
        self.n_epochs = 20
        self.learning_rate = 0.0001
        self.model_save_path = "model.pth"
        self.lambda_gradient = 0.1

class NetworkConfig:
    def __init__(self):
        self.input_dims = [3, 6]
        self.hidden_dim = 64
        self.output_dim = 3
        self.num_layers = 2
        self.dropout_rate = 0.1
        self.knn_k = 3
        self.backbone = "TAGConv"
