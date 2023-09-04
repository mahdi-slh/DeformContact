import json

class Config:
    def __init__(self, updates=None):
        self.dataset = DatasetConfig()  # Initialize nested DatasetConfig class
        self.visualization = VisualizationConfig()  # Initialize nested VisualizationConfig class
        self.dataloader = DataLoaderConfig()  # Initialize nested DataLoaderConfig class
        self.training = TrainingConfig()  # Initialize nested TrainingConfig class
        self.network = NetworkConfig()  # Initialize nested NetworkConfig class

        with open('configs/default.json', 'r') as f:
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
    def save(self, file_path):
        """
        Save the configuration to a JSON file.

        Args:
            file_path (str): The path to the JSON file where the configuration will be saved.
        """
        config_dict = {
            "dataset": self.dataset.__dict__,
            "visualization": self.visualization.__dict__,
            "dataloader": self.dataloader.__dict__,
            "training": self.training.__dict__,
            "network": self.network.__dict__
        }

        with open(file_path, 'w') as config_file:
            json.dump(config_dict, config_file, indent=4)

class DatasetConfig:
    def __init__(self):
        self.root_dir = None
        self.obj_list = None
        self.n_points = None
        self.radius = None
        self.k = None
        self.graph_method = None

class VisualizationConfig:
    def __init__(self):
        self.rigid_radius_contact = None
        self.rigid_radius_deform = None
        self.colors = {
            "contact_rigid": [0, 0, 1],
            "deform_rigid": [1, 1, 0],
            "soft_rest_pcd": [1, 0, 0],
            "soft_def_pcd": [0, 1, 0],
            "lineset": [0.5, 0.5, 0.5],
            "vector": [0, 0, 0]
        }

class DataLoaderConfig:
    def __init__(self):
        self.batch_size = None
        self.shuffle = None

class TrainingConfig:
    def __init__(self):
        self.n_epochs = None
        self.learning_rate = None
        self.model_save_path =None
        self.lambda_gradient = None
        self.lambda_deformable = None

class NetworkConfig:
    def __init__(self):
        self.input_dims = []
        self.hidden_dim = None
        self.output_dim = None
        self.encoder_layers = None
        self.decoder_layers = None
        self.dropout_rate = None
        self.knn_k = None
        self.backbone = None
