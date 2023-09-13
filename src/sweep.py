import wandb
from train import train
from configs.config import Config

sweep_config = {
    "name": "sweeep",
    "method": "bayes",
    "metric": {
        'name': 'validation_loss',
        'goal': 'minimize'  # or 'maximize' depending on your needs
    },
    "parameters": {
        "encoder_layers": {"values": [2, 3]},
        "decoder_layers": {"values": [2, 3]},
        "hidden_dim": {"values": [64,128, 256]},

        "knn_k": {"values": [3, 5,7]},
        "learning_rate": {
            "distribution": "uniform", 
            "min": 0.0001, 
            "max": 0.001
        },
        "lambda_gradient": {
            "distribution": "uniform", 
            "min": 0.01, 
            "max": 1
        }
    }
}



def train_sweep():
    run = wandb.init()
    # wandb.init(project="GeoContact")  # Initialize wandb
    config_path = "configs/everyday.json" 
    config = Config(config_path)  # Initialize your Config object
    config.network.encoder_layers = run.config["encoder_layers"]
    config.network.decoder_layers = run.config["decoder_layers"]
    config.network.hidden_dim = run.config["hidden_dim"]
    config.network.knn_k = run.config["knn_k"]
    config.training.learning_rate = run.config["learning_rate"]
    config.training.lambda_gradient = run.config["lambda_gradient"]
    
    return train(config)


if __name__ == "__main__":

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep=sweep_config, project='GeoContact')

    # Launch the sweep agent
    wandb.agent(sweep_id, function=train_sweep, count=30) 

