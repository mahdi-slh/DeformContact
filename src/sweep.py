from itertools import product
from configs.config import Config
from train import train

# Define the parameter values to sweep over
num_layers_values = [2, 3]
hidden_dim_values = [64, 128]
dropout_rate_values = [0.1, 0.3, 0.5]
knn_k_values = [3, 5]
backbone_values = ["GATConv", "GCNConv", "TAGConv"]
learning_rate_values = [0.001, 0.0001, 0.00001]
lambda_gradient_values = [0.01, 0.1, 1, 10]

# Combine parameter values using itertools.product
parameter_combinations = product(num_layers_values, hidden_dim_values, dropout_rate_values,
                                 knn_k_values, backbone_values, learning_rate_values,
                                 lambda_gradient_values)

# Initialize the Config instance
config = Config()

best_params = None
best_score = float("inf")  # Initialize with a large value

for params in parameter_combinations:
    config = Config()
    num_layers, hidden_dim, dropout_rate, knn_k, backbone, learning_rate, lambda_gradient = params

    # Set the parameters in the config
    config.network["num_layers"] = num_layers
    config.network["hidden_dim"] = hidden_dim
    config.network["dropout_rate"] = dropout_rate
    config.network["knn_k"] = knn_k
    config.network["backbone"] = backbone
    config.training["learning_rate"] = learning_rate
    config.training["lambda_gradient"] = lambda_gradient

    # Train the model with the current parameters and get the validation loss
    avg_val_loss = train(config)

    # Update best parameters if current score is better
    if avg_val_loss < best_score:
        best_score = avg_val_loss
        best_params = params

# Print the best parameters and their corresponding score
print("Best Parameters: ", best_params)
print("Best Score: ", best_score)
