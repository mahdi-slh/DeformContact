# Physics-Informed Graph Neural Networks for Deformation Prediction under Contact

Predict deformation of shapes using physics-informed graphs 

## Prerequisites

- Anaconda or Miniconda installed on your system. If not, download it [here](https://www.anaconda.com/products/individual).
- A Weights & Biases account. Sign up [here](https://wandb.ai/).

## Installation and Setup

1. **Clone the Repository:**
   
   ```sh
   git clone https://github.com/mahdi-slh/GeoContact.git
   cd GeoContact/src
Replace username with your GitHub username and repository with the name of your repository.

2. **Run Setup Script:**
   ```sh
    bash setup.sh

This script will create and activate a conda environment named deform, and install the necessary packages.

3. **Setup Weights & Biases:**

   ```sh
    wandb login
Follow the on-screen instructions to log in to your Weights & Biases account.


4. **Visualize the data**
   ```sh
    python visualize.py
Ensure the config_path variable in the main function of visualize.py is set to the path of your config file.


5. **Train the Model**
   ```sh
    python train.py
Ensure the config_path variable in the main function of train.py is set to the path of your config file.

6. **Evaluate the Model**
   ```sh
    python eval.py
Ensure the config_path variable in the main function of eval.py is set to the path of your config file.

