# Physics-Encoded Graph Neural Networks for Deformation Prediction under Contact

This repository contains the implementation of a model that predicts deformation of shapes using physics encoded graph neural networks. Our approach has been accepted at ICRA 2024, and you can find the preprint of our paper on [arXiv](https://arxiv.org/abs/2402.03466).


![DeformContact](https://mahdi-slh.github.io/pages_static/images/DeformContact_overview.png)

## Prerequisites

- **Anaconda or Miniconda**: Ensure Anaconda or Miniconda is installed on your system. Download it [here](https://www.anaconda.com/products/individual).
- **Weights & Biases Account**: Needed for experiment tracking. Sign up [here](https://wandb.ai/).

## Installation and Setup

1. **Clone the Repository:**
   
   ```sh
   git clone https://github.com/mahdi-slh/DeformContact.git
   cd DeformContact
Replace username with your GitHub username and repository with the name of your repository.

2. **Run Setup Script:**
   ```sh
    bash setup.sh

This script will create and activate a conda environment named deform, and install the necessary packages.

3. **Setup Weights & Biases:**
Login to your Weights & Biases account:

   ```sh
    wandb login
Follow the on-screen instructions.


4. **Download the dataset**
Please download the dataset from [here](https://drive.google.com/file/d/1mWIK1WM-qEE67y9Kvj2UVY7d45fTUBHv/view?usp=sharing) and place it in the following directory within the cloned repository.
   ```sh
    python visualize.py
Ensure the config_path variable in the main function of visualize.py is set to the path of your config file.


5. **Visualize the data**
Run the visualization script:

   ```sh
    python visualize.py
Ensure the config_path variable in the main function of visualize.py is set to the path of your config file.


6. **Train the Model**
Start model training:
   ```sh
    python train.py
Ensure the config_path variable in the main function of train.py is set to the path of your config file.

7. **Evaluate the Model**
Evaluate the trained model:
   ```sh
    python eval.py

