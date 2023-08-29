source ~/.bashrc
false | conda create -n deform pytorch=1.11 torchvision cudatoolkit=10.2 pyg -c pytorch -c pyg
conda activate deform
python -m pip install -U matplotlib tqdm tensorboard scikit-image wandb pymeshlab open3d
# <<<<<<< HEAD
python -m pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
