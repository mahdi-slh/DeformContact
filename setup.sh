source ~/.bashrc
false | conda create -n deform python=3.11 pytorch torchvision cudatoolkit=11.8 pyg -c pytorch -c pyg
conda activate deform
python -m pip install -U matplotlib tqdm wandb open3d
python -m pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl