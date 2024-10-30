env_name=lavis

conda create --name ${env_name} python=3.9 -y && conda activate ${env_name}
# conda create --name ${env_name} python=3.10 -y && conda activate ${env_name}
# pip install -U pip
conda install -c conda-forge mamba -y

mamba install Ninja -y
# Install the correct version of CUDA
mamba install cuda -c nvidia/label/cuda-12.1.0 -y
# conda install nvidia/label/cuda-12.1.0::cuda -y
# mamba install nvidia/label/cuda-12.1.0::cuda -y # with python 3.10

# Install PyTorch and xformers
# You may need to install another xformers version if you use a different PyTorch version
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121


pip install -e .
pip install diffusers["torch"] transformers