#!/bin/bash
#SBATCH --partition=gpu

# Check Operating System
echo "Operating System:"
uname -a
cat /etc/os-release

# Load Conda
conda activate TSA

# Check CUDA Version
echo "CUDA Version:"
nvcc --version

# Check PyTorch Version
echo "PyTorch Version:"
python -c 'import torch; print(torch.__version__)'

# Start Inference
python inference.py --config configs/example.yaml