#!/bin/bash
# Vast.ai Setup Script for OSRS RL
# Image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

echo "=============================================="
echo "OSRS RL - Vast.ai GPU Setup"
echo "=============================================="

# Update and install basics
apt-get update && apt-get install -y git wget unzip

# Clone your repo (or upload files)
# git clone https://github.com/YOUR_USERNAME/osrs-rl.git
# cd osrs-rl

# Install Python dependencies
pip install --upgrade pip
pip install \
    stable-baselines3[extra]==2.3.0 \
    gymnasium==0.29.1 \
    numpy \
    torch \
    tensorboard \
    tqdm

# Verify GPU
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "Setup complete! Run training with:"
echo "  python3 train_unified.py --boss vorkath --frames 8 --timesteps 2000000"
