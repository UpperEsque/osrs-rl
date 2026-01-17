#!/bin/bash
echo "Installing dependencies..."
pip install stable-baselines3[extra] gymnasium numpy tqdm

echo ""
echo "Testing GPU..."
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

echo ""
echo "Ready! Run:"
echo "  python3 train_gpu.py --boss vorkath --timesteps 2000000"
