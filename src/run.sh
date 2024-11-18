# Script to run several experiments of train.py
# Usage: ./run.sh
#!/bin/bash
python train.py --latent_dim 256 --epochs 100 --lr 1e-5 --batch_size 64 --model v3
python train.py --latent_dim 128 --epochs 100 --lr 1e-5 --batch_size 64 --model v3
python train.py --latent_dim 64 --epochs 100 --lr 1e-5 --batch_size 64 --model v3
python train.py --latent_dim 32 --epochs 100 --lr 1e-5 --batch_size 64 --model v3
python train.py --latent_dim 16 --epochs 100 --lr 1e-5 --batch_size 64 --model v3