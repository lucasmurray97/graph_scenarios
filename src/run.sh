# Script to run several experiments of train.py
# Usage: ./run.sh
#!/bin/bash
python train.py --latent_dim 32 --epochs 50 --lr 1e-6 --batch_size 64 --model v3 --capacity 64 --variational_beta 1e-2
python train.py --latent_dim 32 --epochs 50 --lr 1e-6 --batch_size 64 --model v3 --capacity 32 --variational_beta 1e-2
python train.py --latent_dim 32 --epochs 50 --lr 1e-6 --batch_size 64 --model v3 --capacity 128 --variational_beta 1e-2
python train.py --latent_dim 32 --epochs 50 --lr 1e-6 --batch_size 64 --model v3 --capacity 16 --variational_beta 1e-2
python train.py --latent_dim 32 --epochs 50 --lr 1e-6 --batch_size 64 --model v3 --capacity 256 --variational_beta 1e-2