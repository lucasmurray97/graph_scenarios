# Script to run several experiments of train.py
# Usage: ./run.sh
#!/bin/bash
# python train.py --latent_dim 128 --epochs 100 --lr 1e-3 --batch_size 64 --model v3 --capacity 128 --variational_beta 1e-3
python train.py --latent_dim 128 --epochs 100 --lr 1e-3 --batch_size 64 --model v3 --capacity 256 --variational_beta 1e-2
python train.py --latent_dim 128 --epochs 100 --lr 1e-3 --batch_size 64 --model v3 --capacity 512 --variational_beta 5e-2
python train.py --latent_dim 128 --epochs 100 --lr 1e-3 --batch_size 64 --model v3 --capacity 256 --variational_beta 3e-3
