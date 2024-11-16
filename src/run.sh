# Script to run several experiments of train.py
# Usage: ./run.sh
#!/bin/bash
python train.py --latent_dim 128 --epochs 200 --lr 1e-3 --variational_beta 100000 --batch_size 128 --model v1
python train.py --latent_dim 128 --epochs 200 --lr 1e-4 --variational_beta 100000 --batch_size 128 --model v1

python train.py --latent_dim 128 --epochs 200 --lr 1e-3 --variational_beta 80000 --batch_size 128 --model v1
python train.py --latent_dim 128 --epochs 200 --lr 1e-4 --variational_beta 80000 --batch_size 128 --model v1


python train.py --latent_dim 128 --epochs 200 --lr 1e-3 --variational_beta 1e-4 --batch_size 128 --model v2
python train.py --latent_dim 128 --epochs 200 --lr 1e-4 --variational_beta 1e-4 --batch_size 128 --model v2

python train.py --latent_dim 128 --epochs 200 --lr 1e-3 --variational_beta 1e-5 --batch_size 128 --model v2
python train.py --latent_dim 128 --epochs 200 --lr 1e-4 --variational_beta 1e-5 --batch_size 128 --model v2