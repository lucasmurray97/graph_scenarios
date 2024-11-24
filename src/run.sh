# Script to run several experiments of train.py
# Usage: ./run.sh
#!/bin/bash
python train_graph_vae.py --latent_dim 64 --epochs 50 --lr 1e-6 --batch_size 64 --model v1 --capacity 128 --variational_beta 1.0 --loss focal --alpha 0.9 --gamma 0.5
python train_graph_vae.py --latent_dim 64 --epochs 50 --lr 1e-6 --batch_size 64 --model v1 --capacity 128 --variational_beta 1.0 --loss focal --alpha 0.8 --gamma 0.5
python train_graph_vae.py --latent_dim 64 --epochs 50 --lr 1e-6 --batch_size 64 --model v1 --capacity 128 --variational_beta 1.0 --loss focal --alpha 0.9 --gamma 1.0
python train_graph_vae.py --latent_dim 64 --epochs 50 --lr 1e-6 --batch_size 64 --model v1 --capacity 128 --variational_beta 1.0 --loss focal --alpha 0.8 --gamma 1.0
python train_graph_vae.py --latent_dim 64 --epochs 50 --lr 1e-6 --batch_size 64 --model v1 --capacity 128 --variational_beta 1.0 --loss focal --alpha 0.85 --gamma 0.8