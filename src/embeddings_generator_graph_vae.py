import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.datasets as D
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges, to_dense_adj, to_networkx
import sys
#add path to utils
sys.path.append('../../')
from utils import GraphDataset, reconstruct_matrix
from networks.graph_vae import GRAPH_VAE
from networks.graph_vae_v2 import GRAPH_VAE_V2
from networks.gae import GRAPH_VAE_V3
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch.utils.data import random_split
import argparse
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

dataset = GraphDataset(root='../data/sub20/graphs')

# Define the parameters
params = {
    'distribution_std': 0.1,
    'variational_beta': 10.,
    "capacity": 128,
    'loss': 'focal',
    'alpha': 0.8,
    'gamma': 1.0
}

input_dim = dataset.num_features
latent_dim = 64
model = GRAPH_VAE(input_dim, latent_dim, params).to("cuda")
# Load weights
model.load_state_dict(torch.load(f"networks/weights/GRAPH_VAE_latent=64_lr=1e-06_epochs=50_variational_beta=10.0_capacity=128_alpha=0.8_gamma=1.0.pt"))
model.eval()
# Create the DataLoader
loader = DataLoader(dataset, batch_size=1, shuffle=False)

embeddings = []
correspondance = {}
for i, batch in enumerate(loader):
    batch = batch.to("cuda")
    output, mu, log = model(batch.x, batch.edge_index, batch.batch)
    embeddings.append((mu.detach().cpu().numpy(), log.detach().cpu().numpy()))
    correspondance[i] = batch.original_graph.item()

# Save list of numpy arrays
np.save("embeddings/embeddings_graph_vae.npy", embeddings)
# Save correspondance dict
np.save("embeddings/correspondance_graph_vae.npy", correspondance)

"""
# Load embeddings and correspondance
embeddings = np.load("embeddings/embeddings.npy", allow_pickle=True)
correspondance = np.load("embeddings/correspondance.npy", allow_pickle=True).item()
"""
