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
from utils import GraphDataset, reconstruct_matrix, GraphDatasetV3
from networks.graph_vae import GRAPH_VAE
from networks.graph_vae_v2 import GRAPH_VAE_V2
from networks.gae import GAE
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch.utils.data import random_split
import argparse
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from tqdm import tqdm
import os
import json

ROOT = "/home/lucas/graph_scenarios/data/sub20/graphs"
WEIGHTS = "networks/weights/GRAPH_VAE_V3_UNDIR_latent=128_lr=0.0005_epochs=2000_variational_beta=0.05_capacity=256_dec_layers=4_enc_layers=3_ign_layers=2_edge_layers=4_best.pt"
OUT_DIR = "embeddings"
dataset = GraphDatasetV3(root=ROOT)
DEVICE = torch.device("cuda")
BATCH_SIZE = 256
# Define the parameters
params = {
    'distribution_std': 0.1,
    'variational_beta': 0.05,
    'capacity': 1024,
    'dec_layers': 2,
    'enc_layers': 3,
    'ign_layers': 3,
    'ign_layers': 2,
}

input_dim = 5
latent_dim = 128
model = GAE(input_dim, latent_dim, params, dataset.template).to("cuda")
# Load weights
model.load_state_dict(torch.load(f"./networks/weights/VGAE_UNDIR_latent=128_lr=0.0002_epochs=1000_variational_beta=0.05_capacity=1024_dec_layers=2_enc_layers=3_ign_layers=2_edge_layers=2_best.pt")["model_state_dict"])
model.eval()
# Create the DataLoader
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers = 6)

emb_mu = []
emb_logvar = []
correspondence = {}
idx_counter = 0
with torch.inference_mode():
    pbar = tqdm(total=len(dataset), desc="Embedding graphs", unit="graph", dynamic_ncols=True)
    for batch in loader:
        batch = batch.to(DEVICE, non_blocking=True)
        # forward: returns (output, mu, logvar)
        _, mu, log = model(batch.x, batch.edge_index_enc, batch)   # mu/log: [B, z]
        mu = model.pool(mu, batch.batch)
        log = model.pool(log, batch.batch)
        mu_cpu  = mu.detach().cpu().numpy()
        log_cpu = log.detach().cpu().numpy()

        # split per-graph even when B>1
        data_list = batch.to_data_list()
        for j, g in enumerate(data_list):
            emb_mu.append(mu_cpu[j])
            emb_logvar.append(log_cpu[j])

            # try several common field names from your pipeline
            gid = getattr(g, "original_graph", None)
            if gid is None: gid = getattr(g, "graph_id", None)
            if gid is None: gid = getattr(g, "gid", None)

            # normalize to int when possible
            if torch.is_tensor(gid):
                gid = int(gid.item())
            elif isinstance(gid, (np.integer,)):
                gid = int(gid)
            correspondence[idx_counter] = gid if gid is not None else idx_counter
            idx_counter += 1
        B = batch.num_graphs
        pbar.update(B) 
    pbar.close()


# ---------- save ----------
emb_mu  = np.asarray(emb_mu)      # [num_graphs, z]
emb_log = np.asarray(emb_logvar)  # [num_graphs, z]

np.save(os.path.join(OUT_DIR, "mu_gae.npy"), emb_mu)
np.save(os.path.join(OUT_DIR, "logvar_gae.npy"), emb_log)
with open(os.path.join(OUT_DIR, "correspondence_gae.json"), "w") as f:
    json.dump(correspondence, f, indent=2)

print(f"Saved:\n  {os.path.join(OUT_DIR,'mu_gae.npy')}\n  {os.path.join(OUT_DIR,'logvar_gae.npy')}\n  {os.path.join(OUT_DIR,'correspondence_gae.json')}")

