# extract_embeddings_v3.py
import os, sys, json
import numpy as np
import torch
from torch_geometric.loader import DataLoader

# add your src path if needed
# sys.path.append('/home/lucas/graph_scenarios/src')

from utils import GraphDatasetV3
from networks.graph_vae_v3 import GRAPH_VAE_V3
from tqdm import tqdm

# ---------- config ----------
ROOT = "/home/lucas/graph_scenarios/data/sub20/graphs"
WEIGHTS = "networks/weights/GRAPH_VAE_V3_UNDIR_latent=128_lr=0.0005_epochs=2000_variational_beta=0.05_capacity=256_dec_layers=4_enc_layers=3_ign_layers=2_edge_layers=4_best.pt"
OUT_DIR = "embeddings"
BATCH_SIZE = 256
LATENT_DIM = 128
DEVICE = torch.device("cuda")

# must match what you trained with (only the needed bits)
params = {
    'distribution_std': 0.1,
    'variational_beta': 0.05,
    'capacity': 256,
    'dec_layers': 4,
    'edge_layers': 4,
    'ign_layers': 2,
    "edge_pos_weight": 1.0,   # you already use this
    "edge_neg_weight": 1.0,    # NEW: >1 penalizes negatives → higher precision
    "edge_loss_lambda": 1.5,
    "ign_loss_lambda": 0.005,
}    

# ---------- data / model ----------
dataset = GraphDatasetV3(root=ROOT)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers = 6)

# input_dim = 5 for (pos2 + alt + slope + fuel_scalar) in your V3 setup
model = GRAPH_VAE_V3(input_dim=5, latent_dim=LATENT_DIM, params=params, template=dataset.template).to(DEVICE)
state = torch.load(WEIGHTS, map_location=DEVICE)
# allow non-strict in case you added buffers/heads later
model.load_state_dict(state if isinstance(state, dict) else state["model_state_dict"], strict=False)
model.eval()

os.makedirs(OUT_DIR, exist_ok=True)

emb_mu = []
emb_logvar = []
correspondence = {}   # idx_in_order -> original graph id (if available)

idx_counter = 0
with torch.inference_mode():
    pbar = tqdm(total=len(dataset), desc="Embedding graphs", unit="graph", dynamic_ncols=True)
    for batch in loader:
        batch = batch.to(DEVICE, non_blocking=True)
        # forward: returns (output, mu, logvar)
        _, mu, log = model(batch.x, batch.edge_index_enc, batch)   # mu/log: [B, z]
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

np.save(os.path.join(OUT_DIR, "mu_graph_vae_v3.npy"), emb_mu)
np.save(os.path.join(OUT_DIR, "logvar_graph_vae_v3.npy"), emb_log)
with open(os.path.join(OUT_DIR, "correspondence_graph_vae_v3.json"), "w") as f:
    json.dump(correspondence, f, indent=2)

print(f"Saved:\n  {os.path.join(OUT_DIR,'mu_graph_vae_v3.npy')}\n  {os.path.join(OUT_DIR,'logvar_graph_vae_v3.npy')}\n  {os.path.join(OUT_DIR,'correspondence_graph_vae_v3.json')}")
