import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.datasets as D
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from utils import (
    GraphDataset, reconstruct_matrix, GraphDatasetV3,
    evaluate_all, metrics_on_batch_tm, estimate_edge_pos_weight, set_seed
)
from networks.graph_vae import GRAPH_VAE
from networks.graph_vae_v2 import GRAPH_VAE_V2
from networks.graph_vae_v3 import GRAPH_VAE_V3  # <- your UNDIRECTED V3
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch.utils.data import random_split
import argparse
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import json
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import os

# --------------------
# Determinism
# --------------------
SEED = 12345
set_seed(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # or ":4096:8"
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# --------------------
# CLI
# --------------------
parser = argparse.ArgumentParser()
parser.add_argument('--latent_dim', type=int, required=True)
parser.add_argument('--epochs', type=int, required=True, default=100)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--distribution_std', type=float, default=0.1)
parser.add_argument('--variational_beta', type=float, default=1.)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--model', type=str, default='v3')
parser.add_argument('--capacity', type=int, default=8)
parser.add_argument('--dec_layers', type=int, default=2)
parser.add_argument('--enc_layers', type=int, default=2)
parser.add_argument('--ign_layers', type=int, default=2)
parser.add_argument('--edge_layers', type=int, default=2)
parser.add_argument('--dec_residual', type=bool, default=True)
args = parser.parse_args()

latent_dim = args.latent_dim
epochs = args.epochs
lr = args.lr
variational_beta = args.variational_beta
distribution_std = args.distribution_std
batch_size = args.batch_size
model_name = args.model
capacity = args.capacity
dec_layers = args.dec_layers
enc_layers = args.enc_layers
ign_layers = args.ign_layers
edge_layers = args.edge_layers
dec_residual = args.dec_residual

# --------------------
# Params
# --------------------
params = {
    'distribution_std': distribution_std,
    'variational_beta': variational_beta,
    'capacity': capacity,
    'dec_layers': dec_layers,
    'enc_layers': enc_layers,
    'ign_layers': ign_layers,
    'edge_layers': edge_layers,
    'dec_residual': dec_residual,
    # edge loss shaping
    "edge_pos_weight": 1,
    "edge_neg_weight": 1,
    "edge_loss_lambda": 1,
    # optional ignition weight (if used inside your loss)
    "ign_loss_lambda": 0.005,
}
models = {
    "v1": GRAPH_VAE,
    "v2": GRAPH_VAE_V2,
    "v3": GRAPH_VAE_V3,  # UNDIRECTED variant
}

# --------------------
# Data
# --------------------
root = "../data/sub20/graphs"
dataset = GraphDatasetV3(root=root)

print("Splitting the dataset")
train_size = int(0.8 * len(dataset))
val_size   = int(0.1 * len(dataset))
test_size  = len(dataset) - train_size - val_size
print(f"Train size: {train_size}, Val size: {val_size}")

g_split = torch.Generator().manual_seed(SEED)
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size], generator=g_split
)

# If you ever see CUDA + fork issues, set num_workers=0 or ensure CPU-only ops in __getitem__
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=6, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# --------------------
# Model / Optim
# --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = 5  # pos2 + alt + slope + fuel_scalar (as built in your dataset encoder inputs)
model = models[model_name](input_dim, latent_dim, params, dataset.template).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

run_name = f"{model.name}_UNDIR_ld={latent_dim}_cap={capacity}_vb={variational_beta}_bs={batch_size}_lr={lr}_dec_layers={dec_layers}_enc_layers={enc_layers}_ign_layers={ign_layers}_edge_layers={edge_layers}"
writer = SummaryWriter(log_dir=f"runs/{run_name}")

# --------------------
# Train
# --------------------
train_loss = []
train_recon_loss = []
train_kl_loss = []

val_loss = []
val_recon_loss = []
val_kl_loss = []

model.train()
global_step = 0
val_global_step = 0
ckpt_base = f'networks/weights/{model.name}_UNDIR_latent={latent_dim}_lr={lr}_epochs={epochs}_variational_beta={variational_beta}_capacity={capacity}_dec_layers={dec_layers}_enc_layers={enc_layers}_ign_layers={ign_layers}_edge_layers={edge_layers}'
best_val_loss = float("inf")

try:
    for epoch in tqdm(range(epochs)):
        train_epoch_loss = 0.0
        train_epoch_recon_loss = 0.0
        train_epoch_kl_loss = 0.0

        # optional sub-loss trackers (edge/ign) if your loss returns them
        train_epoch_edge_loss = 0.0
        train_epoch_ign_loss  = 0.0

        n = 0
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
            optimizer.zero_grad(set_to_none=True)
            batch = batch.to(device, non_blocking=True)

            output, mu, log = model(batch.x, batch.edge_index_enc, batch.batch)
            
            recon_loss, kl_loss, loss, loss_dict = model.loss(output, batch, mu, log, dataset.template)

            loss.backward()
            optimizer.step()

            train_epoch_loss       += float(loss.item())
            train_epoch_recon_loss += float(recon_loss.item())
            train_epoch_kl_loss    += float(kl_loss.item())

            # grab optional components if present
            if isinstance(loss_dict, dict):
                if "edge" in loss_dict and loss_dict["edge"] is not None:
                    train_epoch_edge_loss += float(loss_dict["edge"].item())
                if "ign" in loss_dict and loss_dict["ign"] is not None:
                    train_epoch_ign_loss  += float(loss_dict["ign"].item())

            # quick on-batch metrics (edges only)
            with torch.no_grad():
                mb_metrics = metrics_on_batch_tm(model, batch, dataset.template, device, edge_thr=0.5)

            # --- log to TB ---
            writer.add_scalar("train/loss/total", loss.item(), global_step)
            writer.add_scalar("train/loss/recon", recon_loss.item(), global_step)
            writer.add_scalar("train/loss/kl",    kl_loss.item(), global_step)
            if isinstance(loss_dict, dict) and "edge" in loss_dict and loss_dict["edge"] is not None:
                writer.add_scalar("train/loss/edge", loss_dict["edge"].item(), global_step)
            if isinstance(loss_dict, dict) and "ign" in loss_dict and loss_dict["ign"] is not None:
                writer.add_scalar("train/loss/ign",  loss_dict["ign"].item(),  global_step)

            for k, v in mb_metrics.items():  # edge metrics only
                writer.add_scalar(f"train/{k}", v, global_step)

            global_step += 1
            n += 1

        # epoch means
        train_loss.append(train_epoch_loss / max(1, n))
        train_recon_loss.append(train_epoch_recon_loss / max(1, n))
        train_kl_loss.append(train_epoch_kl_loss / max(1, n))

        # ----------------- Validation -----------------
        model.eval()
        val_epoch_loss = 0.0
        val_epoch_recon_loss = 0.0
        val_epoch_kl_loss = 0.0

        val_epoch_edge_loss = 0.0
        val_epoch_ign_loss  = 0.0

        m = 0
        val_metric_sums = defaultdict(float)
        val_metric_count = 0

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                batch = batch.to(device, non_blocking=True)
                output, mu, log = model(batch.x, batch.edge_index_enc, batch.batch)

                recon_loss, kl_loss, loss, loss_dict = model.loss(output, batch, mu, log, dataset.template)

                val_epoch_loss       += float(loss.item())
                val_epoch_recon_loss += float(recon_loss.item())
                val_epoch_kl_loss    += float(kl_loss.item())

                if isinstance(loss_dict, dict):
                    if "edge" in loss_dict and loss_dict["edge"] is not None:
                        val_epoch_edge_loss += float(loss_dict["edge"].item())
                    if "ign" in loss_dict and loss_dict["ign"] is not None:
                        val_epoch_ign_loss  += float(loss_dict["ign"].item())

                mb_metrics_val = metrics_on_batch_tm(model, batch, dataset.template, device, edge_thr=0.5)
                for k, v in mb_metrics_val.items():
                    writer.add_scalar(f"val/{k}", v, val_global_step)
                    val_metric_sums[k] += v
                val_metric_count += 1
                val_global_step += 1
                m += 1

        val_loss.append(val_epoch_loss / max(1, m))
        val_recon_loss.append(val_epoch_recon_loss / max(1, m))
        val_kl_loss.append(val_epoch_kl_loss / max(1, m))

        if val_metric_count > 0:
            for k, s in val_metric_sums.items():
                writer.add_scalar(f"epoch/val_{k}", s / val_metric_count, epoch)

        writer.add_scalar("epoch/val_loss",       val_epoch_loss / max(1, m), epoch)
        writer.add_scalar("epoch/val_recon_loss", val_epoch_recon_loss / max(1, m), epoch)
        writer.add_scalar("epoch/val_kl_loss",    val_epoch_kl_loss / max(1, m), epoch)
        if val_epoch_edge_loss > 0:
            writer.add_scalar("epoch/val_edge_loss", val_epoch_edge_loss / max(1, m), epoch)
        if val_epoch_ign_loss > 0:
            writer.add_scalar("epoch/val_ign_loss",  val_epoch_ign_loss  / max(1, m), epoch)

        val_avg = val_epoch_loss / max(1, m)
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "val_global_step": val_global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_avg,
                "params": params,
            }, f"{ckpt_base}_best.pt")
            print(f"[epoch {epoch}] New best val loss {val_avg:.6f} -> saved to {ckpt_base}_best.pt")

        model.train()

    # --------------------
    # Plots
    # --------------------
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend()
    plt.savefig(f'experiments/{model.name}_UNDIR_train_loss_latent={latent_dim}_lr={lr}_epochs={epochs}_vb={variational_beta}_cap={capacity}_dec_layers={dec_layers}_enc_layers={enc_layers}_ign_layers={ign_layers}_edge_layers={edge_layers}.png')
    plt.clf()

    plt.plot(train_recon_loss, label='train_recon_loss')
    plt.plot(val_recon_loss, label='val_recon_loss')
    plt.legend()
    plt.savefig(f'experiments/{model.name}_UNDIR_train_recon_loss_latent={latent_dim}_lr={lr}_epochs={epochs}_vb={variational_beta}_cap={capacity}_dec_layers={dec_layers}_enc_layers={enc_layers}_ign_layers={ign_layers}_edge_layers={edge_layers}.png')
    plt.clf()

    plt.plot(train_kl_loss, label='train_kl_loss')
    plt.plot(val_kl_loss, label='val_kl_loss')
    plt.legend()
    plt.savefig(f'experiments/{model.name}_UNDIR_train_kl_loss_latent={latent_dim}_lr={lr}_epochs={epochs}_vb={variational_beta}_cap={capacity}_dec_layers={dec_layers}_enc_layers={enc_layers}_ign_layers={ign_layers}_edge_layers={edge_layers}.png')
    plt.clf()

    # --------------------
    # Save final weights
    # --------------------
    torch.save(
        model.state_dict(),
        f'networks/weights/{model.name}_UNDIR_latent={latent_dim}_lr={lr}_epochs={epochs}_vb={variational_beta}_cap={capacity}_dec_layers={dec_layers}_enc_layers={enc_layers}_ign_layers={ign_layers}_edge_layers={edge_layers}.pt'
    )

    # --------------------
    # Final eval & dump
    # --------------------
    model.eval()
    metrics_train = evaluate_all(model, train_loader, dataset.template, device)
    save_path = f"experiments/{model.name}_UNDIR_results_latent={latent_dim}_lr={lr}_epochs={epochs}_vb={variational_beta}_cap={capacity}_dec_layers={dec_layers}_enc_layers={enc_layers}_ign_layers={ign_layers}_edge_layers={edge_layers}_train.json"
    with open(save_path, "w") as f:
        json.dump(metrics_train, f, indent=2)
    print(f"Saved metrics_train to {save_path}")

    metrics_val = evaluate_all(model, val_loader, dataset.template, device)
    save_path = f"experiments/{model.name}_UNDIR_results_latent={latent_dim}_lr={lr}_epochs={epochs}_vb={variational_beta}_cap={capacity}_dec_layers={dec_layers}_enc_layers={enc_layers}_ign_layers={ign_layers}_edge_layers={edge_layers}_val.json"
    with open(save_path, "w") as f:
        json.dump(metrics_val, f, indent=2)
    print(f"Saved metrics_val to {save_path}")

except KeyboardInterrupt:
    # graceful plots & saves on interrupt
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend()
    plt.savefig(f'experiments/{model.name}_UNDIR_train_loss_latent={latent_dim}_lr={lr}_epochs={epochs}_vb={variational_beta}_cap={capacity}_dec_layers={dec_layers}_enc_layers={enc_layers}_ign_layers={ign_layers}_edge_layers={edge_layers}.png')
    plt.clf()

    plt.plot(train_recon_loss, label='train_recon_loss')
    plt.plot(val_recon_loss, label='val_recon_loss')
    plt.legend()
    plt.savefig(f'experiments/{model.name}_UNDIR_train_recon_loss_latent={latent_dim}_lr={lr}_epochs={epochs}_vb={variational_beta}_cap={capacity}_dec_layers={dec_layers}_enc_layers={enc_layers}_ign_layers={ign_layers}_edge_layers={edge_layers}.png')
    plt.clf()

    plt.plot(train_kl_loss, label='train_kl_loss')
    plt.plot(val_kl_loss, label='val_kl_loss')
    plt.legend()
    plt.savefig(f'experiments/{model.name}_UNDIR_train_kl_loss_latent={latent_dim}_lr={lr}_epochs={epochs}_vb={variational_beta}_cap={capacity}_dec_layers={dec_layers}_enc_layers={enc_layers}_ign_layers={ign_layers}_edge_layers={edge_layers}.png')
    plt.clf()

    torch.save(
        model.state_dict(),
        f'networks/weights/{model.name}_UNDIR_latent={latent_dim}_lr={lr}_epochs={epochs}_vb={variational_beta}_cap={capacity}_dec_layers={dec_layers}_enc_layers={enc_layers}_ign_layers={ign_layers}_edge_layers={edge_layers}.pt'
    )

    model.eval()
    metrics_train = evaluate_all(model, train_loader, dataset.template, device)
    save_path = f"experiments/{model.name}_UNDIR_results_latent={latent_dim}_lr={lr}_epochs={epochs}_vb={variational_beta}_cap={capacity}_dec_layers={dec_layers}_enc_layers={enc_layers}_ign_layers={ign_layers}_edge_layers={edge_layers}_train.json"
    with open(save_path, "w") as f:
        json.dump(metrics_train, f, indent=2)
    print(f"Saved metrics_train to {save_path}")

    metrics_val = evaluate_all(model, val_loader, dataset.template, device)
    save_path = f"experiments/{model.name}_UNDIR_results_latent={latent_dim}_lr={lr}_epochs={epochs}_vb={variational_beta}_cap={capacity}_dec_layers={dec_layers}_enc_layers={enc_layers}_ign_layers={ign_layers}_edge_layers={edge_layers}_val.json"
    with open(save_path, "w") as f:
        json.dump(metrics_val, f, indent=2)
    print(f"Saved metrics_val to {save_path}")

finally:
    writer.flush()
    writer.close()
