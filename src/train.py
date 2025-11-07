import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.datasets as D
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges, to_dense_adj
from utils import GraphDataset, reconstruct_matrix, GraphDatasetV3, evaluate_all, metrics_on_batch_tm, estimate_edge_pos_weight, set_seed
from networks.graph_vae import GRAPH_VAE
from networks.graph_vae_v2 import GRAPH_VAE_V2
from networks.graph_vae_v3 import GRAPH_VAE_V3
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

SEED = 12345
set_seed(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # or ":4096:8"
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument('--latent_dim', type=int, required=True)
parser.add_argument('--epochs', type=int, required=True, default = 100)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--distribution_std', type=float, default=0.1)
parser.add_argument('--variational_beta', type=float, default=1.)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--model', type=str, default='v1')
parser.add_argument('--capacity', type=int, default=8)
parser.add_argument('--dec_layers', type=int, default=2)
parser.add_argument('--edge_layers', type=int, default=2)
parser.add_argument('--dec_residual', type=bool, default=True)

# Recover command line arguments
args = parser.parse_args()
latent_dim = args.latent_dim
epochs = args.epochs
lr = args.lr
variational_beta = args.variational_beta
distribution_std = args.distribution_std
batch_size = args.batch_size
model_name = args.model
capacity = args.capacity
dec_layers= args.dec_layers
edge_layers= args.edge_layers
dec_residual= args.dec_residual

# Define the parameters
params = {
    'distribution_std': distribution_std,
    'variational_beta': variational_beta,
    'capacity': capacity,
    # ---- added depth controls for GRAPH_VAE_V3 decoder ----
    'dec_layers': args.dec_layers,
    'edge_layers': args.edge_layers,
    'dec_residual': args.dec_residual,
}
params.update({
    "edge_pos_weight": 5.4,   # you already use this
    "edge_neg_weight": 1.5,    # NEW: >1 penalizes negatives → higher precision
    "edge_loss_lambda": 2.5,   # NEW: scales the whole edge loss
})

models = {
    "v1": GRAPH_VAE,
    "v2": GRAPH_VAE_V2,
    "v3": GRAPH_VAE_V3,
}

# Load the dataset
root = "/home/lucas/graph_scenarios/data/sub20/graphs"
dataset = GraphDatasetV3(root=root)

# Split the dataset into training, validation, and test sets
print("Splitting the dataset")
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
print(f"Train size: {train_size}, Val size: {val_size}")

# Seed split
g_split = torch.Generator().manual_seed(SEED)
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=g_split)

# Create the DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Define the model and the optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = 5 #dataset.num_features
model = models[model_name](input_dim, latent_dim, params, dataset.template).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Print number of params:
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

run_name = f"{model.name}_ld={latent_dim}_cap={capacity}_vb={variational_beta}_bs={batch_size}_lr={lr}_dec_layers={dec_layers}_edge_layers={edge_layers}"
writer = SummaryWriter(log_dir=f"runs/{run_name}")

# edge_pos_weight, base_rate = estimate_edge_pos_weight(train_loader, dataset.template, device)
# print(f"Edge base-rate ~{base_rate:.4f} → pos_weight ≈ {edge_pos_weight:.1f}")

# Train the model
train_loss = []
train_recon_loss = []
train_kl_loss = []

val_loss = []
val_recon_loss = []
val_kl_loss = []
model.train()
global_step = 0
val_global_step = 0
ckpt_base = f'networks/weights/{model.name}_latent={latent_dim}_lr={lr}_epochs={epochs}_variational_beta={variational_beta}_capacity={capacity}_dec_layers={dec_layers}_edge_layers={edge_layers}'
best_val_loss = float("inf")

try:
    for epoch in tqdm(range(epochs)):
        train_epoch_loss = 0
        train_epoch_recon_loss = 0
        train_epoch_kl_loss = 0
        train_epoch_node_loss = 0
        train_epoch_edge_loss = 0
        n = 0
        for i, batch in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            batch = batch.to(device)
            output, mu, log = model(batch.x, batch.edge_index_enc, batch.batch)
            graph_list = batch.to_data_list()
            recon_loss, kl_loss, loss, node_loss, edge_loss = model.loss(output, batch, mu, log, dataset.template)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()
            train_epoch_recon_loss += recon_loss.item()
            train_epoch_kl_loss += kl_loss.item() 
            train_epoch_node_loss += node_loss.item() 
            train_epoch_edge_loss += edge_loss.item() 
            
            with torch.no_grad():
                mb_metrics = metrics_on_batch_tm(model, batch, dataset.template, device,
                                                node_thr=0.5, edge_thr=0.5)
            # --- log to TB ---
            writer.add_scalar("train/loss/total", loss.item(), global_step)
            writer.add_scalar("train/loss/recon", recon_loss.item(), global_step)
            writer.add_scalar("train/loss/kl",    kl_loss.item(), global_step)
            writer.add_scalar("train/loss/node",    node_loss.item(), global_step)
            writer.add_scalar("train/loss/edge",    edge_loss.item(), global_step)
            for k, v in mb_metrics.items():
                writer.add_scalar(f"train/{k}", v, global_step)
            global_step += 1
            n += 1

        train_loss.append(train_epoch_loss / n)
        train_recon_loss.append(train_epoch_recon_loss / n)
        train_kl_loss.append(train_epoch_kl_loss / n)

        model.eval()
        val_epoch_loss = 0
        val_epoch_recon_loss = 0
        val_epoch_kl_loss = 0
        m = 0
        val_metric_sums = defaultdict(float)   # accumulate per-batch metrics to average later
        val_metric_count = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                batch = batch.to(device, non_blocking=True)
                output, mu, log = model(batch.x, batch.edge_index_enc, batch.batch)
                graph_list = batch.to_data_list()
                #true_adjacency = reconstruct_matrix(graph_list).to(device)
                recon_loss, kl_loss, loss, _, _ = model.loss(output, batch, mu, log, dataset.template)
                val_epoch_loss += loss.item()
                val_epoch_recon_loss += recon_loss.item()
                val_epoch_kl_loss += kl_loss.item()
                mb_metrics_val = metrics_on_batch_tm(model, batch, dataset.template, device,
                                                    node_thr=0.5, edge_thr=0.5)
                for k, v in mb_metrics_val.items():
                    writer.add_scalar(f"val/{k}", v, val_global_step)
                    val_metric_sums[k] += v
                val_metric_count += 1
                val_global_step += 1
                m += 1
            val_loss.append(val_epoch_loss / m)
            val_recon_loss.append(val_epoch_recon_loss / m)
            val_kl_loss.append(val_epoch_kl_loss / m)
            if val_metric_count > 0:
                for k, s in val_metric_sums.items():
                    writer.add_scalar(f"epoch/val_{k}", s / val_metric_count, epoch)
            writer.add_scalar("epoch/val_loss",       val_epoch_loss / max(1, m), epoch)
            writer.add_scalar("epoch/val_recon_loss", val_epoch_recon_loss / max(1, m), epoch)
            writer.add_scalar("epoch/val_kl_loss",    val_epoch_kl_loss / max(1, m), epoch)
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
    # Plot the training and validation losses
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend()
    plt.savefig(f'experiments/{model.name}_train_loss_latent={latent_dim}_lr={lr}_epochs={epochs}_variational_beta={variational_beta}_capacity={capacity}_dec_layers={dec_layers}_edge_layers={edge_layers}.png')
    plt.clf()


    plt.plot(train_recon_loss, label='train_recon_loss')
    plt.plot(val_recon_loss, label='val_recon_loss')
    plt.legend()
    plt.savefig(f'experiments/{model.name}_train_recon_loss_latent={latent_dim}_lr={lr}_epochs={epochs}_variational_beta={variational_beta}_capacity={capacity}_dec_layers={dec_layers}_edge_layers={edge_layers}.png')
    plt.clf()


    plt.plot(train_kl_loss, label='train_kl_loss')
    plt.plot(val_kl_loss, label='val_kl_loss')
    plt.legend()
    plt.savefig(f'experiments/{model.name}_train_kl_loss_latent={latent_dim}_lr={lr}_epochs={epochs}_variational_beta={variational_beta}_capacity={capacity}_dec_layers={dec_layers}_edge_layers={edge_layers}.png')
    plt.clf()


    # Save the model
    torch.save(model.state_dict(), f'networks/weights/{model.name}_latent={latent_dim}_lr={lr}_epochs={epochs}_variational_beta={variational_beta}_capacity={capacity}_dec_layers={dec_layers}_edge_layers={edge_layers}.pt')

    # ---------- run it ----------
    model.eval()
    metrics_train = evaluate_all(model, train_loader, dataset.template, device)

    # Save
    save_path = f"experiments/{model.name}_results_latent={latent_dim}_lr={lr}_epochs={epochs}_variational_beta={variational_beta}_capacity={capacity}_dec_layers={dec_layers}_edge_layers={edge_layers}_train.json"
    with open(save_path, "w") as f:
        json.dump(metrics_train, f, indent=2)
    print(f"Saved metrics_train to {save_path}")

    metrics_val = evaluate_all(model, val_loader, dataset.template, device)

    # Save
    save_path = f"experiments/{model.name}_results_latent={latent_dim}_lr={lr}_epochs={epochs}_variational_beta={variational_beta}_capacity={capacity}_dec_layers={dec_layers}_edge_layers={edge_layers}_val.json"
    with open(save_path, "w") as f:
        json.dump(metrics_val, f, indent=2)
    print(f"Saved metrics_val to {save_path}")

except KeyboardInterrupt:
    # Plot the training and validation losses
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend()
    plt.savefig(f'experiments/{model.name}_train_loss_latent={latent_dim}_lr={lr}_epochs={epochs}_variational_beta={variational_beta}_capacity={capacity}_dec_layers={dec_layers}_edge_layers={edge_layers}.png')
    plt.clf()


    plt.plot(train_recon_loss, label='train_recon_loss')
    plt.plot(val_recon_loss, label='val_recon_loss')
    plt.legend()
    plt.savefig(f'experiments/{model.name}_train_recon_loss_latent={latent_dim}_lr={lr}_epochs={epochs}_variational_beta={variational_beta}_capacity={capacity}_dec_layers={dec_layers}_edge_layers={edge_layers}.png')
    plt.clf()


    plt.plot(train_kl_loss, label='train_kl_loss')
    plt.plot(val_kl_loss, label='val_kl_loss')
    plt.legend()
    plt.savefig(f'experiments/{model.name}_train_kl_loss_latent={latent_dim}_lr={lr}_epochs={epochs}_variational_beta={variational_beta}_capacity={capacity}_dec_layers={dec_layers}_edge_layers={edge_layers}.png')
    plt.clf()


    # Save the model
    torch.save(model.state_dict(), f'networks/weights/{model.name}_latent={latent_dim}_lr={lr}_epochs={epochs}_variational_beta={variational_beta}_capacity={capacity}_dec_layers={dec_layers}_edge_layers={edge_layers}.pt')

    # ---------- run it ----------
    model.eval()
    metrics_train = evaluate_all(model, train_loader, dataset.template, device)

    # Save
    save_path = f"experiments/{model.name}_results_latent={latent_dim}_lr={lr}_epochs={epochs}_variational_beta={variational_beta}_capacity={capacity}_dec_layers={dec_layers}_edge_layers={edge_layers}_train.json"
    with open(save_path, "w") as f:
        json.dump(metrics_train, f, indent=2)
    print(f"Saved metrics_train to {save_path}")

    metrics_val = evaluate_all(model, val_loader, dataset.template, device)

    # Save
    save_path = f"experiments/{model.name}_results_latent={latent_dim}_lr={lr}_epochs={epochs}_variational_beta={variational_beta}_capacity={capacity}_dec_layers={dec_layers}_edge_layers={edge_layers}_val.json"
    with open(save_path, "w") as f:
        json.dump(metrics_val, f, indent=2)
    print(f"Saved metrics_val to {save_path}")

finally:
    # always flush/close TB
    writer.flush()
    writer.close()