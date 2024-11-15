import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.datasets as D
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges, to_dense_adj
from utils import GraphDataset, reconstruct_matrix
from networks.graph_vae import GRAPH_VAE
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch.utils.data import random_split
import argparse
import matplotlib.pyplot as plt

from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--latent_dim', type=int, required=True)
parser.add_argument('--epochs', type=int, required=True, default = 100)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--distribution_std', type=float, default=0.1)
parser.add_argument('--variational_beta', type=float, default=1.)
parser.add_argument('--batch_size', type=int, default=16)

# Recover command line arguments
args = parser.parse_args()
latent_dim = args.latent_dim
epochs = args.epochs
lr = args.lr
variational_beta = args.variational_beta
distribution_std = args.distribution_std
batch_size = args.batch_size

# Define the parameters
params = {
    'distribution_std': distribution_std,
    'variational_beta': variational_beta,
}

# Load the dataset
dataset = GraphDataset(root='../data/sub20/graphs')

# Split the dataset into training, validation, and test sets
print("Splitting the dataset")
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
print(f"Train size: {train_size}, Val size: {val_size}")

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create the DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the model and the optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = dataset.num_features
model = GRAPH_VAE(input_dim, latent_dim, params).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Print number of params:
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

# Train the model
train_loss = []
train_recon_loss = []
train_kl_loss = []

val_loss = []
val_recon_loss = []
val_kl_loss = []
for _ in tqdm(range(epochs)):
    model.training = True
    train_epoch_loss = 0
    train_epoch_recon_loss = 0
    train_epoch_kl_loss = 0
    for i, batch in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        batch = batch.to(device)
        output, mu, log = model(batch.x, batch.edge_index, batch.batch)
        graph_list = batch.to_data_list()
        true_adjacency = reconstruct_matrix(graph_list).to(device)
        recon_loss, kl_loss, loss = model.loss(output, true_adjacency, mu, log)
        loss.backward()
        optimizer.step()
        train_epoch_loss += loss.item()
        train_epoch_recon_loss += recon_loss.item()
        train_epoch_kl_loss += kl_loss.item()
    train_loss.append(train_epoch_loss)
    train_recon_loss.append(train_epoch_recon_loss)
    train_kl_loss.append(train_epoch_kl_loss)

    model.training = False
    val_epoch_loss = 0
    val_epoch_recon_loss = 0
    val_epoch_kl_loss = 0
    for i, batch in enumerate(val_loader):
        batch = batch.to(device)
        output, mu, log = model(batch.x, batch.edge_index, batch.batch)
        graph_list = batch.to_data_list()
        true_adjacency = reconstruct_matrix(graph_list).to(device)
        recon_loss, kl_loss, loss = model.loss(output, true_adjacency, mu, log)
        val_epoch_loss += loss.item()
        val_epoch_recon_loss += recon_loss.item()
        val_epoch_kl_loss += kl_loss.item()
    val_loss.append(val_epoch_loss)
    val_recon_loss.append(val_epoch_recon_loss)
    val_kl_loss.append(val_epoch_kl_loss)

# Plot the training and validation losses
plt.plot(train_loss[1:], label='train_loss')
plt.plot(val_loss[1:], label='val_loss')
plt.legend()
plt.savefig(f'experiments/loss_latent={latent_dim}_lr={lr}_epochs={epochs}.png')
plt.clf()

plt.plot(train_recon_loss[1:], label='train_recon_loss')
plt.plot(val_recon_loss[1:], label='val_recon_loss')
plt.legend()
plt.savefig(f'experiments/recon_loss_latent={latent_dim}_lr={lr}_epochs={epochs}.png')
plt.clf()

plt.plot(train_kl_loss[1:], label='train_kl_loss')
plt.plot(val_kl_loss[1:], label='val_kl_loss')
plt.legend()
plt.savefig(f'experiments/kl_loss_latent={latent_dim}_lr={lr}_epochs={epochs}.png')
plt.clf()

# Save the model
torch.save(model.state_dict(), f'networks/weights/model_latent={latent_dim}_lr={lr}_epochs={epochs}.pt')




