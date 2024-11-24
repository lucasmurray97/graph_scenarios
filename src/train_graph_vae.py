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
from networks.graph_vae_v2 import GRAPH_VAE_V2
from networks.gae import GRAPH_VAE_V3
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch.utils.data import random_split
import argparse
import matplotlib.pyplot as plt
from torch_geometric.utils import negative_sampling

from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--latent_dim', type=int, required=True)
parser.add_argument('--epochs', type=int, required=True, default = 100)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--distribution_std', type=float, default=0.1)
parser.add_argument('--variational_beta', type=float, default=1.)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--model', type=str, default='v1')
parser.add_argument('--capacity', type=int, default=8)
parser.add_argument('--loss', type=str, default="bce")
parser.add_argument('--alpha', type=float, default=0.75)
parser.add_argument('--gamma', type=float, default=1.0)

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
loss = args.loss
alpha = args.alpha
gamma = args.gamma

# Define the parameters
params = {
    'distribution_std': distribution_std,
    'variational_beta': variational_beta,
    'capacity': capacity,
    'loss': loss,
    'alpha': alpha,
    'gamma': gamma,
}

models = {
    "v1": GRAPH_VAE,
    "v2": GRAPH_VAE_V2,
    "v3": GRAPH_VAE_V3,
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
model = models[model_name](input_dim, latent_dim, params).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Print number of params:
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Train the model
train_loss = []
train_recon_loss = []
train_kl_loss = []

val_loss = []
val_recon_loss = []
val_kl_loss = []
#model.train_()
for _ in tqdm(range(epochs)):
    train_epoch_loss = 0
    train_epoch_recon_loss = 0
    train_epoch_kl_loss = 0
    n = 0
    for i, batch in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        batch = batch.to(device)
        output, mu, log = model(batch.x, batch.edge_index, batch.batch)
        graph_list = batch.to_data_list()
        recon_loss, kl_loss, loss = model.loss(output, mu, log, batch.to_data_list())  
        loss.backward()
        optimizer.step()
        train_epoch_loss += loss.item()
        train_epoch_recon_loss += recon_loss.item()
        train_epoch_kl_loss += kl_loss.item() 
        n += 1
    train_loss.append(train_epoch_loss / n)
    train_recon_loss.append(train_epoch_recon_loss / n)
    train_kl_loss.append(train_epoch_kl_loss / n)

    val_epoch_loss = 0
    val_epoch_recon_loss = 0
    val_epoch_kl_loss = 0
    m = 0
    for i, batch in enumerate(val_loader):
        batch = batch.to(device)
        output, mu, log = model(batch.x, batch.edge_index, batch.batch)
        recon_loss, kl_loss, loss = model.loss(output, mu, log, batch.to_data_list())
        val_epoch_loss += loss.item()
        val_epoch_recon_loss += recon_loss.item()
        val_epoch_kl_loss += kl_loss.item()
        m += 1
    val_loss.append(val_epoch_loss / m)
    val_recon_loss.append(val_epoch_recon_loss / m)
    val_kl_loss.append(val_epoch_kl_loss / m)

# Plot the training and validation losses
plt.plot(train_loss, label='train_loss')
#plt.plot(val_loss, label='val_loss')
plt.legend()
plt.savefig(f'experiments/{model.name}_train_loss_latent={latent_dim}_lr={lr}_epochs={epochs}_variational_beta={variational_beta}_capacity={capacity}_alpha={alpha}_gamma={gamma}.png')
plt.clf()

# Plot the training and validation losses
#plt.plot(train_loss, label='train_loss')
plt.plot(val_loss, label='val_loss')
plt.legend()
plt.savefig(f'experiments/{model.name}_val_loss_latent={latent_dim}_lr={lr}_epochs={epochs}_variational_beta={variational_beta}_capacity={capacity}_alpha={alpha}_gamma={gamma}.png')
plt.clf()

plt.plot(train_recon_loss, label='train_recon_loss')
#plt.plot(val_recon_loss, label='val_recon_loss')
plt.legend()
plt.savefig(f'experiments/{model.name}_train_recon_loss_latent={latent_dim}_lr={lr}_epochs={epochs}_variational_beta={variational_beta}_capacity={capacity}_alpha={alpha}_gamma={gamma}.png')
plt.clf()

#plt.plot(train_recon_loss, label='train_recon_loss')
plt.plot(val_recon_loss, label='val_recon_loss')
plt.legend()
plt.savefig(f'experiments/{model.name}_val_recon_loss_latent={latent_dim}_lr={lr}_epochs={epochs}_variational_beta={variational_beta}_capacity={capacity}_alpha={alpha}_gamma={gamma}.png')
plt.clf()


plt.plot(train_kl_loss, label='train_kl_loss')
#plt.plot(val_kl_loss, label='val_kl_loss')
plt.legend()
plt.savefig(f'experiments/{model.name}_train_kl_loss_latent={latent_dim}_lr={lr}_epochs={epochs}_variational_beta={variational_beta}_capacity={capacity}_alpha={alpha}_gamma={gamma}.png')
plt.clf()

#plt.plot(train_kl_loss, label='train_kl_loss')
plt.plot(val_kl_loss, label='val_kl_loss')
plt.legend()
plt.savefig(f'experiments/{model.name}_val_kl_loss_latent={latent_dim}_lr={lr}_epochs={epochs}_variational_beta={variational_beta}_capacity={capacity}_alpha={alpha}_gamma={gamma}.png')
plt.clf()


# Save the model
torch.save(model.state_dict(), f'networks/weights/{model.name}_latent={latent_dim}_lr={lr}_epochs={epochs}_variational_beta={variational_beta}_capacity={capacity}_alpha={alpha}_gamma={gamma}.pt')

# Evaluate the model
accuracy = []
precision = []
recall = []
f1 = []
model.eval()
model.training = False
for i, batch in enumerate(val_loader):
    batch = batch.to(device)
    output, mu, log = model(batch.x, batch.edge_index, batch.batch)
    true_vector = model.generate_binary_edge_vector_directed(batch.to_data_list())
    # Binarize predictions
    y_pred_binary = (output >= 0.5).float()   
    y_true = model.generate_binary_edge_vector_directed(batch.to_data_list())
    y_pred = (output >= 0.5).float()
    TP = torch.sum((y_true == 1) & (y_pred_binary == 1)).item()  # True Positives
    FP = torch.sum((y_true == 0) & (y_pred_binary == 1)).item()  # False Positives
    FN = torch.sum((y_true == 1) & (y_pred_binary == 0)).item()  # False Negatives
    TN = torch.sum((y_true == 0) & (y_pred_binary == 0)).item()  # True Negatives
    # Metrics calculations
    acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    pre = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_ = 2 * (pre * rec) / (pre + rec) if (pre + rec) > 0 else 0.0
    # Exact match accuracy (all labels match)
    accuracy.append(acc)
    precision.append(pre)
    recall.append(rec)
    f1.append(f1_)
# write down results into a json
import json
results = {
    "accuracy": sum(accuracy)/len(accuracy),
    "precision": sum(precision)/len(precision),
    "recall": sum(recall)/len(recall),
    "f1": sum(f1)/len(f1),
}
# Save the results
with open(f'experiments/{model.name}_results_latent={latent_dim}_lr={lr}_epochs={epochs}_variational_beta={variational_beta}_capacity={capacity}_alpha={alpha}_gamma={gamma}.json', 'w') as f:
    json.dump(results, f)





