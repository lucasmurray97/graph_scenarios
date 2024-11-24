import torch
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn.pool import global_mean_pool, global_add_pool
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
# Add .. to sys path
import sys
sys.path.append("..")
from utils import generate_grid_edges


class GRAPH_VAE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, params):
        super(GRAPH_VAE, self).__init__()
        self.training = True
        self.distribution_std = params['distribution_std']
        self.variational_beta = params['variational_beta']
        self.name = "GRAPH_VAE"
        self.loss_function = params['loss']
        self.alpha = params['alpha']
        self.gamma = params['gamma']
        self.capacity = params['capacity']

        # Define MLP for GINConv
        def make_mlp(input_dim, hidden_dim, final=False):
            return Sequential(
                Linear(input_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU() if not final else torch.nn.Identity()
            )

        # Encoder:
        self.conv1_mu = GINConv(make_mlp(input_dim, self.capacity))
        self.conv2_mu = GINConv(make_mlp(self.capacity, self.capacity * 2))
        self.conv3_mu = GINConv(make_mlp(self.capacity * 2, self.capacity * 2))
        self.conv4_mu = GINConv(make_mlp(self.capacity * 2, latent_dim, True))

        self.conv1_log = GINConv(make_mlp(input_dim, self.capacity))
        self.conv2_log = GINConv(make_mlp(self.capacity, self.capacity * 2))
        self.conv3_log = GINConv(make_mlp(self.capacity * 2, self.capacity * 2))
        self.conv4_log = GINConv(make_mlp(self.capacity * 2, latent_dim, True))
        self.pool = global_mean_pool

        # Decoder:
        self.fc1 = torch.nn.Linear(latent_dim, 128)
        self.fc2 = torch.nn.Linear(128, 256)
        self.fc3 = torch.nn.Linear(256, 512)
        self.fc4 = torch.nn.Linear(512, 2964)

        self.grid_edges = generate_grid_edges(20)
        


    def encode(self, x, edge_index, batch=torch.Tensor([0])):
        #print(x.shape, edge_index.shape, batch.shape)
        mu = self.conv1_mu(x, edge_index).relu()
        #print(mu)
        #print(mu.shape)
        mu = self.conv2_mu(mu, edge_index).relu()
        #print(mu.shape)
        mu = self.conv3_mu(mu, edge_index).relu()
        #print(mu.shape)
        mu = self.conv4_mu(mu, edge_index)
        #print(mu)
        mu = self.pool(mu, batch)
        #print(mu)
        log = self.conv1_log(x, edge_index).relu()
        #print(x.shape)
        log = self.conv2_log(log, edge_index).relu()
        #print(x.shape)
        log = self.conv3_log(log, edge_index).relu()
        #print(x.shape)
        log = self.conv4_log(log, edge_index)
        log = self.pool(log, batch)
        #print(x.shape)
        return mu, log
    
    def decode(self, z):
        z = F.relu(self.fc1(z))
        #print(z.shape)
        z = F.relu(self.fc2(z))
        #print(z)
        z = F.relu(self.fc3(z))
        #print(z)
        z = self.fc4(z).sigmoid()
        #print(z)
        return z
    
    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.normal(torch.zeros(std.shape), self.distribution_std).to("cuda")
            sample = mu + (eps * std)
            return sample
        else:
            return mu
        
    def loss(self, output, mu, logvar, batch):
        recon_loss = self.recon_loss(output, batch)
        kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss, kldivergence, recon_loss + self.variational_beta * kldivergence
    
    def forward(self, x, edge_index, batch):
        mu, logvar = self.encode(x, edge_index, batch)
        z = self.latent_sample(mu, logvar)
        output = self.decode(z)
        return output, mu, logvar
    
    def generate_binary_edge_vector_directed(self, batch):
        # Get true_edge_index per graph in batch
        all_edges = self.grid_edges
        # Convert true_edge_index to a set of directed edges (tuples)
        tensors = []
        for b in batch:
            translated_index = []
            for edge in zip(b.edge_index[0], b.edge_index[1]):
                translated_index.append((b.original_ids[edge[0]].item(), b.original_ids[edge[1]].item()))
            true_edges = set(tuple(pair) for pair in translated_index)
            
            # Generate binary vector (no sorting of edges)
            binary_vector = [1 if edge in true_edges else 0 for edge in all_edges]  
            tensors.append(torch.tensor(binary_vector, dtype=torch.float))
        output = torch.stack(tensors).to("cuda")
        return output
    
    def recon_loss(self, output, batch):
        true_vector = self.generate_binary_edge_vector_directed(batch)
        if self.loss_function == "bce":
            loss = F.binary_cross_entropy(output, true_vector, reduction='sum')
        else:
            # Convert logits to probabilities
            
            # Compute the binary cross-entropy loss
            bce_loss = F.binary_cross_entropy(output, true_vector, reduction='none')
            # Compute the modulating factor (1 - p_t)^gamma
            p_t = output * true_vector + (1 - output) * (1 - true_vector)
            modulating_factor = (1 - p_t) ** self.gamma
            # Compute the alpha weighting factor
            alpha_t = self.alpha * true_vector + (1 - self.alpha) * (1 - true_vector)
            # Combine factors to compute focal loss
            loss = alpha_t * modulating_factor * bce_loss
            loss = loss.sum()
        return loss
    

        

    
