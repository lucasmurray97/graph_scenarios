import torch
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool import global_mean_pool
import torch.nn.functional as F

class GRAPH_VAE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, params):
        super(GRAPH_VAE, self).__init__()
        self.training = True
        self.distribution_std = params['distribution_std']
        self.variational_beta = params['variational_beta']

        # Encoder:
        self.conv1_mu = GCNConv(input_dim, 16)
        self.conv2_mu = GCNConv(16 , 32)
        self.conv3_mu = GCNConv(32 , latent_dim)
        self.conv1_log = GCNConv(input_dim, 16)
        self.conv2_log = GCNConv(16 , 32)
        self.conv3_log = GCNConv(32 , latent_dim)
        self.pool = global_mean_pool

        # Decoder:
        self.fc1 = torch.nn.Linear(latent_dim, 160000)
        


    def encode(self, x, edge_index, batch=torch.Tensor([0])):
        #print(x.shape, edge_index.shape, batch.shape)
        mu = self.conv1_mu(x, edge_index).relu()
        #print(x.shape)
        mu = self.conv2_mu(mu, edge_index).relu()
        #print(x.shape)
        mu = self.conv3_mu(mu, edge_index).relu()
        #print(x.shape)
        mu = self.pool(mu, batch)
        #print(x.shape)
        log = self.conv1_log(x, edge_index).relu()
        #print(x.shape)
        log = self.conv2_log(log, edge_index).relu()
        #print(x.shape)
        log = self.conv3_log(log, edge_index).relu()
        #print(x.shape)
        log = self.pool(log, batch)
        #print(x.shape)
        return mu, log
    
    def decode(self, z):
        z = self.fc1(z).sigmoid()
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
        
    def loss(self, output, x, mu, logvar):
        recon_loss = F.binary_cross_entropy(output.view(-1, 160000), x.view(-1, 160000), reduction='sum')
        kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss, kldivergence, recon_loss + self.variational_beta * kldivergence
    
    def forward(self, x, edge_index, batch):
        mu, logvar = self.encode(x, edge_index, batch)
        z = self.latent_sample(mu, logvar)
        output = self.decode(z).reshape(-1, 400, 400)
        return output, mu, logvar
    
