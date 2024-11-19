import torch
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GINConv, InnerProductDecoder, VGAE, Sequential
from torch_geometric.nn.pool import global_mean_pool
import torch.nn.functional as F

class ENCODER(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, params):
        super(ENCODER, self).__init__()

        # Encoder:
        self.encoder_mu = Sequential('x, edge_index',[
                (GCNConv(input_dim, 64), 'x, edge_index -> x' ),
                torch.nn.ReLU(),
                (GCNConv(64, 128), 'x, edge_index -> x' ),
                torch.nn.ReLU(),
                (GCNConv(128, latent_dim), 'x, edge_index -> x' ),
                torch.nn.ReLU()])
        
        self.encoder_log = Sequential('x, edge_index',[
                (GCNConv(input_dim, 64), 'x, edge_index -> x' ),
                torch.nn.ReLU(),
                (GCNConv(64, 128), 'x, edge_index -> x' ),
                torch.nn.ReLU(),
                (GCNConv(128, latent_dim), 'x, edge_index -> x' ),
                torch.nn.ReLU()])

    def forward(self, x, edge_index):
        mu = self.encoder_mu(x, edge_index)
        log = self.encoder_log(x, edge_index)
        return mu, log
    

class GRAPH_VAE_V3(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, params):
        super(GRAPH_VAE_V3, self).__init__()
        self.training = True
        self.distribution_std = params['distribution_std']
        self.variational_beta = params['variational_beta']
        self.name = "GRAPH_VAE_V3"

        self.gae = VGAE(ENCODER(input_dim, latent_dim, params))

        self.pool = global_mean_pool


    
    def forward(self, x, edge_index, batch=torch.Tensor([0])):
        mu = self.gae.encode(x, edge_index)
        return self.gae.decode(mu, edge_index), mu, self.gae.__logstd__
    
    def loss(self, x, edge_index, batch=torch.Tensor([0])):
        mu = self.gae.encode(x, edge_index)
        recon_loss = self.gae.recon_loss(mu, edge_index)
        kl_loss = self.gae.kl_loss(mu)
        total_loss = recon_loss + self.variational_beta * kl_loss
        return  recon_loss, kl_loss, total_loss
    
    def pool(self, mu, batch):
        return self.pool(mu, batch)
    
    def train_(self):
        self.gae.training = True
    
    def eval_(self):
        self.gae.training = False
    
    
    


