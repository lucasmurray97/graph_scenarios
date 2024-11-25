import torch
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GINConv, InnerProductDecoder, VGAE, Sequential
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.utils import negative_sampling
import torch.nn.functional as F
from torch.nn import BatchNorm1d
import sys
sys.path.append("..")
from utils import generate_grid_edges
import random

class ENCODER(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, params):
        super(ENCODER, self).__init__()
        self.capacity = params["capacity"]

        # Encoder:
        self.encoder_mu = Sequential('x, edge_index',[
                (GCNConv(input_dim, input_dim * self.capacity), 'x, edge_index -> x' ),
                BatchNorm1d(input_dim * self.capacity),
                torch.nn.ReLU(),
                (GCNConv(input_dim * self.capacity, input_dim * self.capacity * 2), 'x, edge_index -> x' ),
                BatchNorm1d(input_dim * self.capacity * 2),
                torch.nn.ReLU(),
                (GCNConv(input_dim * self.capacity * 2, latent_dim), 'x, edge_index -> x' ),
                ])
        
        self.encoder_log = Sequential('x, edge_index',[
                (GCNConv(input_dim, input_dim * self.capacity), 'x, edge_index -> x' ),
                torch.nn.ReLU(),
                BatchNorm1d(input_dim * self.capacity),
                (GCNConv(input_dim * self.capacity, input_dim * self.capacity * 2), 'x, edge_index -> x' ),
                torch.nn.ReLU(),
                BatchNorm1d(input_dim * self.capacity * 2),
                (GCNConv(input_dim * self.capacity * 2, latent_dim), 'x, edge_index -> x' ),
                ])

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
        self.capacity = params['capacity']
        self.name = "GRAPH_VAE_V3"

        self.gae = VGAE(ENCODER(input_dim, latent_dim, params))
        self.pool_layer = global_mean_pool
        self.grid_edges = generate_grid_edges(20)

    
    def forward(self, x, edge_index, batch=torch.Tensor([0])):
        mu = self.gae.encode(x, edge_index)
        return self.gae.decode(mu, edge_index), mu, self.gae.__logstd__
    
    def loss(self, x, edge_index, batch=torch.Tensor([0])):
        mu = self.gae.encode(x, edge_index)
        recon_loss = self.recon_loss(mu, edge_index, batch)
        kl_loss = self.gae.kl_loss(mu)
        total_loss = recon_loss + self.variational_beta * kl_loss
        return  recon_loss, kl_loss, total_loss
    
    def pool(self, mu, log, batch):
        return self.pool_layer(mu, batch), self.pool_layer(log, batch)
    
    def recon_loss(self, x, edge_index, batch=torch.Tensor([0])):
        missing = []
        for b in batch:
            current_edges_set = set(map(tuple, b.original_ids[b.edge_index].t().tolist()))
            all_edges_set = set(map(tuple, self.grid_edges))
            
            # Find edges in all_possible_edges that are not in current edges
            missing_edges_set = all_edges_set - current_edges_set

            missing_edges = random.choices(list(missing_edges_set), k=b.edge_index.shape[1])
            # Convert back to edge_index format
            missing_edges = torch.tensor(list(missing_edges), dtype=torch.long).t()
            missing.append(missing_edges)
        missing_edges = torch.cat(missing, dim=1).cuda()
        try:
            value = (x[missing_edges[0]] * x[missing_edges[1]]).sum(dim=1)
            loss = self.gae.recon_loss(x, pos_edge_index=edge_index, neg_edge_index=missing_edges)
        except:
            print(x.shape)
            print(missing_edges.shape)
            print(missing_edges)
            loss = self.gae.recon_loss(x, pos_edge_index=edge_index)
        return loss
    
    def train_(self):
        self.gae.training = True
    
    def eval_(self):
        self.gae.training = False

    def test(self, z, pos_edge_index,
             neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to evaluate
                against.
            neg_edge_index (torch.Tensor): The negative edges to evaluate
                against.
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.gae.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.gae.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), (pred >= 0.5).detach().cpu().numpy()
        return accuracy_score(y, pred), precision_score(y, pred), recall_score(y, pred), f1_score(y, pred)

    
    
    


