import torch
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool import global_mean_pool
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data, Batch
from utils import GridTemplate

class GRAPH_VAE_V3(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, params, template):
        super().__init__()
        # --- hyperparams ---
        C  = int(params.get("capacity", 128))        # capacity knob
        dp = float(params.get("dropout", 0.0))       # optional dropout
        self.distribution_std = params['distribution_std']
        self.variational_beta = params['variational_beta']
        self.name = "GRAPH_VAE_V3" 
        pw = float(params.get("edge_pos_weight", 0.0))
        if pw > 0:
            self.register_buffer("edge_pos_weight", torch.tensor(pw))

        nw = float(params.get("edge_neg_weight", 1.0))
        self.register_buffer("edge_neg_weight", torch.tensor(nw))

        self.edge_loss_lambda = float(params.get("edge_loss_lambda", 1.0))

        # --- template / grid ---
        self.template = template
        self.N = int(template.Nmax)                  # adapt to grid size
        self.fuel_classes = int(params.get("fuel_classes", 5))

        # --- encoder (scaled by capacity) ---
        h1, h2 = C, 2 * C

        self.conv1_mu  = GCNConv(input_dim, h1)
        self.conv2_mu  = GCNConv(h1, h2)
        self.conv3_mu  = GCNConv(h2, latent_dim)

        self.conv1_log = GCNConv(input_dim, h1)
        self.conv2_log = GCNConv(h1, h2)
        self.conv3_log = GCNConv(h2, latent_dim)

        self.enc_dropout = nn.Dropout(dp) if dp > 0 else nn.Identity()
        self.pool = global_mean_pool

        # --- decoder (scaled by capacity) ---
        # encoder (unchanged except scaled width)
        h1, h2 = C, 2 * C
        self.conv1_mu  = GCNConv(input_dim, h1)
        self.conv2_mu  = GCNConv(h1, h2)
        self.conv3_mu  = GCNConv(h2, latent_dim)
        self.conv1_log = GCNConv(input_dim, h1)
        self.conv2_log = GCNConv(h1, h2)
        self.conv3_log = GCNConv(h2, latent_dim)
        self.enc_dropout = nn.Dropout(dp) if dp > 0 else nn.Identity()
        self.pool = global_mean_pool

        # --- decoder (DEEPER & configurable) ---
        dec_in = latent_dim + 2
        dec_h  = 2 * C
        dec_layers  = int(params.get("dec_layers", 2))     # node trunk depth
        edge_layers = int(params.get("edge_layers", 2))    # edge MLP depth
        residual    = bool(params.get("dec_residual", True))
        norm        = params.get("dec_norm", "layer")      # "layer" or None

        # shared node trunk → heads
        self.node_mlp = DeepMLP(
            in_dim=dec_in, hidden=dec_h, out_dim=dec_h,
            num_layers=dec_layers, dropout=dp,
            activation=nn.ReLU(), residual=residual, norm=norm
        )
        self.node_exist_mlp = nn.Linear(dec_h, 1)
        self.fuel_mlp       = nn.Linear(dec_h, self.fuel_classes)
        self.alt_mlp        = nn.Linear(dec_h, 1)
        self.slope_mlp      = nn.Linear(dec_h, 1)

        # edge head MLP (from latent+pos directly)
        self.edge_mlp = DeepMLP(
            in_dim=dec_in, hidden=dec_h, out_dim=8,
            num_layers=edge_layers, dropout=dp,
            activation=nn.ReLU(), residual=residual, norm=norm
        )

        self.register_buffer("pos_all", template.pos_all.detach().clone())  # decoupled CPU copy


        # Bidirectional penalty
        self.bidir_lambda = float(params.get("bidir_lambda", 0.2))  # strength of the penalty
        # Build list of (e, e_rev) pairs once
        src, dst = self.template.edge_index_cand.cpu().tolist()
        lookup = {(s, d): i for i, (s, d) in enumerate(zip(src, dst))}
        pairs = []
        for i, (s, d) in enumerate(zip(src, dst)):
            j = lookup.get((d, s))
            if j is not None and i < j:      # keep each undirected pair once
                pairs.append([i, j])
        pairs = torch.tensor(pairs, dtype=torch.long) if len(pairs) else torch.empty(0, 2, dtype=torch.long)
        self.register_buffer("bidir_pairs", pairs)

    def broadcast_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, latent_dim] or [latent_dim]
        returns: [B, N, latent_dim]
        """
        if z.dim() == 1:             # allow single-graph decode
            z = z.unsqueeze(0)       # [1, latent_dim]
        return z.unsqueeze(1).expand(-1, self.N, -1)  # [B, N, latent_dim]

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
    
    def decode(self, z: torch.Tensor):
        """
        z: [B, latent_dim] or [latent_dim]
        returns:
        nodes: [B, N, 1]          (existence logits)
        edges: [B, N, 8]          (one logit per 8 directions per node)
        fuel:  [B, N, C]          (class logits)
        alt:   [B, N, 1]
        slope: [B, N, 1]    # if you add slope_mlp like alt_mlp
        """
        Z = self.broadcast_latent(z)            # [B, N, latent_dim]
        B, N, D = Z.shape
        pos = self.pos_all.unsqueeze(0).expand(B, -1, -1)              # [B,N,2]
        Zp = torch.cat([Z, pos], dim=-1)   

        # Node heads
        h = self.node_mlp(Zp.reshape(B*N, D+2))
        nodes = self.node_exist_mlp(h).reshape(B, N, 1)
        fuel  = self.fuel_mlp(h).reshape(B, N, self.fuel_classes)
        alt   = self.alt_mlp(h).reshape(B, N, 1)
        # If you also predict slope:
        slope = self.slope_mlp(h).reshape(B, N, 1)

        # Edge head (per node → 8 directions)
        edges = self.edge_mlp(Zp.reshape(B*N, D+2)).reshape(B, N, 8)

        return nodes, edges, fuel, alt, slope
    
    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.normal(torch.zeros(std.shape), self.distribution_std).to(mu.device)
            sample = mu + (eps * std)
            return sample
        else:
            return mu

    def kl_divergence(self, mu, logvar):
        nll = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return nll

    def bce_nodes(self, logits, targets):
        loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="sum")
        return loss
    
    def bce_masked(self, logits, targets, mask):
        loss = F.binary_cross_entropy_with_logits(logits[mask], targets[mask].float(), reduction="sum")
        return loss
    
    def mse_masked(self, preds, targets, mask):
        return F.mse_loss(preds[mask], targets[mask], reduce="sum")
        

    def loss(self, output, batch, mu, logvar, template, beta_kl=1e-3):
        """
        output: nodes[B,N,1], edges[B,N,8], fuels[B,N,C], alts[B,N,1], slopes[B,N,1]
        batch:  has y_node_present[B*N], y_fuel[B*N], y_alt[B*N], y_slope[B*N], y_edge[B*E]
        """
        nodes, edges8, fuels, alts, slopes = output
        B = batch.num_graphs
        N = template.Nmax
        E = template.edge_index_cand.size(1)
        device = nodes.device

        # reshape targets
        y_node  = batch.y_node_present.view(B, N).float().to(device)   # [B,N]
        y_fuel  = batch.y_fuel.view(B, N).long().to(device)            # [B,N]
        y_alt   = batch.y_alt.view(B, N).to(device)                    # [B,N]
        y_slope = batch.y_slope.view(B, N).to(device)                  # [B,N]
        y_edgeE = batch.y_edge.view(B, E).float().to(device)           # [B,E]

        # ----- node existence BCE -----
        node_loss = F.binary_cross_entropy_with_logits(
            nodes.squeeze(-1), y_node, reduction="mean"
        )

        # ----- fuel CE on present nodes -----
        present_mask = (y_node == 1)                                   # [B,N]
        if present_mask.any():
            fuel_loss = F.cross_entropy(
                fuels[present_mask], y_fuel[present_mask], reduction="mean"
            )
            # alt/slope MSE (you can upgrade to Gaussian NLL later)
            alt_loss   = F.mse_loss(alts.squeeze(-1)[present_mask],   y_alt[present_mask],   reduction="mean")
            slope_loss = F.mse_loss(slopes.squeeze(-1)[present_mask], y_slope[present_mask], reduction="mean")
        else:
            fuel_loss = alt_loss = slope_loss = torch.tensor(0., device=device)

        # ----- edge BCE: map [B,E] targets to [B,N,8] positions -----
        node_dir_to_e   = template.node_dir_to_e.to(device)                       # [N,8], -1 if invalid
        node_dir_to_dst = template.node_dir_to_dst.to(device)                     # [N,8], -1 if invalid
        valid_nd = (node_dir_to_e >= 0)                                # [N,8]

        # gather edge targets for valid (node,dir)
        e_idx = node_dir_to_e[valid_nd]                                # [M]
        y_edge_valid = y_edgeE.gather(1, e_idx.unsqueeze(0).expand(B, -1))  # [B,M]

        # build gating mask: both endpoints exist
        dst_idx = template.node_dir_to_dst.to(device)                     # [N, 8]
        valid_nd = (dst_idx >= 0)                              # [N, 8]
        dst_idx_safe = dst_idx.clamp_min(0)                    # replace -1 with 0

        # y_node: [B, N]  -> make it [B, N, 8] so it matches index dims
        y_node_exp = y_node.unsqueeze(-1).expand(-1, -1, 8)    # [B, N, 8]

        # gather along dim=1 (the N dimension), using per-(node,dir) destination indices
        # index must be [B, N, 8]
        idx = dst_idx_safe.unsqueeze(0).expand(B, -1, -1)      # [B, N, 8]

        # result: for each batch b, node n, direction d -> y_node[b, dst_idx[n,d]]
        y_dst = torch.gather(y_node_exp, 1, idx)               # [B, N, 8]

        # later, when masking edges, also require valid direction + both endpoints exist:
        both_exist = (y_node.unsqueeze(-1) == 1) & (y_dst == 1) & valid_nd.unsqueeze(0) 

        # fetch logits at those valid positions
        edges8_flat = edges8.view(B, -1)                                 # [B,N*8]
        nd_pos = torch.nonzero(valid_nd, as_tuple=False)                  # [M,2] -> (n, d)
        idx_flat = (nd_pos[:,0] * 8 + nd_pos[:,1]).to(device)            # [M]
        logits_valid = edges8_flat.index_select(1, idx_flat)             # [B,M]

        # mask by both_exist (boolean per-B)
        both_exist_flat = both_exist.view(B, -1).index_select(1, idx_flat)  # [B,M]
        # labels/logits for edges we supervise this step
        labels = y_edge_valid[both_exist_flat].float()      # [K]
        logits = logits_valid[both_exist_flat]              # [K]
        if labels.numel() > 0:
            # class weights
            pos_w = getattr(self, "edge_pos_weight", None)   # tensor or None
            neg_w = getattr(self, "edge_neg_weight", torch.tensor(1.0, device=device))

            # per-sample 'weight' to upweight NEGATIVE labels (penalize FPs)
            w = torch.where(labels > 0.5, torch.ones_like(labels), neg_w.expand_as(labels))
            edge_loss_raw = F.binary_cross_entropy_with_logits(
                logits, labels,
                weight=w,                   # applies to both classes
                pos_weight=pos_w,           # extra factor for positives (if provided)
                reduction="mean"
            )
            edge_loss = self.edge_loss_lambda * edge_loss_raw
        else:
            edge_loss = torch.tensor(0., device=device)

        ## Bidirectional penalty

        E = self.template.edge_index_cand.size(1)
        node_dir_to_e = self.template.node_dir_to_e.to(device)            # [N,8] with -1 for invalid
        valid_nd = (node_dir_to_e >= 0)
        e_idx = node_dir_to_e[valid_nd]                   # [M] edge ids for valid (node,dir)

        # logits for valid (node,dir) → scatter to [B,E]
        edges8_flat = edges8.view(B, -1)                  # [B,N*8]
        nd_pos = valid_nd.nonzero(as_tuple=False)         # [M,2] (n,d)
        idx_flat = (nd_pos[:, 0] * 8 + nd_pos[:, 1]).to(device)   # [M]
        logits_valid = edges8_flat.index_select(1, idx_flat)       # [B,M]

        logits_E = torch.full((B, E), -float("inf"), device=device)  # -inf → sigmoid ~ 0
        logits_E.scatter_(1, e_idx.unsqueeze(0).expand(B, -1), logits_valid)
        p_E = torch.sigmoid(logits_E)                      # [B,E]

        # gate pairs by GT node existence (stable)
        src_all, dst_all = self.template.edge_index_cand
        both_exist_E = (y_node[:, src_all] == 1) & (y_node[:, dst_all] == 1)  # [B,E]

        bidir_loss = torch.tensor(0., device=device)
        if self.bidir_pairs.numel() > 0:
            e1 = self.bidir_pairs[:, 0]     # [P]
            e2 = self.bidir_pairs[:, 1]     # [P]

            p1 = p_E[:, e1]                  # [B,P]
            p2 = p_E[:, e2]                  # [B,P]
            mask = both_exist_E[:, e1] & both_exist_E[:, e2]   # [B,P]

            # Hinge-style mutual exclusion: penalize when both are large
            pen = F.relu(p1 + p2 - 1.0)      # smooth, bounded gradient
            if mask.any():
                bidir_loss = pen[mask].mean()

            # add to edge loss
            edge_loss = edge_loss + self.bidir_lambda * bidir_loss

        # ----- KL -----
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        recon = node_loss + fuel_loss + alt_loss + slope_loss + edge_loss
        total = recon + beta_kl * kl
        return recon, kl, total, node_loss, edge_loss

    
    def forward(self, x, edge_index, batch):
        mu, logvar = self.encode(x, edge_index, batch)
        z = self.latent_sample(mu, logvar)
        output = self.decode(z)
        return output, mu, logvar


    # ----------------------------
    # Sampling
    # ----------------------------

    @torch.no_grad()
    def sample_from_latent(
        self,
        z,                        # [z_dim]
        template: GridTemplate,
        tau_node: float = 0.5,
        tau_edge: float = 0.5,
    ):
        self.eval()
        # Fake a batch of size 1 for reuse of batched decoder
        Z = z.unsqueeze(0)  # [1,z]
        node_out = self.dec.decode_nodes_batched(Z, template.pos_all)
        logits_node = node_out["logits_node"][0]          # [Nmax]
        p_node = torch.sigmoid(logits_node)

        edge_logits = self.dec.decode_edges_batched(Z, template.pos_all,
                                                    template.edge_index_cand, template.dir_id)[0]  # [E_all]

        # Gate edges by node probabilities (add logits == multiply probs)
        logit_gate = torch.logit(p_node.clamp(1e-6, 1-1e-6))
        src, dst = template.edge_index_cand
        logits_edge_gated = edge_logits + logit_gate[src] + logit_gate[dst]
        p_edge = torch.sigmoid(logits_edge_gated)

        nodes_keep = p_node > tau_node
        edges_keep = p_edge > tau_edge

        return {
            "p_node": p_node, "nodes_keep": nodes_keep,
            "fuel_logits": node_out["fuel_logits"][0],
            "alt_mu": node_out["alt_mu"][0], "slope_mu": node_out["slope_mu"][0],
            "p_edge": p_edge, "edges_keep": edges_keep,
            "edge_index_cand": template.edge_index_cand
        }
    
class DeepMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        num_layers: int = 4,          # depth (>=2: input→hidden, (L-2) hidden blocks, hidden→out)
        dropout: float = 0.0,
        activation: nn.Module = nn.ReLU(),
        residual: bool = True,        # residual on hidden→hidden blocks
        norm: str | None = "layer",   # "layer" or None
    ):
        super().__init__()
        assert num_layers >= 2, "num_layers must be >= 2"

        self.act = activation
        self.dp = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.residual = residual
        self.use_norm = (norm == "layer")

        # first layer: in → hidden
        self.in_lin = nn.Linear(in_dim, hidden)
        self.in_norm = nn.LayerNorm(hidden) if self.use_norm else nn.Identity()

        # middle hidden blocks (hidden → hidden), residual-capable
        self.blocks = nn.ModuleList()
        self.blocks_norm = nn.ModuleList()
        for _ in range(max(0, num_layers - 2)):
            self.blocks.append(nn.Linear(hidden, hidden))
            self.blocks_norm.append(nn.LayerNorm(hidden) if self.use_norm else nn.Identity())

        # final layer: hidden → out
        self.out_lin = nn.Linear(hidden, out_dim)

    def forward(self, x):
        # in → hidden
        x = self.in_lin(x)
        x = self.in_norm(x)
        x = self.act(x)
        x = self.dp(x)

        # hidden blocks with residual
        for lin, ln in zip(self.blocks, self.blocks_norm):
            y = lin(x)
            y = ln(y)
            y = self.act(y)
            y = self.dp(y)
            if self.residual:
                x = x + y
            else:
                x = y

        # hidden → out
        return self.out_lin(x)