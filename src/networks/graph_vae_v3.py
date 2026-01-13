import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool, Sequential


class GRAPH_VAE_V3(nn.Module):
    """
    Minimal GraphVAE for grids:
      - Encoder: GCN → (mu, logvar)
      - Decoder: per-node features H_i from [z || (x,y) || slot_emb]
                 * Undirected edge logits for canonical neighbor pairs only
                 * Ignition logits over nodes (softmax → single source)

    Outputs of decode:
      edges_u_logit: [B, E_u]   (undirected neighbor pairs)
      ign_logits:    [B, N]     (softmax over dim=1)
    """

    def __init__(self, input_dim, latent_dim, params, template):
        super().__init__()
        # ---- hyperparams
        C  = int(params.get("capacity", 128))
        dp = float(params.get("dropout", 0.0))
        self.distribution_std = float(params.get("distribution_std", 0.1))
        self.variational_beta = float(params.get("variational_beta", 1.0))
        self.name = "GRAPH_VAE_V3"

        # Optional class-imbalance weight for positive edges
        pw = float(params.get("edge_pos_weight", 0.0))
        self.register_buffer("edge_pos_weight",
                             torch.tensor(pw, dtype=torch.float) if pw > 0 else torch.tensor(0.0))
        ew = float(params.get("edge_loss_lambda", 1.0))
        iw = float(params.get("ign_loss_lambda", 0.0))
        self.register_buffer("edge_loss_lambda", torch.tensor(ew, dtype=torch.float))
        self.register_buffer("ign_loss_lambda", torch.tensor(iw, dtype=torch.float))
        # ---- template/grid
        self.template = template
        self.N = int(template.Nmax)

        # ---- slot embedding (to avoid identical node features)
        self.d_slot = int(params.get("d_slot", 256))
        self.slot_emb = nn.Embedding(self.N, self.d_slot)

        # ---------------- Encoder ----------------
        enc_layers  = int(params.get("enc_layers", 2))     # node trunk depth
        self.mu_convs, self.mu_norms = self.build_sage_tower(input_dim, latent_dim, C, enc_layers)
        self.lv_convs, self.lv_norms = self.build_sage_tower(input_dim, latent_dim, C, enc_layers)
        self.p_drop = 0.2

        # ---------------- Decoder ----------------

        # --- decoder (DEEPER & configurable) ---
        dec_in = latent_dim + 2 + self.d_slot
        dec_h  = 2 * C
        dec_layers  = int(params.get("dec_layers", 2))     # node trunk depth
        ign_layers = int(params.get("ign_layers", 2))    # edge MLP depth
        edge_layers = int(params.get("edge_layers", 2))    # edge MLP depth
        residual    = bool(params.get("dec_residual", True))
        norm        = params.get("dec_norm", "layer")      # "layer" or None

        # shared node trunk → heads
        self.node_mlp = DeepMLP(
            in_dim=dec_in, hidden=dec_h, out_dim=dec_h,
            num_layers=dec_layers, dropout=dp,
            activation=nn.ReLU(), residual=residual, norm=norm
        )

        self.edge_proj = DeepMLP(
            in_dim=dec_h, hidden=2 * dec_h, out_dim=dec_h,
            num_layers=edge_layers, dropout=dp,
            activation=nn.ReLU(), residual=residual, norm=norm
        )

        # Ignition head: scalar per node → softmax across N
        self.ignition_head = DeepMLP(
            in_dim=dec_h, hidden=2 * dec_h, out_dim=1,
            num_layers=ign_layers, dropout=dp,
            activation=nn.ReLU(), residual=residual, norm=norm
        )

        self.edge_scale = nn.Parameter(torch.tensor(5.0))
        self.edge_bias  = nn.Parameter(torch.tensor(0.0))


        # Cache normalized positions
        self.register_buffer("pos_all", template.pos_all.detach().clone())

        self.E_u        = template.E_u
        u_src      = template.undir_src      # [E_u]
        u_dst      = template.undir_dst      # [E_u]
        dir_e1     = template.undir_e1       # [E_u] (a→b) or -1
        dir_e2     = template.undir_e2       # [E_u] (b→a) or -1

        self.register_buffer("undir_src", torch.tensor(u_src, dtype=torch.long))
        self.register_buffer("undir_dst", torch.tensor(u_dst, dtype=torch.long))
        self.register_buffer("undir_e1",  torch.tensor(dir_e1, dtype=torch.long))  # a→b (or -1)
        self.register_buffer("undir_e2",  torch.tensor(dir_e2, dtype=torch.long))  # b→a (or -1)

    # --------- VAE utils ---------
    def build_sage_tower(self, in_dim, out_dim, hidden_dim, num_layers):
        """
        Returns (convs, norms) for a SAGE tower with `num_layers` layers.
        Layout:
        - if num_layers == 1: in_dim -> out_dim
        - else: in_dim -> hidden ... -> hidden -> out_dim
        Norms: LayerNorm after every hidden layer; Identity on the final (latent) layer.
        """
        convs = nn.ModuleList()
        norms = nn.ModuleList()

        if num_layers == 1:
            convs.append(SAGEConv(in_dim, out_dim, aggr='mean'))
            norms.append(nn.Identity())  # no norm on final latent layer
        else:
            # first: in -> hidden
            convs.append(SAGEConv(in_dim, hidden_dim, aggr='mean'))
            norms.append(nn.LayerNorm(hidden_dim))
            # middle: hidden -> hidden (num_layers - 2 times)
            for _ in range(num_layers - 2):
                convs.append(SAGEConv(hidden_dim, hidden_dim, aggr='mean'))
                norms.append(nn.LayerNorm(hidden_dim))
            # last: hidden -> latent
            convs.append(SAGEConv(hidden_dim, out_dim, aggr='mean'))
            norms.append(nn.Identity())  # final layer (latent), no norm
        return convs, norms
    
    def _encode_tower(self, x, edge_index, batch, convs, norms):
        L = len(convs)
        for i, (conv, norm) in enumerate(zip(convs, norms)):
            x = conv(x, edge_index)
            x = norm(x)
            # apply nonlinearity/dropout on all but the final (latent) layer
            if i < L - 1:
                x = F.relu(x, inplace=True)
                x = F.dropout(x, p=self.enc_p_drop, training=self.training)
        return global_mean_pool(x, batch)

    def _tower(self, x, edge_index, batch, convs, norms):
        for conv, norm in zip(convs, norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.p_drop, training=self.training)
        return global_mean_pool(x, batch.batch)  # graph-level embedding
    
    def broadcast_latent(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return z.unsqueeze(1).expand(-1, self.N, -1)  # [B, N, zdim]

    def encode(self, x, edge_index, batch):
        mu  = self._tower(x, edge_index, batch, self.mu_convs, self.mu_norms)
        log = self._tower(x, edge_index, batch, self.lv_convs, self.lv_norms)
        return mu, log

    def latent_sample(self, mu, logvar):
        if self.training:
            std = (0.5 * logvar).exp_()
            eps = torch.randn_like(std) * self.distribution_std
            return mu + eps * std
        return mu

    # --------- Decoder ---------
    def decode(self, z: torch.Tensor):
        """
        Returns:
          edges_u_logit: [B, E_u]  (undirected neighbor pairs)
          ign_logits:    [B, N]    (softmax over nodes → single ignition)
        """
        Z = self.broadcast_latent(z)                        # [B, N, z]
        B, N, _ = Z.shape
        pos = self.pos_all.unsqueeze(0).expand(B, -1, -1)  # [B, N, 2]
        slot_ids = torch.arange(N, device=Z.device).unsqueeze(0).expand(B, -1)
        slot_e   = self.slot_emb(slot_ids)                 # [B, N, d_slot]

        Zp = torch.cat([Z, pos, slot_e], dim=-1)           # [B, N, dec_in]
        H = self.node_mlp(Zp.reshape(B * N, Zp.size(-1))).reshape(B, N, -1)  # [B, N, dec_h]

        # Ignition
        ign_logits = self.ignition_head(H).squeeze(-1)     # [B, N]

        # Undirected pairs
        Q = self.edge_proj(H)
        Q = F.normalize(Q, p=2, dim=-1)
        s_idx = self.undir_src  # [E_u]
        t_idx = self.undir_dst  # [E_u]
        Hs = Q[:, s_idx, :]      # [B, E_u, dec_h]
        Ht = Q[:, t_idx, :]      # [B, E_u, dec_h]
        cos = (Hs * Ht).sum(dim=-1)              # [-1, 1] ideally, but yours ~0.97..1.0
        edges_u_logit = self.edge_scale * cos + self.edge_bias
 

        return edges_u_logit, ign_logits

    # --------- Loss (edges + ignition + KL) ---------
    def loss(self, output, batch, mu, logvar, template, beta_kl=1e-3):
        # unpack your UNDIRECTED outputs (adjust if you return a dict)
        edge_u_logits, ign_logits = output        # [B, E_u], [B, N] (optional)

        device = edge_u_logits.device
        B      = batch.num_graphs
        E_u    = self.E_u      # UNDIRECTED edge count
        N      = template.Nmax

        y_edge_u = batch.y_edge_u.view(B, E_u).float().to(device)  # [B, E_u]

        # edge loss (pos_weight optional)
        edge_w = getattr(self, "edge_loss_lambda", 1.)
        pos_w = getattr(self, "edge_pos_weight", None)
        edge_loss = F.binary_cross_entropy_with_logits(
            edge_u_logits, y_edge_u,
            pos_weight=pos_w, reduction="mean"
        ) * edge_w

        # optional ignition loss (only if you provide a target)
        ign_loss = None
        ign_w = getattr(self, "ign_loss_lambda", 1.)
        if hasattr(batch, "y_ign_idx"):  # integer index per graph
            ign_loss = F.cross_entropy(
                ign_logits, 
                batch.y_ign_idx.view(B).to(device), 
                reduction="mean") * ign_w
            
        # print(torch.max(ign_logits, dim=1)[1], batch.y_ign_idx.view(B).to(device))

        # KL as before
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # recon = edges (+ ignition if present)
        recon = edge_loss + (ign_loss if isinstance(ign_loss, torch.Tensor) else 0.0)
        total = recon + beta_kl * kl

        # return a dict of components (you can keep your old signature if you prefer)
        return recon, kl, total, {"edge": edge_loss, "ign": ign_loss}

    # --------- Forward ---------
    def forward(self, x, edge_index, batch):
        mu, logvar = self.encode(x, edge_index, batch)
        z = self.latent_sample(mu, logvar)
        output = self.decode(z)
        return output, mu, logvar

    
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