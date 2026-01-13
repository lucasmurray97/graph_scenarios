import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, VGAE, global_mean_pool

# -------- Encoder used by VGAE --------
class ENCODER(nn.Module):
    """
    Parametrized VGAE encoder (SAGEConv) with LayerNorm:
      - depth: enc_layers (>=1)
      - returns per-node mu and logvar
    """
    def __init__(self, input_dim: int, latent_dim: int, capacity, enc_layers):
        super().__init__()
        C = int(capacity)
        self.enc_layers = int(enc_layers)
        hidden = C

        self.mu_convs, self.mu_norms = self._build_sage_tower(
            in_dim=input_dim, out_dim=latent_dim, hidden=hidden, L=self.enc_layers
        )
        self.lv_convs, self.lv_norms = self._build_sage_tower(
            in_dim=input_dim, out_dim=latent_dim, hidden=hidden, L=self.enc_layers
        )

        self.act = nn.ReLU(inplace=True)
        self.dp  = nn.Identity()

    def _build_sage_tower(self, in_dim: int, out_dim: int, hidden: int, L: int):
        """Builds (convs, norms) with LayerNorm on hidden layers only."""
        assert L >= 1, "enc_layers must be >= 1"
        convs = nn.ModuleList()
        norms = nn.ModuleList()

        if L == 1:
            convs.append(SAGEConv(in_dim, out_dim, aggr='mean'))
            norms.append(nn.Identity())  # no norm on final latent layer
            return convs, norms

        # first: in -> hidden
        convs.append(SAGEConv(in_dim, hidden, aggr='mean'))
        norms.append(nn.LayerNorm(hidden))

        # middle: hidden -> hidden
        for _ in range(max(0, L - 2)):
            convs.append(SAGEConv(hidden, hidden, aggr='mean'))
            norms.append(nn.LayerNorm(hidden))

        # last: hidden -> latent
        convs.append(SAGEConv(hidden, out_dim, aggr='mean'))
        norms.append(nn.Identity())  # final layer (latent), no norm

        return convs, norms

    def _run_tower(self, x, edge_index, convs, norms):
        for i, (conv, norm) in enumerate(zip(convs, norms)):
            x = conv(x, edge_index)      # [num_nodes, dim]
            x = norm(x)                  # LayerNorm over feature dim
            if i < len(convs) - 1:       # no act/dropout on the latent layer
                x = self.act(x)
                x = self.dp(x)
        return x

    def forward(self, x, edge_index):
        mu  = self._run_tower(x, edge_index, self.mu_convs, self.mu_norms)
        log = self._run_tower(x, edge_index, self.lv_convs, self.lv_norms)
        return mu, log


class GAE(nn.Module):
    """
    VGAE encoder → node latents z_i.
    Decoder:
      • Undirected feasible edges U scored by scaled dot product of node latents.
      • Ignition logits over N canonical slots via a per-node MLP head.

    Batch must provide:
      - edge_index_enc   : encoder edges over present nodes
      - y_node_present   : [B*N] or [B,N] {0,1} present-mask per canonical slot
      - y_edge_u         : [B*E_u] or [B, E_u] undirected edge labels
      - y_ign_idx        : [B] ignition slot id
    """
    def __init__(self, input_dim, latent_dim, params, template):
        super().__init__()
        self.name = "VGAE"
        self.distribution_std = float(params.get("distribution_std", 0.1))
        self.variational_beta = float(params.get("variational_beta", 1.0))

        # VGAE encoder (provides z + KL via internals)
        C = int(params.get("capacity", 8))
        enc_layers = int(params.get("enc_layers", 8))
        self.gae = VGAE(ENCODER(input_dim, latent_dim, capacity=C, enc_layers=enc_layers))

        # Template / feasible undirected edges
        self.template = template
        self.N  = int(template.Nmax)
        self.Eu = int(template.E_u)

        # Feasible undirected pairs (canonical)
        self.register_buffer("u_src", torch.as_tensor(template.undir_src, dtype=torch.long))
        self.register_buffer("u_dst", torch.as_tensor(template.undir_dst, dtype=torch.long))

        # Scaled dot-product for edges
        self.edge_scale = nn.Parameter(torch.tensor(5.0))
        self.edge_bias  = nn.Parameter(torch.tensor(0.0))

        # Ignition per-node MLP head: z_i -> scalar logit
        ign_hidden = int(params.get("ign_hidden", C))
        ign_layers = int(params.get("ign_layers", 2))
        blocks = []
        in_dim = latent_dim
        for l in range(max(0, ign_layers - 1)):
            blocks += [nn.Linear(in_dim, ign_hidden), nn.ReLU(inplace=True)]
            in_dim = ign_hidden
        blocks += [nn.Linear(in_dim, 1)]  # scalar logit per node
        self.ign_head = nn.Sequential(*blocks)

        # Loss weights
        self.edge_pos_weight  = float(params.get("edge_pos_weight", 0.0)) or None
        self.edge_loss_lambda = float(params.get("edge_loss_lambda", 1.0))
        self.ign_loss_lambda  = float(params.get("ign_loss_lambda", 1.0))

    # -------- helpers --------
    def _scatter_to_full(self, z_local, present_slots, N):
        """Place per-graph node latents into full [N,d] tensor (zeros elsewhere)."""
        d = z_local.size(-1)
        Z_full = z_local.new_zeros(N, d)
        Z_full[present_slots] = z_local
        return Z_full

    def _edge_logits_from_full(self, Z_full):
        """Score feasible undirected pairs from full [N,d] node latents."""
        zs = Z_full[self.u_src]                # [E_u, d]
        zt = Z_full[self.u_dst]                # [E_u, d]
        dot = (zs * zt).sum(dim=-1)            # [E_u]
        return self.edge_scale * dot + self.edge_bias
    
    def pool(self, x, batch):
        return global_mean_pool(x, batch)

    # -------- API --------
    def forward(self, x, edge_index_enc, batch):
        """
        Returns:
          edge_u_logits: [B, E_u]
          ign_logits   : [B, N]  (absent slots set later in loss; here we return raw)
          mu, logstd   : VGAE posterior params (per-node)
        """
        device = x.device
        z = self.gae.encode(x, edge_index_enc)            # [sum(n_b), d]
        mu, logstd = self.gae.__mu__, self.gae.__logstd__

        B = int(batch.batch.max().item()) + 1
        N = self.N
        present_all = getattr(batch, "y_node_present", None)
        assert present_all is not None, "batch.y_node_present is required"
        present_all = present_all.view(B, N).to(device)

        edge_logits_list = []
        ign_logits_list  = []

        for b in range(B):
            mask_local   = (batch.batch == b)
            z_local      = z[mask_local]                     # [n_b, d]
            present_b    = (present_all[b] == 1)             # [N]
            present_slots= present_b.nonzero(as_tuple=False).view(-1)  # [n_b]

            # Safety check (your Data keeps x in sorted slot order)
            assert z_local.size(0) == present_slots.numel(), \
                f"Mismatch: z_local {z_local.size(0)} vs present_slots {present_slots.numel()}"

            Z_full = self._scatter_to_full(z_local, present_slots, N)   # [N, d]

            # Edges
            logits_u = self._edge_logits_from_full(Z_full)              # [E_u]
            edge_logits_list.append(logits_u.unsqueeze(0))

            # Ignition: per-node scalar from z_i
            ign_b = self.ign_head(Z_full).squeeze(-1)                   # [N]
            # (we won't mask here; do it in the loss to keep gradients clean)
            ign_logits_list.append(ign_b.unsqueeze(0))

        edge_u_logits = torch.cat(edge_logits_list, dim=0)  # [B, E_u]
        ign_logits    = torch.cat(ign_logits_list,  dim=0)  # [B, N]
        return (edge_u_logits, ign_logits), mu, logstd

    def loss(self, output, batch, mu, logstd, template, beta_kl=1e-3):
        """
        output: (edge_u_logits[B,E_u], ign_logits[B,N])
        batch : y_edge_u[B,E_u], y_node_present[B,N], y_ign_idx[B]
        """
        (edge_u_logits, ign_logits) = output
        device = edge_u_logits.device

        B, E_u = edge_u_logits.size()
        N      = self.N

        y_edge_u = batch.y_edge_u.view(B, E_u).float().to(device)
        present  = batch.y_node_present.view(B, N).to(device)        # {0,1}
        y_ign    = batch.y_ign_idx.view(B).long().to(device)

        # ---- Edge loss (mask pairs where both endpoints are present) ----
        s = self.u_src
        t = self.u_dst
        both_present = (present[:, s] * present[:, t]).bool()         # [B, E_u]

        if self.edge_pos_weight is not None and self.edge_pos_weight > 0:
            bce = F.binary_cross_entropy_with_logits(
                edge_u_logits[both_present], y_edge_u[both_present],
                pos_weight=torch.tensor(self.edge_pos_weight, device=device),
                reduction="mean"
            )
        else:
            bce = F.binary_cross_entropy_with_logits(
                edge_u_logits[both_present], y_edge_u[both_present],
                reduction="mean"
            )
        edge_loss = self.edge_loss_lambda * bce

        # ---- Ignition loss (Cross-Entropy over present nodes) ----
        # Mask absent slots to -inf so CE ignores them
        mask_absent = (present == 0)                                   # [B,N]
        ign_masked = ign_logits.masked_fill(mask_absent, float("-inf"))# [B,N]
        ign_loss = F.cross_entropy(ign_masked, y_ign, reduction="mean")
        ign_loss = self.ign_loss_lambda * ign_loss

        # ---- KL from VGAE internals ----
        kl = self.gae.kl_loss()  # uses stored __mu__/__logstd__

        recon = edge_loss + ign_loss
        total = recon + beta_kl * kl
        return recon, kl, total, {"edge": edge_loss, "ign": ign_loss}

    # Convenience: sample node-latents from posterior
    def latent_sample(self, mu, logstd):
        std = logstd.exp()
        eps = torch.randn_like(std) * self.distribution_std
        return mu + eps * std
