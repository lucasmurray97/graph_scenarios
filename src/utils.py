import torch
from torch_geometric.data import Dataset
from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils import to_dense_adj
from torch import Tensor
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Iterable
import glob
import networkx as nx
import pickle
from torch_geometric.data import Data, Batch
import codecs
import numpy as np
from torchvision.transforms import Normalize
from sklearn.metrics import roc_auc_score, average_precision_score, roc_auc_score, average_precision_score, precision_recall_fscore_support
import math
import json
import torch.nn.functional as F
import networkx as nx
from torchmetrics.classification import (
    BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score,
    MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score,
)

import os, glob
import random

class GraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        """
        This class is a Dataset for PyTorch Geometric that reads a list of pickle files
        and returns a PyG Graph object. The pickle files are NetworkX DiGraph objects and are
        sampled from the directory root. The class also stores the original ids of the nodes
        args: root: str, path to the directory containing the pickle files
        output: None
        """
        ids = {} # dictionary to store the ids of the files
        n = 0
        # iterate over the files in the directory
        for fname in glob.glob(f'{root}/*.pickle'):
            ids[n] = fname
            n += 1
        self.ids = ids

    def __len__(self):
        # return the number of files in the directory
        return len(self.ids)

    def __getitem__(self, idx):
        # load the pickle file and return a PyG Graph object corresponding to the idx file
        G = pickle.load(open(f'{self.ids[idx]}', 'rb'))
        for node in G.nodes():
            G.nodes[node]['original_ids'] = node - 1
        pyg_graph = from_networkx(G)
        id_ = self.ids[idx].split("/")[-1]
        pyg_graph.original_graph = torch.tensor(float(id_.split("_")[1].split(".pickle")[0]))
        # List nodes in pyg_graph:
        pyg_graph.x = torch.tensor([1 for _ in range(len(G))], dtype=torch.float).unsqueeze(1)
        return pyg_graph
    
    
def reconstruct_matrix(graph_list):
    global_matrix = []
    for graph in graph_list:
        node_mapping = graph.original_ids
        mapped_edge_index = torch.tensor(
        [[node_mapping[int(i)] for i in edge_pair] for edge_pair in graph.edge_index.T],
            dtype=torch.long).T
        matrix = to_dense_adj(mapped_edge_index, max_num_nodes=400)[0]
        global_matrix.append(matrix)
    # We concatenate the list of matrices into a tensor:
    return torch.cat(global_matrix)
    
def generate_grid_edges(grid_size):
    edges = []
    for i in range(grid_size):
        for j in range(grid_size):
            node = i * grid_size + j
            neighbors = [
                (i - 1, j),  # up
                (i + 1, j),  # down
                (i, j - 1),  # left
                (i, j + 1),  # right
                (i - 1, j - 1),  # top-left diagonal
                (i - 1, j + 1),  # top-right diagonal
                (i + 1, j - 1),  # bottom-left diagonal
                (i + 1, j + 1),  # bottom-right diagonal
            ]
            for ni, nj in neighbors:
                if 0 <= ni < grid_size and 0 <= nj < grid_size:
                    neighbor_node = ni * grid_size + nj
                    edges.append((node, neighbor_node))
    return edges
    
class GraphDatasetV2(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        """
        This class is a Dataset for PyTorch Geometric that reads a list of pickle files
        and returns a PyG Graph object. The pickle files are NetworkX DiGraph objects and are
        sampled from the directory root. The class also stores the original ids of the nodes
        args: root: str, path to the directory containing the pickle files
        output: None
        """
        ids = {} # dictionary to store the ids of the files
        n = 0
        # iterate over the files in the directory
        for fname in glob.glob(f'{root}/*.pickle'):
            ids[n] = fname
            n += 1
        self.ids = ids

    def __len__(self):
        # return the number of files in the directory
        return len(self.ids)

    def __getitem__(self, idx):
        # load the pickle file and return a PyG Graph object corresponding to the idx file
        G = pickle.load(open(f'{self.ids[idx]}', 'rb'))
        for node in G.nodes():
            G.nodes[node]['original_ids'] = node - 1
        pyg_graph = from_networkx(G)
        id_ = self.ids[idx].split("/")[-1]
        pyg_graph.original_graph = torch.tensor(float(id_.split("_")[1].split(".pickle")[0]))
        # Add parameters of fuel, asc, slope, ...
        pyg_graph.x = torch.tensor([1 for _ in range(len(G))], dtype=torch.float).unsqueeze(1)
        return pyg_graph
    
    
def reconstruct_matrix(graph_list):
    global_matrix = []
    for graph in graph_list:
        node_mapping = graph.original_ids
        mapped_edge_index = torch.tensor(
        [[node_mapping[int(i)] for i in edge_pair] for edge_pair in graph.edge_index.T],
            dtype=torch.long).T
        matrix = to_dense_adj(mapped_edge_index, max_num_nodes=400)[0]
        global_matrix.append(matrix)
    # We concatenate the list of matrices into a tensor:
    return torch.cat(global_matrix)
    
def generate_grid_edges(grid_size):
    edges = []
    for i in range(grid_size):
        for j in range(grid_size):
            node = i * grid_size + j
            neighbors = [
                (i - 1, j),  # up
                (i + 1, j),  # down
                (i, j - 1),  # left
                (i, j + 1),  # right
                (i - 1, j - 1),  # top-left diagonal
                (i - 1, j + 1),  # top-right diagonal
                (i + 1, j - 1),  # bottom-left diagonal
                (i + 1, j + 1),  # bottom-right diagonal
            ]
            for ni, nj in neighbors:
                if 0 <= ni < grid_size and 0 <= nj < grid_size:
                    neighbor_node = ni * grid_size + nj
                    edges.append((node, neighbor_node))
    return edges
    

# ----------------------------
# Grid template & utilities
# ----------------------------

DIR8: List[Tuple[int, int]] = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]
DIR_TO_ID: Dict[Tuple[int,int], int] = {d:i for i, d in enumerate(DIR8)}

@dataclass
class GridTemplate:
    H: int
    W: int
    Nmax: int
    pos_all: Tensor              # [Nmax, 2] normalized positions (-1..1)
    edge_index_cand: Tensor      # [2, E_all] directed 8-neighbor candidate edges
    dir_id: Tensor               # [E_all] int64 in [0..7]
    node_mask_all: Tensor        # [Nmax] bool (True in-bounds)
    edge_mask_all: Tensor        # [E_all] bool (True for all candidates)

def build_grid_template(H: int, W: int, device: Optional[torch.device]=None) -> GridTemplate:
    device = torch.device('cpu')
    Nmax = H * W
    rows = torch.arange(H, device=device, dtype=torch.float32)
    cols = torch.arange(W, device=device, dtype=torch.float32)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    rr_n = (rr / (H-1) - 0.5) * 2.0 if H > 1 else torch.zeros_like(rr)
    cc_n = (cc / (W-1) - 0.5) * 2.0 if W > 1 else torch.zeros_like(cc)
    pos_all = torch.stack([rr_n.reshape(-1), cc_n.reshape(-1)], dim=1)  # [Nmax, 2]

    src_list, dst_list, dir_list = [], [], []
    for r in range(H):
        for c in range(W):
            i = r * W + c
            for (dr, dc) in DIR8:
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < H and 0 <= c2 < W:
                    j = r2 * W + c2
                    src_list.append(i); dst_list.append(j); dir_list.append(DIR_TO_ID[(dr, dc)])

    edge_index_cand = torch.tensor([src_list, dst_list], device=device, dtype=torch.long)
    dir_id = torch.tensor(dir_list, device=device, dtype=torch.long)
    node_mask_all = torch.ones(Nmax, device=device, dtype=torch.bool)
    edge_mask_all = torch.ones(edge_index_cand.size(1), device=device, dtype=torch.bool)
    template = GridTemplate(H, W, Nmax, pos_all, edge_index_cand, dir_id, node_mask_all, edge_mask_all)
    src, dst = template.edge_index_cand
    node_dir_to_e = torch.full((Nmax, 8), -1, dtype=torch.long, device=device)   # -> edge idx or -1
    node_dir_to_dst = torch.full((Nmax, 8), -1, dtype=torch.long, device=device) # -> dst slot or -1
    for e in range(src.numel()):
        s = int(src[e]); d = int(dir_id[e]); t = int(dst[e])
        node_dir_to_e[s, d] = e
        node_dir_to_dst[s, d] = t
    template.node_dir_to_e, template.node_dir_to_dst = node_dir_to_e, node_dir_to_dst
    return template


# ----------------------------
# Nx -> PyG Data (Path A)
# ----------------------------

def build_graph_data_from_nx(
    G: nx.DiGraph,
    template: GridTemplate,
    fuel_classes: int,
    use_true_edges_for_encoder: bool = True,
    alt_key: str = "altitude",
    slope_key: str = "slope",
    fuel_key: str = "fuel",
    slot_key: str = "slot",
) -> Data:
    """
    Convert a networkx.DiGraph into a PyG Data with full-grid targets (Path A).
    Each node must provide a canonical slot via:
      * node attribute 'slot' (int in [0..Nmax-1]), or
      * node attributes 'row','col', or
      * the node itself is a slot int or (row,col) tuple.

    Node attributes used:
      * fuel class in [0..fuel_classes-1] (key 'fuel' by default)
      * altitude (float) keyed by alt_key ('altitude' or 'alt')
      * slope (float) keyed by slope_key ('slope')
    """
    def to_slot(node):
        a = G.nodes[node]
        if slot_key in a:
            s = int(a[slot_key])
        else:
            raise ValueError(f"Cannot infer slot for edge endpoint {node}.")
        return s
    device = torch.device("cpu")   # <--- force CPU here
    H, W, Nmax = template.H, template.W, template.Nmax

    # --- resolve slot ids for nodes and collect per-node attributes ---
    active_idx: List[int] = []
    fuel_vals: List[int] = []
    alt_vals: List[float] = []
    slope_vals: List[float] = []

    # First pass: determine slot ids and which nodes are in the graph
    for u, attrs in G.nodes(data=True):
        slot = int(attrs[slot_key])
        if not (0 <= slot < Nmax):
            raise ValueError(f"Slot {slot} out of bounds for H={H}, W={W}.")
        active_idx.append(slot)
    # To keep deterministic order, sort by slot
    order = sorted(range(len(active_idx)), key=lambda i: active_idx[i])
    active_idx = [active_idx[i] for i in order]

    # Now collect attributes in the same order
    nodes_in_order = [list(G.nodes())[i] for i in order]
    for u in nodes_in_order:
        attrs = G.nodes[u]
        # fuel
        if fuel_key not in attrs:
            raise ValueError(f"Node {u} missing '{fuel_key}' attribute.")
        fuel = int(attrs[fuel_key])
        if not (0 <= fuel < fuel_classes):
            raise ValueError(f"Fuel class {fuel} out of bounds [0,{fuel_classes-1}] for node {u}.")
        fuel_vals.append(int(fuel))
        # altitude
        alt_attr = attrs.get(alt_key, attrs.get("alt"))
        if alt_attr is None:
            raise ValueError(f"Node {u} missing altitude attribute ('{alt_key}' or 'alt').")
        alt_vals.append(float(alt_attr))
        # slope
        slope_attr = attrs.get(slope_key)
        if slope_attr is None:
            raise ValueError(f"Node {u} missing slope attribute ('{slope_key}').")
        slope_vals.append(float(slope_attr))

    Na = len(active_idx)
    fuel_t = torch.tensor(fuel_vals, dtype=torch.long, device=device)
    alt_t  = torch.tensor(alt_vals, dtype=torch.float32, device=device)
    slope_t= torch.tensor(slope_vals, dtype=torch.float32, device=device)

    # --- full-grid targets ---
    y_node = torch.zeros(Nmax, dtype=torch.long, device=device)
    y_node[active_idx] = 1

    y_fuel  = torch.full((Nmax,), -100, dtype=torch.long, device=device)  # -100 = ignore
    y_alt   = torch.zeros(Nmax, dtype=torch.float32, device=device)
    y_slope = torch.zeros(Nmax, dtype=torch.float32, device=device)
    y_fuel[active_idx]  = fuel_t
    y_alt[active_idx]   = alt_t
    y_slope[active_idx] = slope_t

    # --- candidate edge labels (directed) ---
    E_all = template.edge_index_cand.size(1)
    y_edge = torch.zeros(E_all, dtype=torch.long, device=device)

    # Build a fast lookup for candidate edges
    src_all, dst_all = template.edge_index_cand
    cand_lookup: Dict[Tuple[int,int], int] = {(int(src_all[e].item()), int(dst_all[e].item())): e for e in range(E_all)}

    # Iterate true directed edges in G; only label those that are in the candidate set
    for (u, v) in G.edges():
        # Map u,v to slots
        # reuse same mapping logic:
        su, sv = to_slot(u), to_slot(v)
        idx = cand_lookup.get((su, sv))
        if idx is not None:
            y_edge[idx] = 1

    # --- encoder view (present nodes only) ---
    idx_tensor = torch.tensor(active_idx, dtype=torch.long, device=device)
    pos_enc = template.pos_all[idx_tensor]                      # [Na, 2]
    # Minimal encoder features: pos + alt + slope + fuel_scalar
    fuel_scalar = fuel_t.float().unsqueeze(1)
    x_enc = torch.cat([pos_enc, alt_t.unsqueeze(1), slope_t.unsqueeze(1), fuel_scalar], dim=1)  # [Na, 5]

    # Encoder edges
    if G.number_of_edges() > 0:
        # Build local index: slot -> 0..Na-1
        loc = {slot: i for i, slot in enumerate(active_idx)}
        enc_src, enc_dst = [], []
        for (u, v) in G.edges():
            su = to_slot(u)
            sv = to_slot(v)
            if su in loc and sv in loc:
                enc_src.append(loc[su]); enc_dst.append(loc[sv])
        if len(enc_src) == 0:
            # fall back to induced 8-neighbor graph
            enc_src, enc_dst = [], []
            for (i, j) in zip(src_all.tolist(), dst_all.tolist()):
                if i in loc and j in loc: enc_src.append(loc[i]); enc_dst.append(loc[j])

    edge_index_enc = torch.tensor([enc_src, enc_dst], dtype=torch.long, device=device)

    data = Data(
        x=x_enc,                        # [Na, 5] (pos2 + alt + slope + fuel_scalar)
        edge_index_enc=edge_index_enc,  # [2, E_enc]
        y_node_present=y_node,          # [Nmax] {0,1}
        y_fuel=y_fuel,                  # [Nmax] {-100 or class}
        y_alt=y_alt,                    # [Nmax] float
        y_slope=y_slope,                # [Nmax] float
        y_edge=y_edge                   # [E_all] {0,1}
    )
    return data


class GraphDatasetV3(torch.utils.data.Dataset):
    def __init__(self, root,
                 use_true_edges_for_encoder: bool = True,
                 alt_key: str = "altitude", slope_key: str = "slope", fuel_key: str = "fuel",
                 slot_key: str = "slot"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.template = build_grid_template(20, 20, device=device)
        self.use_true_edges_for_encoder = use_true_edges_for_encoder
        self.alt_key, self.slope_key, self.fuel_key = alt_key, slope_key, fuel_key
        self.slot_key= slot_key
        ids = {} # dictionary to store the ids of the files
        n = 0
        # iterate over the files in the directory
        for fname in glob.glob(f'{root}/*.pickle'):
            ids[n] = fname
            n += 1
        self.ids = ids
        self.landscape_dir = f'{root}/../instance'
            # Loads elevation .asc into a numpy array
        with codecs.open(f'{self.landscape_dir}/elevation.asc', encoding='utf-8-sig', ) as f:
            line = "_"
            elevation = []
            while line:
                line = f.readline()
                line_list = line.split()
                if len(line_list) > 2:
                    for i in line_list:
                        elevation.append(float(i))
        elevation = np.array(elevation)

        # Loads slope .asc into a numpy array
        with codecs.open(f'{self.landscape_dir}/slope.asc', encoding='utf-8-sig', ) as f:
            line = "_"
            slope = []
            while line:
                line = f.readline()
                line_list = line.split()
                if len(line_list) > 2:
                    for i in line_list:
                        slope.append(float(i))
        slope = np.array(slope)

        # Loads elevation .saz into a numpy array
        with codecs.open(f'{self.landscape_dir}/Forest.asc', encoding='utf-8-sig', ) as f:
            line = "_"
            fuel = []
            while line:
                line = f.readline()
                line_list = line.split()
                if len(line_list) > 2:
                    for i in line_list:
                        fuel.append(float(i))
        fuel = np.array(fuel)
        uniq = np.unique(fuel)                      # sorted unique values
        lut = {v: i for i, v in enumerate(uniq)}  # 1..K

        # vectorized remap
        fuel = np.fromiter((lut[v] for v in fuel), dtype=np.int32, count=fuel.size)

        self.fuel_classes = len(uniq)  

        # Loads elevation .saz into a numpy array
        with codecs.open(f'{self.landscape_dir}/saz.asc', encoding='utf-8-sig', ) as f:
            line = "_"
            saz = []
            while line:
                line = f.readline()
                line_list = line.split()
                if len(line_list) > 2:
                    for i in line_list:
                        saz.append(float(i))
        saz = np.array(saz)

        # Stacks array into a tensor, generating a landscape tensor
        self.landscape = torch.from_numpy(np.stack([ elevation, slope, saz]))
        # We compute means + std per channel to normalize
        means = torch.mean(self.landscape, dim=(1))
        stds = torch.std(self.landscape, dim=(1))
        norm = Normalize(means, stds)
        # Normalizes landscape
        self.landscape = norm(self.landscape.view(3, 20, 20)).view(3, 400)
        self.landscape = torch.stack((torch.from_numpy(fuel), self.landscape[0], self.landscape[1], self.landscape[1]), dim=0)
        

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        G = pickle.load(open(f'{self.ids[idx]}', 'rb'))
        for node in G.nodes():
            G.nodes[node]['slot'] = node - 1
            G.nodes[node]['altitude'] = self.landscape[1][node - 1]
            G.nodes[node]['slope'] = self.landscape[2][node - 1]
            G.nodes[node]['fuel'] = self.landscape[0][node - 1]
        return build_graph_data_from_nx(
            G=G,
            template=self.template,
            fuel_classes=self.fuel_classes,
            use_true_edges_for_encoder=self.use_true_edges_for_encoder,
            alt_key=self.alt_key, slope_key=self.slope_key, fuel_key=self.fuel_key,
            slot_key=self.slot_key
        )
    
def sigmoid(x):
    return 1 / (1 + torch.exp(-x)) if torch.is_tensor(x) else 1 / (1 + np.exp(-x))

def evaluate_all(model, val_loader, template, device, beta_kl=1e-3, edge_thr=0.5):
    model.eval()
    # --- node counters ---
    node_correct = 0
    node_total   = 0
    node_tp = node_fp = node_fn = 0

    # --- fuel/cont. stats ---
    fuel_correct = 0
    fuel_total   = 0
    # collect preds/labels to compute P/R/F1
    fuel_true_list = []
    fuel_pred_list = []

    alt_ae = alt_se = 0.0
    slope_ae = slope_se = 0.0
    feat_count = 0

    # --- edge counters ---
    edge_correct = 0
    edge_total   = 0
    edge_tp = edge_fp = edge_fn = 0
    edge_logits_all = []
    edge_labels_all = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            # forward
            output, mu, logvar = model(batch.x, batch.edge_index_enc, batch.batch)
            nodes, edges8, fuels, alts, slopes = output  # [B,N,1], [B,N,8], [B,N,C], [B,N,1], [B,N,1]
            B = batch.num_graphs
            N = template.Nmax
            E = template.edge_index_cand.size(1)

            # targets
            y_node  = batch.y_node_present.view(B, N).to(device)     # [B,N], {0,1}
            y_fuel  = batch.y_fuel.view(B, N).to(device)             # [B,N], {-100 or class}
            y_alt   = batch.y_alt.view(B, N).to(device)              # [B,N]
            y_slope = batch.y_slope.view(B, N).to(device)            # [B,N]
            y_edgeE = batch.y_edge.view(B, E).to(device)             # [B,E]

            # -------- Node existence metrics --------
            node_probs = torch.sigmoid(nodes.squeeze(-1))            # [B,N]
            node_pred  = (node_probs >= 0.5).long()
            y_node_long = y_node.long()

            node_correct += (node_pred == y_node_long).sum().item()
            node_total   += y_node.numel()

            node_tp += ((node_pred == 1) & (y_node_long == 1)).sum().item()
            node_fp += ((node_pred == 1) & (y_node_long == 0)).sum().item()
            node_fn += ((node_pred == 0) & (y_node_long == 1)).sum().item()

            # -------- Fuel / Alt / Slope on present nodes --------
            present_mask = (y_node == 1) & (y_fuel >= 0)                  # ignore -100
            if present_mask.any():
                fuel_pred = fuels.argmax(dim=-1)                          # [B,N]
                # accuracy
                fuel_correct += (fuel_pred[present_mask] == y_fuel[present_mask]).sum().item()
                fuel_total   += int(present_mask.sum().item())
                # collect for P/R/F1
                fuel_true_list.append(y_fuel[present_mask].detach().cpu())
                fuel_pred_list.append(fuel_pred[present_mask].detach().cpu())

                # continuous
                alt_pred   = alts.squeeze(-1)
                slope_pred = slopes.squeeze(-1)

                diff_alt   = (alt_pred  - y_alt).abs()[present_mask]
                diff_slope = (slope_pred - y_slope).abs()[present_mask]

                alt_ae   += diff_alt.sum().item()
                alt_se   += (diff_alt ** 2).sum().item()
                slope_ae += diff_slope.sum().item()
                slope_se += (diff_slope ** 2).sum().item()
                feat_count += int(present_mask.sum().item())

            # -------- Edge metrics (node/dir → edge_idx) --------
            node_dir_to_e   = template.node_dir_to_e.to(device)                    # [N,8], -1 invalid
            node_dir_to_dst = template.node_dir_to_dst.to(device)                  # [N,8]
            valid_nd = (node_dir_to_e >= 0)                             # [N,8]

            # Gather labels for valid (node,dir)
            e_idx = node_dir_to_e[valid_nd]                             # [M]
            y_edge_valid = y_edgeE.gather(1, e_idx.unsqueeze(0).expand(B, -1))  # [B,M]

            # Gate: both endpoints exist
            dst_idx_safe = node_dir_to_dst.clamp_min(0)                 # [N,8]
            y_node_exp   = y_node.unsqueeze(-1).expand(-1, -1, 8)       # [B,N,8]
            idx_for_gath = dst_idx_safe.unsqueeze(0).expand(B, -1, -1)  # [B,N,8]
            y_dst = torch.gather(y_node_exp, 1, idx_for_gath)           # [B,N,8]
            both_exist = (y_node.unsqueeze(-1) == 1) & (y_dst == 1) & valid_nd.unsqueeze(0)  # [B,N,8]

            # Logits at valid (node,dir)
            edges8_flat = edges8.view(B, -1)                            # [B,N*8]
            nd_pos = valid_nd.nonzero(as_tuple=False)                   # [M,2]
            idx_flat = (nd_pos[:, 0] * 8 + nd_pos[:, 1]).to(device)     # [M]
            logits_valid = edges8_flat.index_select(1, idx_flat)        # [B,M]

            both_exist_flat = both_exist.view(B, -1).index_select(1, idx_flat)  # [B,M]
            if both_exist_flat.any():
                logits_keep = logits_valid[both_exist_flat]             # [K]
                labels_keep = y_edge_valid[both_exist_flat]             # [K]

                probs_keep = torch.sigmoid(logits_keep)
                edge_pred  = (probs_keep >= edge_thr).float()

                edge_correct += (edge_pred == labels_keep).sum().item()
                edge_total   += labels_keep.numel()

                lk = labels_keep.long()
                ep = edge_pred.long()
                edge_tp += ((ep == 1) & (lk == 1)).sum().item()
                edge_fp += ((ep == 1) & (lk == 0)).sum().item()
                edge_fn += ((ep == 0) & (lk == 1)).sum().item()

                edge_logits_all.append(logits_keep.detach().cpu())
                edge_labels_all.append(labels_keep.detach().cpu())

    # ---- aggregate ----
    eps = 1e-12

    node_acc = node_correct / max(1, node_total)
    node_prec = node_tp / max(1, (node_tp + node_fp))
    node_rec  = node_tp / max(1, (node_tp + node_fn))
    node_f1   = (2 * node_prec * node_rec) / max(eps, (node_prec + node_rec))

    fuel_acc = fuel_correct / max(1, fuel_total)

    # fuel precision/recall/F1 (multi-class)
    fuel_precision_micro = fuel_recall_micro = fuel_f1_micro = None
    fuel_precision_macro = fuel_recall_macro = fuel_f1_macro = None
    fuel_precision_weighted = fuel_recall_weighted = fuel_f1_weighted = None

    if len(fuel_true_list) > 0:
        import numpy as np
        y_true = torch.cat(fuel_true_list, dim=0).numpy()
        y_pred = torch.cat(fuel_pred_list, dim=0).numpy()
        
        p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )
        p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        p_weight, r_weight, f_weight, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        fuel_precision_micro, fuel_recall_micro, fuel_f1_micro = float(p_micro), float(r_micro), float(f_micro)
        fuel_precision_macro, fuel_recall_macro, fuel_f1_macro = float(p_macro), float(r_macro), float(f_macro)
        fuel_precision_weighted, fuel_recall_weighted, fuel_f1_weighted = float(p_weight), float(r_weight), float(f_weight)
    else:
        # micro averages equal accuracy in single-label multi-class
        fuel_precision_micro = fuel_recall_micro = fuel_f1_micro = fuel_acc

    alt_mae   = alt_ae / max(1, feat_count)
    slope_mae = slope_ae / max(1, feat_count)
    alt_rmse   = math.sqrt(alt_se   / max(1, feat_count))
    slope_rmse = math.sqrt(slope_se / max(1, feat_count))

    edge_acc = edge_correct / max(1, edge_total)
    edge_prec = edge_tp / max(1, (edge_tp + edge_fp))
    edge_rec  = edge_tp / max(1, (edge_tp + edge_fn))
    edge_f1   = (2 * edge_prec * edge_rec) / max(eps, (edge_prec + edge_rec))

    metrics = {
        # nodes
        "node_acc": node_acc,
        "node_precision": node_prec,
        "node_recall": node_rec,
        "node_f1": node_f1,
        # edges
        "edge_acc": edge_acc,
        "edge_precision": edge_prec,
        "edge_recall": edge_rec,
        "edge_f1": edge_f1,
        # fuel
        "fuel_acc": fuel_acc,
        "fuel_precision_micro": fuel_precision_micro,
        "fuel_recall_micro": fuel_recall_micro,
        "fuel_f1_micro": fuel_f1_micro,
        "fuel_precision_macro": fuel_precision_macro,
        "fuel_recall_macro": fuel_recall_macro,
        "fuel_f1_macro": fuel_f1_macro,
        "fuel_precision_weighted": fuel_precision_weighted,
        "fuel_recall_weighted": fuel_recall_weighted,
        "fuel_f1_weighted": fuel_f1_weighted,
        # continuous
        "alt_mae": alt_mae,
        "alt_rmse": alt_rmse,
        "slope_mae": slope_mae,
        "slope_rmse": slope_rmse,
    }

    return metrics


@torch.no_grad()
def outputs_to_networkx(
    model,                          # GRAPH_VAE_V3
    output=None,                    # (nodes, edges8, fuels, alts, slopes)
    z=None,                         # optional latent [B,z] or [z]
    tau_node: float = 0.5,
    tau_edge: float = 0.5,
    gate_mode: str = "logit",       # "logit" | "prob" | "none"
    return_list: bool = False,      # if False and B==1, return a single graph
    prune_isolates: bool = True,
):
    """
    Reconstruct NX DiGraph(s) from model outputs.

    Node attrs: slot, row, col, pos_norm, p_node, fuel, fuel_prob, alt, slope
    Edge attrs: dir, p_edge (gated), p_edge_raw (ungated)

    gate_mode:
      - "logit": p = sigmoid( logits_edge + logit(p_src) + logit(p_dst) )
      - "prob" : p = sigmoid(logits_edge) * p_src * p_dst
      - "none" : p = sigmoid(logits_edge)
    """
    assert (output is not None) or (z is not None), "Provide either `output` or `z`."
    device = next(model.parameters()).device
    template = model.template
    H, W, N = template.H, template.W, template.Nmax

    # Decode if needed
    if output is None:
        if z.dim() == 1:
            z = z.unsqueeze(0)
        output = model.decode(z.to(device))
    nodes_logits, edges8_logits, fuel_logits, alt_pred, slope_pred = output
    B = nodes_logits.size(0)

    # Build node->dir->dst map once
    src_all, dst_all = template.edge_index_cand   # [2,E]
    dir_id = template.dir_id                      # [E]
    node_dir_to_dst = torch.full((N, 8), -1, dtype=torch.long, device=device)
    for e in range(src_all.numel()):
        s = int(src_all[e]); d = int(dir_id[e]); t = int(dst_all[e])
        node_dir_to_dst[s, d] = t
    valid_nd = (node_dir_to_dst >= 0)

    # Helper for gating edges
    def gate_edges(logits_e, p_src, p_dst):
        if gate_mode == "logit":
            logit_src = torch.logit(p_src.clamp(1e-6, 1-1e-6))
            logit_dst = torch.logit(p_dst.clamp(1e-6, 1-1e-6))
            return torch.sigmoid(logits_e + logit_src + logit_dst)
        elif gate_mode == "prob":
            return torch.sigmoid(logits_e) * p_src * p_dst
        else:  # "none"
            return torch.sigmoid(logits_e)

    graphs = []
    for b in range(B):
        # ---- Nodes ----
        logits_n = nodes_logits[b, :, 0]                         # [N]
        p_node   = torch.sigmoid(logits_n)                       # [N]
        keep_n   = (p_node > tau_node)                           # [N]

        # Node attributes
        fuel_b    = fuel_logits[b]                               # [N,C]
        fuel_cls  = fuel_b.argmax(dim=-1)                        # [N]
        fuel_prob = torch.softmax(fuel_b, dim=-1).max(dim=-1).values
        alt_b     = alt_pred[b, :, 0]                            # [N]
        slope_b   = slope_pred[b, :, 0]                          # [N]

        # Positions & slot info
        pos_norm = getattr(model, "pos_all", template.pos_all)   # [N,2]
        rows = torch.arange(N, device=device) // W
        cols = torch.arange(N, device=device) %  W

        import networkx as nx
        G = nx.DiGraph()
        idx_present = torch.nonzero(keep_n, as_tuple=False).flatten()
        for i in idx_present.tolist():
            G.add_node(
                i,
                slot=int(i),
                row=int(rows[i].item()),
                col=int(cols[i].item()),
                pos_norm=(float(pos_norm[i,0].item()), float(pos_norm[i,1].item())),
                p_node=float(p_node[i].item()),
                fuel=int(fuel_cls[i].item()),
                fuel_prob=float(fuel_prob[i].item()),
                alt=float(alt_b[i].item()),
                slope=float(slope_b[i].item()),
            )

        # quick exit if no nodes predicted
        if G.number_of_nodes() == 0:
            graphs.append(G)
            continue

        # ---- Edges ----
        edges8_b = edges8_logits[b]                              # [N,8]
        nd_valid = valid_nd & keep_n.unsqueeze(1)                # [N,8] valid source present
        src_idx, dir_idx = torch.nonzero(nd_valid, as_tuple=True)

        if src_idx.numel() > 0:
            dst_idx = node_dir_to_dst[src_idx, dir_idx]          # [M]
            dst_present = keep_n[dst_idx]                        # [M]
            mask_both = dst_present
            src_idx, dir_idx, dst_idx = src_idx[mask_both], dir_idx[mask_both], dst_idx[mask_both]

            if src_idx.numel() > 0:
                logits_e   = edges8_b[src_idx, dir_idx]          # [M]
                p_src      = p_node[src_idx]                     # [M]
                p_dst      = p_node[dst_idx]                     # [M]
                p_edge_raw = torch.sigmoid(logits_e)
                p_edge     = gate_edges(logits_e, p_src, p_dst)

                # --- anti–2-cycle inference guard ---
                # Keep at most one direction per unordered pair, choosing the higher-prob one.
                selected = {}  # key=(min(s,t), max(s,t)) -> dict with chosen (s,t,d,pr,pr_raw)
                s_list = src_idx.tolist()
                d_list = dir_idx.tolist()
                t_list = dst_idx.tolist()
                pr_list = p_edge.tolist()
                prr_list = p_edge_raw.tolist()

                for s, d, t, pr, pr_raw in zip(s_list, d_list, t_list, pr_list, prr_list):
                    if pr < tau_edge:                   # threshold first
                        continue
                    if s == t:                          # (shouldn't happen in 8-neigh), ignore self-loops
                        continue
                    key = (s, t) if s < t else (t, s)  # undirected key
                    prev = selected.get(key)
                    if (prev is None) or (pr > prev["pr"]):
                        selected[key] = {"s": int(s), "t": int(t), "d": int(d),
                                         "pr": float(pr), "pr_raw": float(pr_raw)}

                # add only the selected directions
                for info in selected.values():
                    s, t = info["s"], info["t"]
                    if (s in G) and (t in G):
                        G.add_edge(s, t, dir=info["d"], p_edge=info["pr"], p_edge_raw=info["pr_raw"])

        # ---- prune isolates (unconnected nodes) ----
        if prune_isolates:
            iso = list(nx.isolates(G))
            if iso:
                G.remove_nodes_from(iso)

        graphs.append(G)

    if (not return_list) and len(graphs) == 1:
        return graphs[0]
    return graphs


@torch.no_grad()
def metrics_on_batch_tm(model, batch, template, device, node_thr=0.5, edge_thr=0.5):
    """
    Compute metrics on the CURRENT batch only, using torchmetrics where convenient.
    Returns a flat dict of floats.
    """
    model.eval()
    batch = batch.to(device)
    output, mu, logvar = model(batch.x, batch.edge_index_enc, batch.batch)
    nodes, edges8, fuels, alts, slopes = output  # [B,N,1], [B,N,8], [B,N,C], [B,N,1], [B,N,1]
    B = batch.num_graphs
    N = template.Nmax
    E = template.edge_index_cand.size(1)

    # ------- targets -------
    y_node  = batch.y_node_present.view(B, N).to(device)      # {0,1}
    y_fuel  = batch.y_fuel.view(B, N).to(device)              # {-100 or class}
    y_edgeE = batch.y_edge.view(B, E).to(device)              # {0,1}

    # ------- node existence (binary) -------
    node_probs = torch.sigmoid(nodes.squeeze(-1))             # [B,N] in [0,1]
    node_acc   = BinaryAccuracy(threshold=node_thr).to(device)(node_probs, y_node.int())
    node_prec  = BinaryPrecision(threshold=node_thr).to(device)(node_probs, y_node.int())
    node_rec   = BinaryRecall(threshold=node_thr).to(device)(node_probs, y_node.int())
    node_f1    = BinaryF1Score(threshold=node_thr).to(device)(node_probs, y_node.int())

    # ------- fuel (multiclass on present nodes only) -------
    present_mask = (y_node == 1) & (y_fuel >= 0)
    fuel_acc = fuel_prec_macro = fuel_rec_macro = fuel_f1_macro = torch.tensor(0., device=device)
    if present_mask.any():
        fuel_pred = fuels.argmax(dim=-1)                  # [B,N]
        y_true_f  = y_fuel[present_mask]
        y_pred_f  = fuel_pred[present_mask]
        C = int(fuels.size(-1))

        fuel_acc        = MulticlassAccuracy(num_classes=C, average="micro").to(device)(y_pred_f, y_true_f)
        fuel_prec_macro = MulticlassPrecision(num_classes=C, average="macro").to(device)(y_pred_f, y_true_f)
        fuel_rec_macro  = MulticlassRecall(num_classes=C, average="macro").to(device)(y_pred_f, y_true_f)
        fuel_f1_macro   = MulticlassF1Score(num_classes=C, average="macro").to(device)(y_pred_f, y_true_f)

    # ------- edges (binary; only for valid (src,dir) and where both endpoints exist) -------
    node_dir_to_e   = template.node_dir_to_e.to(device)       # [N,8], -1 invalid
    node_dir_to_dst = template.node_dir_to_dst.to(device)     # [N,8], -1 invalid
    valid_nd = (node_dir_to_e >= 0)

    # gather GT labels on valid slots
    e_idx = node_dir_to_e[valid_nd]                           # [M]
    y_edge_valid = y_edgeE.gather(1, e_idx.unsqueeze(0).expand(B, -1))  # [B,M]

    # gate by endpoints existence
    dst_idx_safe = node_dir_to_dst.clamp_min(0)
    y_node_exp   = y_node.unsqueeze(-1).expand(-1, -1, 8)     # [B,N,8]
    idx_for_gath = dst_idx_safe.unsqueeze(0).expand(B, -1, -1)
    y_dst = torch.gather(y_node_exp, 1, idx_for_gath)         # [B,N,8]
    both_exist = (y_node.unsqueeze(-1) == 1) & (y_dst == 1) & valid_nd.unsqueeze(0)  # [B,N,8]

    # logits at valid (node,dir)
    edges8_flat = edges8.view(B, -1)                          # [B,N*8]
    nd_pos = valid_nd.nonzero(as_tuple=False)                 # [M,2]
    idx_flat = (nd_pos[:, 0] * 8 + nd_pos[:, 1]).to(device)   # [M]
    logits_valid = edges8_flat.index_select(1, idx_flat)      # [B,M]
    both_exist_flat = both_exist.view(B, -1).index_select(1, idx_flat)  # [B,M]

    edge_acc = edge_prec = edge_rec = edge_f1 = torch.tensor(0., device=device)
    if both_exist_flat.any():
        # keep only entries where both endpoints exist
        logits_keep = logits_valid[both_exist_flat]           # [K]
        labels_keep = y_edge_valid[both_exist_flat].int()     # [K]
        probs_keep  = torch.sigmoid(logits_keep)              # [K]

        edge_acc  = BinaryAccuracy(threshold=edge_thr).to(device)(probs_keep, labels_keep)
        edge_prec = BinaryPrecision(threshold=edge_thr).to(device)(probs_keep, labels_keep)
        edge_rec  = BinaryRecall(threshold=edge_thr).to(device)(probs_keep, labels_keep)
        edge_f1   = BinaryF1Score(threshold=edge_thr).to(device)(probs_keep, labels_keep)

    # Return raw floats
    return {
        "node/acc":  float(node_acc.item()),
        "node/prec": float(node_prec.item()),
        "node/rec":  float(node_rec.item()),
        "node/f1":   float(node_f1.item()),
        "fuel/acc":  float(fuel_acc.item()),
        "fuel/prec_macro": float(fuel_prec_macro.item()),
        "fuel/rec_macro":  float(fuel_rec_macro.item()),
        "fuel/f1_macro":   float(fuel_f1_macro.item()),
        "edge/acc":  float(edge_acc.item()),
        "edge/prec": float(edge_prec.item()),
        "edge/rec":  float(edge_rec.item()),
        "edge/f1":   float(edge_f1.item()),
    }

@torch.no_grad()
def estimate_edge_pos_weight(train_loader, template, device):
    total_pos = 0
    total_cnt = 0
    for batch in train_loader:
        batch = batch.to(device)
        B = batch.num_graphs
        N = template.Nmax
        E = template.edge_index_cand.size(1)

        y_node  = batch.y_node_present.view(B, N)        # [B,N]
        y_edgeE = batch.y_edge.view(B, E)                 # [B,E]

        node_dir_to_e   = template.node_dir_to_e.to(device)      # [N,8]
        node_dir_to_dst = template.node_dir_to_dst.to(device)    # [N,8]
        valid_nd = (node_dir_to_e >= 0)
        e_idx = node_dir_to_e[valid_nd]                           # [M]
        y_edge_valid = y_edgeE.gather(1, e_idx.unsqueeze(0).expand(B, -1))  # [B,M]

        # gate by endpoints existence
        dst_idx_safe = node_dir_to_dst.clamp_min(0)
        y_node_exp = y_node.unsqueeze(-1).expand(-1, -1, 8)       # [B,N,8]
        idx_g = dst_idx_safe.unsqueeze(0).expand(B, -1, -1)
        y_dst = torch.gather(y_node_exp, 1, idx_g)                # [B,N,8]
        both_exist = (y_node.unsqueeze(-1) == 1) & (y_dst == 1) & valid_nd.unsqueeze(0)  # [B,N,8]

        # flatten to valid positions
        nd_pos = valid_nd.nonzero(as_tuple=False)                  # [M,2]
        idx_flat = (nd_pos[:, 0] * 8 + nd_pos[:, 1]).to(device)    # [M]
        both_exist_flat = both_exist.view(B, -1).index_select(1, idx_flat)  # [B,M]

        labels = y_edge_valid[both_exist_flat].float()             # [K]
        total_pos += labels.sum().item()
        total_cnt += labels.numel()

    p = total_pos / max(1, total_cnt)
    pos_weight = (1 - p) / max(p, 1e-6)
    return pos_weight, p



def _find_latest_matching(base):
    # prioriza _last.pt; si no existe, toma el más nuevo por mtime
    last = f"{base}_best.pt"
    if os.path.exists(last):
        return last
    candidates = glob.glob(f"{base}*.pt")
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism (slower but repeatable)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass  # older PyTorch

def seed_worker(worker_id):
    # make each worker deterministic, based on the initial seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
