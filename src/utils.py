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
    edge_index_cand: Tensor      # [2, E_dir] directed 8-neighbor candidate edges
    dir_id: Tensor               # [E_dir] int64 in [0..7]
    node_mask_all: Tensor        # [Nmax] bool (True in-bounds)
    edge_mask_all: Tensor        # [E_dir] bool (True for all candidates)

    # convenience maps for directed edges
    node_dir_to_e: Tensor        # [Nmax, 8] -> edge idx or -1
    node_dir_to_dst: Tensor      # [Nmax, 8] -> dst slot or -1

    # -------- NEW: precomputed undirected neighbor pairs --------
    # unique pairs (a,b) with a < b among 8-neighbors
    undir_src: Tensor            # [E_u]
    undir_dst: Tensor            # [E_u]
    # mapping back to the two directed edges for each undirected pair
    # -1 if that direction does not exist (shouldn’t happen on a regular grid)
    undir_e1: Tensor             # [E_u] index of a->b in edge_index_cand or -1
    undir_e2: Tensor             # [E_u] index of b->a in edge_index_cand or -1
    E_u: int                     # number of undirected pairs

def build_grid_template(H: int, W: int, device: Optional[torch.device]=None) -> GridTemplate:
    device = device or torch.device('cpu')
    Nmax = H * W

    # node positions in [-1, 1]
    rows = torch.arange(H, device=device, dtype=torch.float32)
    cols = torch.arange(W, device=device, dtype=torch.float32)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    rr_n = (rr / (H-1) - 0.5) * 2.0 if H > 1 else torch.zeros_like(rr)
    cc_n = (cc / (W-1) - 0.5) * 2.0 if W > 1 else torch.zeros_like(cc)
    pos_all = torch.stack([rr_n.reshape(-1), cc_n.reshape(-1)], dim=1)  # [Nmax, 2]

    # directed 8-neighbor candidates
    src_list, dst_list, dir_list = [], [], []
    for r in range(H):
        for c in range(W):
            i = r * W + c
            for (dr, dc) in DIR8:
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < H and 0 <= c2 < W:
                    j = r2 * W + c2
                    src_list.append(i)
                    dst_list.append(j)
                    dir_list.append(DIR_TO_ID[(dr, dc)])

    edge_index_cand = torch.tensor([src_list, dst_list], device=device, dtype=torch.long)  # [2, E_dir]
    dir_id = torch.tensor(dir_list, device=device, dtype=torch.long)                       # [E_dir]
    node_mask_all = torch.ones(Nmax, device=device, dtype=torch.bool)
    edge_mask_all = torch.ones(edge_index_cand.size(1), device=device, dtype=torch.bool)

    # directed helper maps
    src, dst = edge_index_cand
    node_dir_to_e   = torch.full((Nmax, 8), -1, dtype=torch.long, device=device)
    node_dir_to_dst = torch.full((Nmax, 8), -1, dtype=torch.long, device=device)
    for e in range(src.numel()):
        s = int(src[e]); d = int(dir_id[e]); t = int(dst[e])
        node_dir_to_e[s, d] = e
        node_dir_to_dst[s, d] = t

    # --------- Build undirected pairs & mapping to directed edges ---------
    # Use (a,b) with a < b to represent each neighbor pair exactly once
    directed_pairs = list(zip(src_list, dst_list))
    lookup = {(u, v): i for i, (u, v) in enumerate(directed_pairs)}

    u_src, u_dst, e1, e2 = [], [], [], []
    seen = set()
    for (u, v) in directed_pairs:
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        idx_ab = lookup.get((a, b), -1)  # a->b
        idx_ba = lookup.get((b, a), -1)  # b->a
        u_src.append(a)
        u_dst.append(b)
        e1.append(idx_ab)
        e2.append(idx_ba)

    undir_src = torch.tensor(u_src, dtype=torch.long, device=device)
    undir_dst = torch.tensor(u_dst, dtype=torch.long, device=device)
    undir_e1  = torch.tensor(e1,   dtype=torch.long, device=device)
    undir_e2  = torch.tensor(e2,   dtype=torch.long, device=device)
    E_u = int(undir_src.numel())

    # pack template
    template = GridTemplate(
        H=H, W=W, Nmax=Nmax,
        pos_all=pos_all,
        edge_index_cand=edge_index_cand,
        dir_id=dir_id,
        node_mask_all=node_mask_all,
        edge_mask_all=edge_mask_all,
        node_dir_to_e=node_dir_to_e,
        node_dir_to_dst=node_dir_to_dst,
        undir_src=undir_src,
        undir_dst=undir_dst,
        undir_e1=undir_e1,
        undir_e2=undir_e2,
        E_u=E_u,
    )
    return template

# ----------------------------
# Nx -> PyG Data (Path A)
# ----------------------------

# ----------------------------
# Nx -> PyG Data (Path A)  (UNDIRECTED labels)
# ----------------------------

def build_graph_data_from_nx(
    G: nx.DiGraph,
    template: GridTemplate,
    fuel_classes: int,
    y_ign_idx: int,
    use_true_edges_for_encoder: bool = True,
    alt_key: str = "altitude",
    slope_key: str = "slope",
    fuel_key: str = "fuel",
    slot_key: str = "slot",
) -> Data:
    """
    Convert a networkx.DiGraph into a PyG Data with full-grid targets.

    Produces:
      - y_node_present: [Nmax]        (0/1)
      - y_fuel:         [Nmax]        (-100 ignored or class id)
      - y_alt:          [Nmax]        (float)
      - y_slope:        [Nmax]        (float)
      - y_edge_u:       [E_u]         (0/1)  <-- undirected labels over unique neighbor pairs
      - y_edge_dir:     [E_dir]       (0/1)  (kept for compatibility; optional consumer)
    """
    def to_slot(node):
        a = G.nodes[node]
        if slot_key in a:
            return int(a[slot_key])
        raise ValueError(f"Cannot infer slot for edge endpoint {node} (missing '{slot_key}').")

    device = torch.device("cpu")   # keep CPU to avoid CUDA in dataloader workers
    H, W, Nmax = template.H, template.W, template.Nmax

    # --- resolve slot ids for nodes and collect per-node attributes ---
    active_idx: List[int] = []
    fuel_vals: List[int] = []
    alt_vals: List[float] = []
    slope_vals: List[float] = []

    # deterministic node order by slot
    slots_and_nodes = []
    for u, attrs in G.nodes(data=True):
        s = int(attrs[slot_key])
        if not (0 <= s < Nmax):
            raise ValueError(f"Slot {s} out of bounds for H={H}, W={W}.")
        slots_and_nodes.append((s, u))
    slots_and_nodes.sort(key=lambda x: x[0])  # sort by slot

    for s, u in slots_and_nodes:
        active_idx.append(s)
        attrs = G.nodes[u]
        # fuel
        if fuel_key not in attrs:
            raise ValueError(f"Node {u} missing '{fuel_key}' attribute.")
        fuel = int(attrs[fuel_key])
        if not (0 <= fuel < fuel_classes):
            raise ValueError(f"Fuel class {fuel} out of bounds [0,{fuel_classes-1}] for node {u}.")
        fuel_vals.append(fuel)
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

    # tensors for node attrs (in active order)
    fuel_t  = torch.tensor(fuel_vals, dtype=torch.long,    device=device)
    alt_t   = torch.tensor(alt_vals,  dtype=torch.float32, device=device)
    slope_t = torch.tensor(slope_vals,dtype=torch.float32, device=device)

    # --- full-grid node targets ---
    y_node = torch.zeros(Nmax, dtype=torch.long, device=device)
    if active_idx:
        y_node[active_idx] = 1

    y_fuel  = torch.full((Nmax,), -100, dtype=torch.long,    device=device)  # -100 = ignore
    y_alt   = torch.zeros(Nmax,    dtype=torch.float32,      device=device)
    y_slope = torch.zeros(Nmax,    dtype=torch.float32,      device=device)
    if active_idx:
        idx_tensor = torch.tensor(active_idx, dtype=torch.long, device=device)
        y_fuel[idx_tensor]  = fuel_t
        y_alt[idx_tensor]   = alt_t
        y_slope[idx_tensor] = slope_t

    # ---------------------------
    # Edge labels
    # ---------------------------

    # (A) UNDIRECTED labels over unique neighbor pairs
    E_u = template.E_u
    y_edge_u = torch.zeros(E_u, dtype=torch.long, device=device)  # [E_u]

    # map unordered pair (a<b) → undirected index
    # we can build a small lookup once here
    u_src = template.undir_src.tolist()
    u_dst = template.undir_dst.tolist()
    und_lookup: Dict[Tuple[int,int], int] = {(a, b): i for i, (a, b) in enumerate(zip(u_src, u_dst))}

    # For each directed true edge (u,v) in G, set its undirected pair label to 1
    for (u, v) in G.edges():
        su, sv = to_slot(u), to_slot(v)
        if su == sv:
            continue
        a, b = (su, sv) if su < sv else (sv, su)
        idx = und_lookup.get((a, b))
        if idx is not None:
            y_edge_u[idx] = 1

    # (B) (optional) DIRECTED labels for compatibility
    E_dir = template.edge_index_cand.size(1)
    y_edge_dir = torch.zeros(E_dir, dtype=torch.long, device=device)
    # quick directed candidate lookup
    src_all, dst_all = template.edge_index_cand
    cand_lookup_dir: Dict[Tuple[int,int], int] = {
        (int(src_all[e].item()), int(dst_all[e].item())): e for e in range(E_dir)
    }
    for (u, v) in G.edges():
        su, sv = to_slot(u), to_slot(v)
        idx = cand_lookup_dir.get((su, sv))
        if idx is not None:
            y_edge_dir[idx] = 1

    # ---------------------------
    # Encoder view (present nodes only)
    # ---------------------------
    idx_tensor = torch.tensor(active_idx, dtype=torch.long, device=device) if active_idx else torch.zeros(0, dtype=torch.long, device=device)
    pos_enc = template.pos_all[idx_tensor] if active_idx else torch.zeros((0,2), device=device)
    # Minimal encoder features: pos2 + alt + slope + fuel_scalar
    fuel_scalar = fuel_t.float().unsqueeze(1) if active_idx else torch.zeros((0,1), device=device)
    x_enc = torch.cat([pos_enc, alt_t.unsqueeze(1), slope_t.unsqueeze(1), fuel_scalar], dim=1) if active_idx else torch.zeros((0,5), device=device)

    # Encoder edges (use the true graph over active nodes if available; fallback to induced neighbors)
    enc_src, enc_dst = [], []
    if use_true_edges_for_encoder and G.number_of_edges() > 0 and active_idx:
        local = {s: i for i, s in enumerate(active_idx)}   # slot → local idx
        for (u, v) in G.edges():
            su, sv = to_slot(u), to_slot(v)
            if su in local and sv in local:
                enc_src.append(local[su]); enc_dst.append(local[sv])
    else:
        # fallback: induced 8-neighbor graph restricted to active slots
        local = {s: i for i, s in enumerate(active_idx)}
        for a, b in zip(u_src, u_dst):
            if a in local and b in local:
                # add both directions for message passing
                enc_src.append(local[a]); enc_dst.append(local[b])
                enc_src.append(local[b]); enc_dst.append(local[a])

    edge_index_enc = torch.tensor([enc_src, enc_dst], dtype=torch.long, device=device) if len(enc_src) > 0 \
                     else torch.zeros((2,0), dtype=torch.long, device=device)
    ignition = torch.tensor(y_ign_idx, dtype=torch.long)

    # ---------------------------
    # Pack PyG Data
    # ---------------------------
    data = Data(
        x=x_enc,                        # [Na, 5]
        edge_index_enc=edge_index_enc,  # [2, E_enc]
        y_node_present=y_node,          # [Nmax] {0,1}
        y_fuel=y_fuel,                  # [Nmax] {-100 or class}
        y_alt=y_alt,                    # [Nmax] float
        y_slope=y_slope,                # [Nmax] float
        # edge labels:
        y_edge_u=y_edge_u,              # [E_u] undirected labels  (NEW, preferred)
        y_edge_dir=y_edge_dir,          # [E_dir] directed labels  (compatibility)
        y_ign_idx=ignition,                # ignition slot index
    )
    return data



class GraphDatasetV3(Dataset):
    """
    Loads pickled NetworkX graphs from `root/*.pickle`, attaches per-node attributes
    (slot, altitude, slope, fuel), and converts each into a PyG `Data` via
    `build_graph_data_from_nx`, which now produces UNDIRECTED edge labels (y_edge_u).

    Notes
    -----
    * The template is built on CPU to avoid CUDA worker issues in DataLoader.
    * Node ids in your pickles are assumed to be 1-based; we map to slot = node-1.
    * Landscape channels are normalized once and cached.
    """
    def __init__(
        self,
        root: str,
        use_true_edges_for_encoder: bool = True,
        alt_key: str = "altitude",
        slope_key: str = "slope",
        fuel_key: str = "fuel",
        slot_key: str = "slot",
        H: int = 20,
        W: int = 20,
    ):
        # Force CPU in the template to be dataloader-safe with num_workers > 0
        self.template = build_grid_template(H, W, device=torch.device("cpu"))

        self.use_true_edges_for_encoder = use_true_edges_for_encoder
        self.alt_key, self.slope_key, self.fuel_key = alt_key, slope_key, fuel_key
        self.slot_key = slot_key

        # collect files (sorted for determinism)
        paths = sorted(glob.glob(f"{root}/*.pickle"))
        self.ids = {i: p for i, p in enumerate(paths)}

        # ---- load landscape rasters & preprocess ----
        self.landscape_dir = f"{root}/../instance"

        def _load_asc(path):
            vals = []
            with codecs.open(path, encoding="utf-8-sig") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) > 2:
                        vals.extend(float(x) for x in parts)
            return np.array(vals, dtype=np.float32)

        elevation = _load_asc(f"{self.landscape_dir}/elevation.asc")  # shape: (H*W,)
        slope     = _load_asc(f"{self.landscape_dir}/slope.asc")      # shape: (H*W,)
        saz       = _load_asc(f"{self.landscape_dir}/saz.asc")        # shape: (H*W,)

        # fuel (categorical) remapped to 0..K-1
        fuel_raw = _load_asc(f"{self.landscape_dir}/Forest.asc")
        uniq = np.unique(fuel_raw)
        lut = {v: i for i, v in enumerate(uniq)}
        fuel = np.fromiter((lut[v] for v in fuel_raw), dtype=np.int32, count=fuel_raw.size)
        self.fuel_classes = int(len(uniq))

        # stack continuous channels and normalize (elev, slope, saz)
        cont = np.stack([elevation, slope, saz], axis=0)                  # [3, H*W]
        cont_t = torch.from_numpy(cont)                                    # float32
        means = torch.mean(cont_t, dim=1, keepdim=True)                    # [3,1]
        stds  = torch.std(cont_t,  dim=1, keepdim=True).clamp_min(1e-8)   # [3,1]
        cont_norm = (cont_t - means) / stds                                # [3, H*W]

        # final landscape tensor: [4, H*W] = [fuel(one int per cell), elev_n, slope_n, saz_n]
        self.landscape = torch.stack(
            (
                torch.from_numpy(fuel).to(torch.int64),  # fuel indices
                cont_norm[0],                            # normalized elevation
                cont_norm[1],                            # normalized slope
                cont_norm[2],                            # normalized saz
            ),
            dim=0
        )  # shapes: [1 int64, 3 float32], each of length H*W

        self.H, self.W, self.Nmax = H, W, H * W

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        # load pickled NX DiGraph
        G = pickle.load(open(self.ids[idx], "rb"))
        ignition = []
        # attach per-node attributes
        # nodes assumed 1-based; map to 0-based slots
        for node in G.nodes():
            slot = int(node) - 1
            if not (0 <= slot < self.Nmax):
                raise ValueError(f"Node {node} → slot {slot} out of bounds [0,{self.Nmax-1}]")
            G.nodes[node][self.slot_key]   = slot
            G.nodes[node][self.alt_key]    = float(self.landscape[1][slot].item())  # elevation (norm)
            G.nodes[node][self.slope_key]  = float(self.landscape[2][slot].item())  # slope (norm)
            G.nodes[node][self.fuel_key]   = int(self.landscape[0][slot].item())    # fuel class idx
            if G.in_degree(node) == 0:
                # isolated node; assign default fuel class 0
                ignition.append(slot)
                
        # convert to PyG Data (produces y_edge_u for undirected supervision)
        return build_graph_data_from_nx(
            G=G,
            template=self.template,
            fuel_classes=self.fuel_classes,
            use_true_edges_for_encoder=self.use_true_edges_for_encoder,
            alt_key=self.alt_key,
            slope_key=self.slope_key,
            fuel_key=self.fuel_key,
            slot_key=self.slot_key,
            y_ign_idx = ignition[0],
        )
    
def sigmoid(x):
    return 1 / (1 + torch.exp(-x)) if torch.is_tensor(x) else 1 / (1 + np.exp(-x))


@torch.no_grad()
def evaluate_all(model, val_loader, template, device, edge_thr: float = 0.5):
    """
    Evaluate UNDIRECTED edge metrics + ignition top-1 accuracy over val_loader.

    Expects:
      - template.E_u
      - batch.y_edge_u: [E_u] or [B*E_u] labels {0,1}
      - batch.y_ign_idx: [B] (single ignition node index per graph)
      - model(...) returns either:
          * (logits_u[B,E_u], ign_logits[B,N]), or
          * {"edge_u_logits": [B,E_u], "ignition_logits"/"ign_logits": [B,N]}
    """
    model.eval()
    E_u = template.E_u

    # Edge accumulators
    edge_correct = 0
    edge_total   = 0
    edge_tp = edge_fp = edge_fn = 0

    # Ignition accumulators
    ign_correct = 0
    ign_total   = 0

    for batch in val_loader:
        batch = batch.to(device)
        output, mu, logvar = model(batch.x, batch.edge_index_enc, batch.batch)

        # ---- extract logits ----
        if isinstance(output, (tuple, list)):
            assert len(output) >= 2, "Model must return (logits_u, ign_logits)."
            logits_u, ign_logits = output[0], output[1]
        elif isinstance(output, dict):
            logits_u   = output.get("edge_u_logits", output.get("edges_u", None))
            ign_logits = output.get("ignition_logits", output.get("ign_logits", None))
            if logits_u is None or ign_logits is None:
                raise ValueError("Output dict must include 'edge_u_logits' and 'ignition_logits' (or 'ign_logits').")
        else:
            raise ValueError("Model must return (logits_u, ign_logits) or a dict with those keys.")

        if logits_u.dim() == 1:   # single example fallback
            logits_u = logits_u.unsqueeze(0)
        if ign_logits.dim() == 1:
            ign_logits = ign_logits.unsqueeze(0)

        B = logits_u.size(0)
        assert logits_u.size(1) == E_u, f"Expected logits_u [B,{E_u}], got {tuple(logits_u.shape)}"

        # labels
        y_edge_u = batch.y_edge_u.view(B, E_u).to(device).float()
        y_ign    = batch.y_ign_idx.view(B).to(device).long()

        # ---- edges ----
        probs_u = torch.sigmoid(logits_u)
        pred_u  = (probs_u >= edge_thr).float()

        edge_correct += (pred_u == y_edge_u).sum().item()
        edge_total   += y_edge_u.numel()

        lk = y_edge_u.long()
        ep = pred_u.long()
        edge_tp += ((ep == 1) & (lk == 1)).sum().item()
        edge_fp += ((ep == 1) & (lk == 0)).sum().item()
        edge_fn += ((ep == 0) & (lk == 1)).sum().item()

        # ---- ignition (top-1 only) ----
        ign_pred = ign_logits.argmax(dim=1)       # [B]
        ign_correct += (ign_pred == y_ign).sum().item()
        ign_total   += B

    # aggregate
    eps = 1e-12
    edge_acc = edge_correct / max(1, edge_total)
    edge_prec = edge_tp / max(1, (edge_tp + edge_fp))
    edge_rec  = edge_tp / max(1, (edge_tp + edge_fn))
    edge_f1   = (2 * edge_prec * edge_rec) / max(eps, (edge_prec + edge_rec))
    ign_acc   = ign_correct / max(1, ign_total)

    return {
        # edges
        "edge_acc": edge_acc,
        "edge_precision": edge_prec,
        "edge_recall": edge_rec,
        "edge_f1": edge_f1,
        # ignition
        "ign_acc": ign_acc,
        "ign_support": ign_total,
    }

@torch.no_grad()
def metrics_on_batch_tm(model, batch, template, device, edge_thr: float = 0.5):
    """
    UNDIRECTED edge + ignition (top-1) metrics on the CURRENT batch.

    Expects:
      - template.E_u
      - batch.y_edge_u: [B*E_u] or [E_u]
      - batch.y_ign_idx: [B]
      - model(...) returns (logits_u[B,E_u], ign_logits[B,N]) or a dict with those.
    """
    model.eval()
    batch = batch.to(device)
    E_u = template.E_u

    output, mu, logvar = model(batch.x, batch.edge_index_enc, batch.batch)

    # ---- extract logits ----
    if isinstance(output, (tuple, list)):
        assert len(output) >= 2, "Model must return (logits_u, ign_logits)."
        logits_u, ign_logits = output[0], output[1]
    elif isinstance(output, dict):
        logits_u   = output.get("edge_u_logits", output.get("edges_u", None))
        ign_logits = output.get("ignition_logits", output.get("ign_logits", None))
        if logits_u is None or ign_logits is None:
            raise ValueError("Output dict must include 'edge_u_logits' and 'ignition_logits' (or 'ign_logits').")
    else:
        raise ValueError("Model must return (logits_u, ign_logits) or a dict with those keys.")

    if logits_u.dim() == 1:
        logits_u = logits_u.unsqueeze(0)
    if ign_logits.dim() == 1:
        ign_logits = ign_logits.unsqueeze(0)

    B = logits_u.size(0)
    assert logits_u.size(1) == E_u, f"Expected logits_u [B,{E_u}], got {tuple(logits_u.shape)}"

    # labels
    y_edge_u = batch.y_edge_u.view(B, E_u).to(device).float()
    y_ign    = batch.y_ign_idx.view(B).to(device).long()

    # ---- edges ----
    p  = torch.sigmoid(logits_u)
    yhat = (p >= edge_thr).float()

    total   = y_edge_u.numel()
    correct = (yhat == y_edge_u).sum().item()
    tp = ((yhat == 1) & (y_edge_u == 1)).sum().item()
    fp = ((yhat == 1) & (y_edge_u == 0)).sum().item()
    fn = ((yhat == 0) & (y_edge_u == 1)).sum().item()

    eps = 1e-12
    acc  = correct / max(1, total)
    prec = tp / max(1, (tp + fp))
    rec  = tp / max(1, (tp + fn))
    f1   = (2 * prec * rec) / max(eps, (prec + rec))

    # ---- ignition (top-1 only) ----
    ign_pred = ign_logits.argmax(dim=1)     # [B]
    ign_acc  = (ign_pred == y_ign).float().mean().item()

    return {
        # edges
        "edge/acc":  float(acc),
        "edge/prec": float(prec),
        "edge/rec":  float(rec),
        "edge/f1":   float(f1),
        # ignition
        "ign/acc": float(ign_acc),
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
