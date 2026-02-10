#!/usr/bin/env python3
"""
Directed WL (Weisfeiler–Lehman) feature-hashing + k-medoids baseline for scenario selection.

Computes WL label-count features for each DAG using feature hashing and L2 normalization, then selects k representative
graphs via k-medoids clustering with cosine distance.
For large N, uses a CLARA-style approximation plus a lightweight full-data refinement step.

"""

from __future__ import annotations
import argparse
import glob
import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import networkx as nx
import numpy as np
from scipy import sparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE 
from sklearn.preprocessing import normalize
from umap import UMAP


# -----------------------------
# Stable hashing utilities
# -----------------------------
def stable_md5(s: str) -> str:
    """Deterministic hash string (constant-length label)."""
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def stable_bucket(s: str, mod: int) -> int:
    """Deterministic bucket id for feature hashing."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % mod


# -----------------------------
# IO helpers
# -----------------------------
def find_repo_root(start: Optional[Path] = None) -> Path:
    """
    Heuristic: walk up parents until a 'graphs' folder is found.
    Falls back to a reasonable parent.
    """
    if start is None:
        start = Path(__file__).resolve()
    start = start.resolve()
    for parent in [start] + list(start.parents):
        if (parent / "graphs").exists():
            return parent
    return start.parent


def discover_files(graphs_dir: str, pattern: str) -> List[str]:
    return sorted(glob.glob(os.path.join(graphs_dir, pattern)))


def load_nx_dag(path: str) -> nx.DiGraph:
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, nx.DiGraph):
        G = obj
    elif isinstance(obj, nx.MultiDiGraph):
        # Collapse parallel edges; baseline only needs adjacency.
        G = nx.DiGraph(obj)
    else:
        raise TypeError(f"{path}: expected nx.DiGraph (or nx.MultiDiGraph), got {type(obj)}")

    if not nx.is_directed_acyclic_graph(G):
        raise ValueError(f"{path}: graph is not a DAG")

    if G.number_of_nodes() == 0:
        # Avoid empty-feature edge cases.
        G = nx.DiGraph()
        G.add_node(0)

    return G


def save_json(out_path: str | Path, payload: dict) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# -----------------------------
# Edge-weight discretization
# -----------------------------
def discretize_weight(w: float, boundaries: Sequence[float]) -> int:
    """Assign a continuous weight to a discrete bin index using pre-computed boundaries."""
    for i, b in enumerate(boundaries):
        if w <= b:
            return i
    return len(boundaries)


def compute_weight_boundaries(G: nx.DiGraph, n_bins: int = 5, weight_attr: str = "weight") -> List[float]:
    """Compute quantile-based bin boundaries from all edge weights in a graph."""
    weights = [float(d.get(weight_attr, 0)) for _, _, d in G.edges(data=True)]
    if not weights:
        return []
    quantiles = np.linspace(0, 100, n_bins + 1)[1:-1]  # exclude 0% and 100%
    return list(np.percentile(weights, quantiles))


# -----------------------------
# WL feature hashing
# -----------------------------
def initial_labels(
    G: nx.DiGraph,
    node_label_attr: Optional[str],
    use_edge_weights: bool = False,
    weight_attr: str = "weight",
    weight_bins: int = 5,
) -> Dict[object, str]:
    """
    Initial node labels incorporating structural degree AND edge-weight information.

    When use_edge_weights=True, each node's label encodes:
      - in-degree and out-degree (topology)
      - discretized mean incoming edge weight (fire arrival speed)
      - discretized mean outgoing edge weight (fire spread speed)

    This makes the kernel sensitive to *how fast* fire arrives at and departs from each cell,
    not just the connectivity pattern.
    """
    labels: Dict[object, str] = {}

    if use_edge_weights:
        boundaries = compute_weight_boundaries(G, n_bins=weight_bins, weight_attr=weight_attr)

    for v in G.nodes():
        if node_label_attr is not None and node_label_attr in G.nodes[v]:
            labels[v] = str(G.nodes[v][node_label_attr])
        else:
            in_deg = G.in_degree(v)
            out_deg = G.out_degree(v)

            if use_edge_weights and boundaries:
                # Mean incoming weight (how fast fire reaches this cell)
                in_weights = [float(G.edges[u, v].get(weight_attr, 0)) for u in G.predecessors(v)]
                in_wbin = discretize_weight(np.mean(in_weights), boundaries) if in_weights else 0

                # Mean outgoing weight (how fast fire leaves this cell)
                out_weights = [float(G.edges[v, u].get(weight_attr, 0)) for u in G.successors(v)]
                out_wbin = discretize_weight(np.mean(out_weights), boundaries) if out_weights else 0

                labels[v] = f"in{in_deg}_out{out_deg}_wi{in_wbin}_wo{out_wbin}"
            else:
                labels[v] = f"in{in_deg}_out{out_deg}"

    return labels


class DirectedWLKernelHasher:
    """
    Attributed Directed WL subtree feature hashing.

    Produces a sparse feature matrix X (N x hash_dim) where features are hashed counts of
    node labels across WL iterations t=0..h.

    When use_edge_weights=True, the kernel incorporates discretized edge weights (fire spread
    times) into both the initial node labels and the WL refinement signatures. This makes the
    kernel sensitive to fire propagation dynamics, not just graph topology.

    Similarity is computed as cosine similarity between row vectors:
      K = Xn * Xn^T where Xn is row-normalized (L2).
    """

    def __init__(
        self,
        h: int = 3,
        hash_dim: int = 2**18,
        node_label_attr: Optional[str] = None,
        use_edge_weights: bool = False,
        weight_attr: str = "weight",
        weight_bins: int = 5,
    ):
        self.h = int(h)
        self.hash_dim = int(hash_dim)
        self.node_label_attr = node_label_attr
        self.use_edge_weights = use_edge_weights
        self.weight_attr = weight_attr
        self.weight_bins = weight_bins

    def _wl_features_one_graph(self, G: nx.DiGraph) -> Dict[int, float]:
        labels = initial_labels(
            G,
            self.node_label_attr,
            use_edge_weights=self.use_edge_weights,
            weight_attr=self.weight_attr,
            weight_bins=self.weight_bins,
        )
        feat_counts: Dict[int, float] = {}

        # Per-graph weight boundaries (used for edge annotations in refinement)
        boundaries: List[float] = []
        if self.use_edge_weights:
            boundaries = compute_weight_boundaries(G, n_bins=self.weight_bins, weight_attr=self.weight_attr)

        # Iteration 0 label counts
        for lab in labels.values():
            col = stable_bucket(f"0|{lab}", self.hash_dim)
            feat_counts[col] = feat_counts.get(col, 0.0) + 1.0

        # WL refinement iterations (directed: IN + OUT neighborhoods)
        for t in range(1, self.h + 1):
            new_labels: Dict[object, str] = {}
            for v in G.nodes():
                if self.use_edge_weights and boundaries:
                    # Weight-annotated neighbor labels: "label:wbin"
                    in_labs = sorted(
                        (f"{labels[u]}:{discretize_weight(float(G.edges[u, v].get(self.weight_attr, 0)), boundaries)}"
                         for u in G.predecessors(v)),
                        key=str,
                    )
                    out_labs = sorted(
                        (f"{labels[u]}:{discretize_weight(float(G.edges[v, u].get(self.weight_attr, 0)), boundaries)}"
                         for u in G.successors(v)),
                        key=str,
                    )
                else:
                    in_labs = sorted((labels[u] for u in G.predecessors(v)), key=str)
                    out_labs = sorted((labels[u] for u in G.successors(v)), key=str)

                sig = f"{labels[v]}|IN:{','.join(in_labs)}|OUT:{','.join(out_labs)}"
                new_lab = stable_md5(sig)
                new_labels[v] = new_lab

                col = stable_bucket(f"{t}|{new_lab}", self.hash_dim)
                feat_counts[col] = feat_counts.get(col, 0.0) + 1.0

            labels = new_labels

        return feat_counts

    def transform_files(self, graph_files: Sequence[str], chunk_size: int = 5000) -> sparse.csr_matrix:
        """
        Chunked streaming transform: processes graphs in chunks to limit memory.
        Each chunk builds a partial sparse matrix; chunks are vstack-ed at the end.
        """
        n_files = len(graph_files)
        chunks: List[sparse.csr_matrix] = []

        for start in range(0, n_files, chunk_size):
            end = min(start + chunk_size, n_files)
            batch = graph_files[start:end]
            rows: List[int] = []
            cols: List[int] = []
            data: List[float] = []

            for i, p in enumerate(tqdm(batch, desc=f"WL features [{start}-{end}]/{n_files}")):
                G = load_nx_dag(p)
                fd = self._wl_features_one_graph(G)
                for c, v in fd.items():
                    rows.append(i)
                    cols.append(int(c))
                    data.append(float(v))

            chunk = sparse.csr_matrix(
                (data, (rows, cols)),
                shape=(len(batch), self.hash_dim),
                dtype=np.float32,
            )
            chunks.append(chunk)
            del rows, cols, data

        X = sparse.vstack(chunks, format="csr")
        del chunks
        return X


# -----------------------------
# Cosine similarity/distances
# -----------------------------
def cosine_similarity_matrix(Xn: sparse.csr_matrix) -> np.ndarray:
    # Dense NxN (use only when N is moderate)
    K = (Xn @ Xn.T).toarray()
    K = np.clip(K, -1.0, 1.0)
    # Ensure exact self-similarity (helps numerical stability)
    np.fill_diagonal(K, 1.0)
    return K


def cosine_similarity_to_medoids(Xn: sparse.csr_matrix, medoids: Sequence[int]) -> np.ndarray:
    Xm = Xn[list(medoids)]
    S = (Xn @ Xm.T).toarray()
    return np.clip(S, -1.0, 1.0)


def cosine_distance_from_similarity(S: np.ndarray) -> np.ndarray:
    # For unit-norm vectors: d^2 = 2 - 2*cos
    d2 = 2.0 - 2.0 * S
    return np.sqrt(np.maximum(d2, 0.0))


def assign_labels_and_inertia(Xn: sparse.csr_matrix, medoids: Sequence[int]) -> Tuple[np.ndarray, float]:
    S = cosine_similarity_to_medoids(Xn, medoids)      # (N,k)
    D = cosine_distance_from_similarity(S)             # (N,k)
    labels = D.argmin(axis=1)
    inertia = float(D.min(axis=1).sum())
    return labels.astype(int), inertia


# -----------------------------
# k-medoids (k-medoids++ init + Lloyd updates)
# -----------------------------
def kmedoidspp_init(D: np.ndarray, k: int, rng: np.random.Generator) -> List[int]:
    n = D.shape[0]
    first = int(rng.integers(0, n))
    medoids = [first]
    if k == 1:
        return medoids

    dmin = D[:, first].copy()

    while len(medoids) < k:
        probs = dmin**2
        s = float(probs.sum())
        if s <= 0.0:
            candidates = [i for i in range(n) if i not in medoids]
            medoids.append(int(rng.choice(candidates)))
        else:
            probs = probs / s
            cand = int(rng.choice(np.arange(n), p=probs))
            if cand in medoids:
                # resample a few times, then fall back to any unused
                for _ in range(10):
                    cand = int(rng.choice(np.arange(n), p=probs))
                    if cand not in medoids:
                        break
                if cand in medoids:
                    candidates = [i for i in range(n) if i not in medoids]
                    cand = int(rng.choice(candidates))
            medoids.append(cand)

        dmin = np.minimum(dmin, D[:, medoids[-1]])

    return medoids


def k_medoids_lloyd(
    D: np.ndarray,
    k: int,
    rng: np.random.Generator,
    max_iter: int = 50,
) -> Tuple[List[int], np.ndarray, float]:
    """
    Simple k-medoids: assign -> recompute medoid within each cluster.

    Note: Not full PAM swap (kept intentionally simple and fast). Selection quality is
    improved for large-N via a full-data medoid refinement step outside this routine.
    """
    n = D.shape[0]
    if not (1 <= k <= n):
        raise ValueError(f"Invalid k={k} for n={n}")

    medoids = kmedoidspp_init(D, k, rng)

    for _ in range(max_iter):
        dist_to_m = D[:, medoids]  # (n,k)
        labels = dist_to_m.argmin(axis=1)

        new_medoids = medoids.copy()
        changed = False

        for c in range(k):
            idx = np.where(labels == c)[0]
            if idx.size == 0:
                # pick farthest point from its nearest medoid (reduces empty clusters)
                nearest = dist_to_m.min(axis=1)
                for cand in np.argsort(-nearest):
                    cand = int(cand)
                    if cand not in new_medoids:
                        new_medoids[c] = cand
                        changed = True
                        break
                continue

            Dc = D[np.ix_(idx, idx)]
            sums = Dc.sum(axis=1)
            best = int(idx[int(np.argmin(sums))])
            if best != new_medoids[c]:
                new_medoids[c] = best
                changed = True

        if not changed:
            break
        medoids = new_medoids

    dist_to_m = D[:, medoids]
    labels = dist_to_m.argmin(axis=1)
    inertia = float(dist_to_m.min(axis=1).sum())
    return [int(m) for m in medoids], labels.astype(int), inertia


def k_medoids_multi_init(
    D: np.ndarray,
    k: int,
    rng: np.random.Generator,
    max_iter: int,
    n_init: int,
) -> Tuple[List[int], np.ndarray, float]:
    best_medoids: Optional[List[int]] = None
    best_labels: Optional[np.ndarray] = None
    best_inertia = float("inf")

    for _ in range(int(max(1, n_init))):
        medoids, labels, inertia = k_medoids_lloyd(D, k, rng, max_iter=max_iter)
        if inertia < best_inertia:
            best_inertia = float(inertia)
            best_medoids = medoids
            best_labels = labels

    assert best_medoids is not None and best_labels is not None
    return best_medoids, best_labels, float(best_inertia)


# -----------------------------
# Large-N refinement (improves selection quality)
# -----------------------------
def refine_medoids_on_full_data(
    Xn: sparse.csr_matrix,
    medoids: List[int],
    labels: np.ndarray,
    rng: np.random.Generator,
    candidates_per_cluster: int = 40,
) -> List[int]:
    """
    Improves medoids without building full NxN:
    For each cluster, evaluate a small candidate set (sample of cluster points + current medoid)
    and pick the candidate minimizing sum of cosine distances to all points in the cluster.
    """
    k = len(medoids)
    new_medoids: List[int] = []
    used = set()

    for c in range(k):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            # fallback: pick any unused point
            cand = int(rng.integers(0, Xn.shape[0]))
            while cand in used:
                cand = int(rng.integers(0, Xn.shape[0]))
            new_medoids.append(cand)
            used.add(cand)
            continue

        m = min(int(idx.size), int(max(candidates_per_cluster, np.sqrt(idx.size))))
        cand = rng.choice(idx, size=m, replace=False).astype(int)

        cur = int(medoids[c])
        if cur not in cand:
            cand = np.append(cand, cur)

        # Compute distances from all cluster points to candidate points
        S = (Xn[idx] @ Xn[cand].T).toarray()     # (|idx|, |cand|)
        D = cosine_distance_from_similarity(S)  # (|idx|, |cand|)
        sums = D.sum(axis=0)                    # (|cand|,)

        best = int(cand[int(np.argmin(sums))])

        # Ensure uniqueness (rare but possible due to empty-cluster fallbacks)
        if best in used:
            # choose best unused candidate
            order = np.argsort(sums)
            for j in order:
                b2 = int(cand[int(j)])
                if b2 not in used:
                    best = b2
                    break

        new_medoids.append(best)
        used.add(best)

    return new_medoids


def polish_medoids(
    Xn: sparse.csr_matrix,
    medoids: List[int],
    rng: np.random.Generator,
    n_rounds: int = 2,
    candidates_per_cluster: int = 40,
) -> Tuple[List[int], np.ndarray, float]:
    """
    Alternate:
      assign labels on full data -> refine medoids within clusters (approx) -> repeat
    """
    labels, inertia = assign_labels_and_inertia(Xn, medoids)
    for _ in range(n_rounds):
        medoids = refine_medoids_on_full_data(
            Xn, medoids=medoids, labels=labels, rng=rng, candidates_per_cluster=candidates_per_cluster
        )
        labels, inertia = assign_labels_and_inertia(Xn, medoids)
    return medoids, labels, inertia


# -----------------------------
# Selection APIs
# -----------------------------
@dataclass
class SelectionResult:
    k: int
    medoid_indices: List[int]
    cluster_labels: List[int]
    cluster_sizes: List[int]
    cluster_weights: List[float]
    inertia: float


def full_kmedoids_from_Xn(
    Xn: sparse.csr_matrix,
    k: int,
    rng: np.random.Generator,
    max_iter: int,
    n_init: int = 1,
) -> SelectionResult:
    K = cosine_similarity_matrix(Xn)
    D = cosine_distance_from_similarity(K)
    medoids, labels, inertia = k_medoids_multi_init(D, k, rng, max_iter=max_iter, n_init=n_init)

    n = len(labels)
    sizes = [int(np.sum(labels == c)) for c in range(k)]
    weights = [s / float(n) for s in sizes]
    return SelectionResult(
        k=k,
        medoid_indices=[int(m) for m in medoids],
        cluster_labels=labels.astype(int).tolist(),
        cluster_sizes=sizes,
        cluster_weights=weights,
        inertia=float(inertia),
    )


def clara_kmedoids_from_Xn(
    Xn: sparse.csr_matrix,
    k: int,
    rng: np.random.Generator,
    subset_size: int,
    n_trials: int,
    max_iter: int,
    n_init: int = 1,
    polish_rounds: int = 2,
    polish_candidates: int = 40,
) -> SelectionResult:
    """
    CLARA-style approximate k-medoids without building full NxN.
      - sample subset
      - cluster subset (full distances on subset)
      - lift medoids to global indices
      - evaluate on full set by distances-to-medoids only
      - keep best trial
      - polish medoids on full data (improves selection quality)
    """
    n = Xn.shape[0]
    # CLARA quality drops sharply if subset is too small relative to k
    subset_floor = max(int(subset_size), int(10 * k), int(0.02 * n), 1000)
    subset_size = int(min(n, max(subset_floor, k + 1)))

    best_medoids: Optional[List[int]] = None
    best_inertia = float("inf")
    best_labels: Optional[np.ndarray] = None

    for _ in range(int(n_trials)):
        subset = rng.choice(np.arange(n), size=subset_size, replace=False)
        subset = np.array(sorted(map(int, subset.tolist())), dtype=int)

        Xs = Xn[subset]
        Ks = cosine_similarity_matrix(Xs)
        Ds = cosine_distance_from_similarity(Ks)

        med_sub, _, _ = k_medoids_multi_init(Ds, k, rng, max_iter=max_iter, n_init=n_init)
        med_global = subset[med_sub].tolist()

        labels, inertia = assign_labels_and_inertia(Xn, med_global)

        if inertia < best_inertia:
            best_inertia = inertia
            best_medoids = med_global
            best_labels = labels

    assert best_medoids is not None and best_labels is not None

    # Full-data polishing: improves medoid quality without full NxN
    medoids2, labels2, inertia2 = polish_medoids(
        Xn,
        medoids=best_medoids,
        rng=rng,
        n_rounds=int(polish_rounds),
        candidates_per_cluster=int(polish_candidates),
    )

    labels = labels2
    inertia = float(inertia2)

    sizes = [int(np.sum(labels == c)) for c in range(k)]
    weights = [s / float(n) for s in sizes]

    return SelectionResult(
        k=k,
        medoid_indices=[int(m) for m in medoids2],
        cluster_labels=labels.astype(int).tolist(),
        cluster_sizes=sizes,
        cluster_weights=weights,
        inertia=float(inertia),
    )


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Directed WL feature hashing + k-medoids scenario selection")
    ap.add_argument("--graphs_dir", type=str, default="data/sub20/graphs", help="Directory containing graph pickle files")
    ap.add_argument("--pattern", type=str, default="graph_*.pickle", help="Glob pattern inside graphs_dir")
    ap.add_argument("--out_dir", type=str, default="src/kernels/outputs", help="Output directory")

    ap.add_argument("--k_values", type=int, nargs="+", default=[20, 100], help="K values (e.g., 20 100)")
    ap.add_argument("--wl_iterations", type=int, default=3, help="WL iterations h")
    ap.add_argument("--hash_dim", type=int, default=2**18, help="Feature hashing dimension")
    ap.add_argument("--node_label_attr", type=str, default=None, help="Optional node attribute for initial labels")
    ap.add_argument("--use_edge_weights", action="store_true", default=True,
                    help="Incorporate edge weights into WL labels (default: True)")
    ap.add_argument("--no_edge_weights", dest="use_edge_weights", action="store_false",
                    help="Disable edge-weight attribution (topology-only WL)")
    ap.add_argument("--weight_attr", type=str, default="weight", help="Edge attribute for weights")
    ap.add_argument("--weight_bins", type=int, default=5, help="Number of quantile bins for weight discretization")

    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--max_full_n", type=int, default=1200, help="If N <= this, compute full NxN; else CLARA")

    ap.add_argument("--subset_size", type=int, default=3000, help="CLARA subset size (auto-raised based on k, N)")
    ap.add_argument("--clara_trials", type=int, default=20, help="CLARA trials")
    ap.add_argument("--max_iter", type=int, default=50, help="k-medoids iterations (subset)")
    ap.add_argument("--n_init", type=int, default=5, help="K-medoids restarts per trial")
    ap.add_argument("--polish_rounds", type=int, default=3, help="Full-data polishing rounds (CLARA only)")
    ap.add_argument("--polish_candidates", type=int, default=80, help="Candidates per cluster in polishing")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit for quick tests")
    ap.add_argument("--no_plots", action="store_true", help="Skip PCA/UMAP/t-SNE plots (faster for large N)")
    args = ap.parse_args()

    rng = np.random.default_rng(int(args.seed))

    graphs_dir = Path(args.graphs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = discover_files(str(graphs_dir), args.pattern)
    if args.limit is not None:
        files = files[: int(args.limit)]
    if len(files) == 0:
        raise RuntimeError("No pickle files found. Check --graphs_dir and --pattern.")

    print(f"[info] Found {len(files)} files under {graphs_dir}")

    # WL features (streaming) -> sparse matrix
    hasher = DirectedWLKernelHasher(
        h=int(args.wl_iterations),
        hash_dim=int(args.hash_dim),
        node_label_attr=args.node_label_attr,
        use_edge_weights=args.use_edge_weights,
        weight_attr=args.weight_attr,
        weight_bins=int(args.weight_bins),
    )
    print(f"[info] Edge-weight attribution: {'ENABLED (bins={})'.format(args.weight_bins) if args.use_edge_weights else 'DISABLED'}")
    X = hasher.transform_files(files)

    # sklearn handles zero rows by leaving them as all zeros (no epsilon needed).
    Xn = normalize(X, norm="l2", axis=1, copy=False)
    N = int(Xn.shape[0])
    print(f"[info] Feature matrix: {X.shape} nnz={X.nnz}")
    print(f"[info] N={N} | mode={'FULL' if N <= int(args.max_full_n) else 'CLARA'}")


    for k in [int(v) for v in args.k_values]:

        if N <= int(args.max_full_n):
            res = full_kmedoids_from_Xn(
                Xn,
                k=k,
                rng=rng,
                max_iter=int(args.max_iter),
                n_init=int(args.n_init),
            )
        else:
            res = clara_kmedoids_from_Xn(
                Xn,
                k=k,
                rng=rng,
                subset_size=int(args.subset_size),
                n_trials=int(args.clara_trials),
                max_iter=int(args.max_iter),
                n_init=int(args.n_init),
                polish_rounds=int(args.polish_rounds),
                polish_candidates=int(args.polish_candidates),
            )
        # ---------- Optional Embedding plots (PCA / UMAP / t-SNE) ----------
        if not args.no_plots:
            labels = np.array(res.cluster_labels, dtype=int)
            medoids = np.array(res.medoid_indices, dtype=int)

            S = cosine_similarity_to_medoids(Xn, medoids)
            E = 1.0 - S

            np.save(out_dir / f"embed_k{k}_distances.npy", E)

            Zp = PCA(n_components=2, random_state=int(args.seed)).fit_transform(E)
            plt.figure(figsize=(12, 9))
            plt.scatter(Zp[:, 0], Zp[:, 1], s=6, c=labels, cmap="tab20", alpha=0.55)
            plt.scatter(Zp[medoids, 0], Zp[medoids, 1], s=120, c="black", marker="x")
            plt.title(f"PCA of (1 - cosine similarity to medoids), k={k}")
            plt.tight_layout()
            plt.savefig(out_dir / f"embed_k{k}_pca.png", dpi=220)
            plt.close()

            Zu = UMAP(n_components=2, n_neighbors=30, min_dist=0.10, metric="euclidean",
                       random_state=int(args.seed)).fit_transform(E)
            plt.figure(figsize=(12, 9))
            plt.scatter(Zu[:, 0], Zu[:, 1], s=6, c=labels, cmap="tab20", alpha=0.55)
            plt.scatter(Zu[medoids, 0], Zu[medoids, 1], s=120, c="black", marker="x")
            plt.title(f"UMAP of (1 - cosine similarity to medoids), k={k}")
            plt.tight_layout()
            plt.savefig(out_dir / f"embed_k{k}_umap.png", dpi=220)
            plt.close()
            print(f"[info] Saved plots: embed_k{k}_pca.png, embed_k{k}_umap.png")

        selected_files = [os.path.basename(files[i]) for i in res.medoid_indices]
        payload = {
            "method": "attributed_directed_wl_kmedoids" if args.use_edge_weights else "directed_wl_kmedoids",
            "k": int(k),
            "wl_iterations": int(args.wl_iterations),
            "hash_dim": int(args.hash_dim),
            "node_label_attr": args.node_label_attr,
            "use_edge_weights": args.use_edge_weights,
            "weight_attr": args.weight_attr if args.use_edge_weights else None,
            "weight_bins": int(args.weight_bins) if args.use_edge_weights else None,
            "seed": int(args.seed),
            "mode": "FULL" if N <= int(args.max_full_n) else "CLARA",
            "selected_pickle_files": selected_files,
            "selected_indices": res.medoid_indices,
            "cluster_sizes": res.cluster_sizes,
            "cluster_weights": res.cluster_weights,
            "cluster_labels": res.cluster_labels,
            "inertia": float(res.inertia),
            "total_graphs": int(N),
            "pattern": args.pattern,
            "graphs_dir": str(graphs_dir),
            "polish_rounds": int(args.polish_rounds),
            "polish_candidates": int(args.polish_candidates),
        }

        out_path = out_dir / f"final_selected_graphs_k{k}.json"
        save_json(out_path, payload)

        print(
            f" k={k} -> {out_path} | inertia={res.inertia:.4f} | "
            f"sizes(min/mean/max)={min(res.cluster_sizes)}/{np.mean(res.cluster_sizes):.1f}/{max(res.cluster_sizes)}"
        )


if __name__ == "__main__":
    main()

# como ejecutar:
# python src/kernels/kernel_baseline.py --graphs_dir graphs
