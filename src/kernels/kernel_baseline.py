#!/usr/bin/env python3
"""
Directed WL (Weisfeiler–Lehman) feature-hashing + k-medoids baseline for scenario selection.

- Computes WL label-count features for each directed acyclic graph (DAG).
- Hashes label counts into a fixed-dimensional sparse vector (feature hashing).
- L2-normalizes features and clusters with k-medoids using cosine distance.
- For large N, uses a CLARA-style approximation plus a lightweight full-data refinement step.

Outputs (per k):
  outputs/selected_graphs_k{k}.json
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




EPS = 1e-12


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
# WL feature hashing
# -----------------------------
def initial_labels(G: nx.DiGraph, node_label_attr: Optional[str]) -> Dict[object, str]:
    """
    Initial node labels:
      - If node_label_attr is provided and present on node, use it.
      - Else use simple structural label in{deg}_out{deg}.
    """
    labels: Dict[object, str] = {}
    for v in G.nodes():
        if node_label_attr is not None and node_label_attr in G.nodes[v]:
            labels[v] = str(G.nodes[v][node_label_attr])
        else:
            labels[v] = f"in{G.in_degree(v)}_out{G.out_degree(v)}"
    return labels


class DirectedWLKernelHasher:
    """
    Directed WL subtree feature hashing.

    Produces a sparse feature matrix X (N x hash_dim) where features are hashed counts of
    node labels across WL iterations t=0..h.

    Similarity is computed as cosine similarity between row vectors:
      K = Xn * Xn^T where Xn is row-normalized (L2).
    """

    def __init__(self, h: int = 3, hash_dim: int = 2**18, node_label_attr: Optional[str] = None):
        self.h = int(h)
        self.hash_dim = int(hash_dim)
        self.node_label_attr = node_label_attr

    def _wl_features_one_graph(self, G: nx.DiGraph) -> Dict[int, float]:
        labels = initial_labels(G, self.node_label_attr)
        feat_counts: Dict[int, float] = {}

        # Iteration 0 label counts
        for lab in labels.values():
            col = stable_bucket(f"0|{lab}", self.hash_dim)
            feat_counts[col] = feat_counts.get(col, 0.0) + 1.0

        # WL refinement iterations (directed: IN + OUT neighborhoods)
        for t in range(1, self.h + 1):
            new_labels: Dict[object, str] = {}
            for v in G.nodes():
                in_labs = sorted((labels[u] for u in G.predecessors(v)), key=str)
                out_labs = sorted((labels[u] for u in G.successors(v)), key=str)
                sig = f"{labels[v]}|IN:{','.join(in_labs)}|OUT:{','.join(out_labs)}"
                new_lab = stable_md5(sig)
                new_labels[v] = new_lab

                col = stable_bucket(f"{t}|{new_lab}", self.hash_dim)
                feat_counts[col] = feat_counts.get(col, 0.0) + 1.0

            labels = new_labels

        return feat_counts

    def transform_files(self, graph_files: Sequence[str]) -> sparse.csr_matrix:
        """
        Streaming transform: loads each graph, computes features, and discards the graph.
        Avoids holding all graphs in memory.
        """
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        for i, p in enumerate(tqdm(graph_files, desc="WL features (directed)")):
            G = load_nx_dag(p)
            fd = self._wl_features_one_graph(G)
            for c, v in fd.items():
                rows.append(i)
                cols.append(int(c))
                data.append(float(v))

        X = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(graph_files), self.hash_dim),
            dtype=np.float64,
        )
        return X


def row_l2_norms(X: sparse.csr_matrix) -> np.ndarray:
    return np.sqrt(X.multiply(X).sum(axis=1)).A1


def row_normalize(X: sparse.csr_matrix, norms: np.ndarray) -> sparse.csr_matrix:
    norms = norms.astype(np.float64, copy=False)
    inv = np.zeros_like(norms, dtype=np.float64)
    inv[norms > EPS] = 1.0 / norms[norms > EPS]
    return sparse.diags(inv, format="csr") @ X


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
    ap.add_argument("--out_dir", type=str, default="src/kernels/outputs_final", help="Output directory")

    ap.add_argument("--k_values", type=int, nargs="+", default=[20, 100], help="K values (e.g., 20 100)")
    ap.add_argument("--wl_iterations", type=int, default=3, help="WL iterations h")
    ap.add_argument("--hash_dim", type=int, default=2**18, help="Feature hashing dimension")
    ap.add_argument("--node_label_attr", type=str, default=None, help="Optional node attribute for initial labels")

    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--max_full_n", type=int, default=1200, help="If N <= this, compute full NxN; else CLARA")

    ap.add_argument("--subset_size", type=int, default=3000, help="CLARA subset size (auto-raised based on k, N)")
    ap.add_argument("--clara_trials", type=int, default=20, help="CLARA trials")
    ap.add_argument("--max_iter", type=int, default=50, help="k-medoids iterations (subset)")
    ap.add_argument("--n_init", type=int, default=5, help="K-medoids restarts per trial")
    ap.add_argument("--polish_rounds", type=int, default=3, help="Full-data polishing rounds (CLARA only)")
    ap.add_argument("--polish_candidates", type=int, default=80, help="Candidates per cluster in polishing")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit for quick tests")
    args = ap.parse_args()

    rng = np.random.default_rng(int(args.seed))

    graphs_dir = Path(args.graphs_dir)
    out_dir = Path(args.out_dir)
  
    

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
    )
    X = hasher.transform_files(files)

    norms = row_l2_norms(X)
    Xn = row_normalize(X, norms)
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
        # ---------- Embedding plots (PCA / UMAP / t-SNE) ----------


        labels = np.array(res.cluster_labels, dtype=int)
        medoids = np.array(res.medoid_indices, dtype=int)

        # Build N×k embedding: distance-to-medoids (cheap and works great for large N)
        S = cosine_similarity_to_medoids(Xn, medoids)      # (N, k)
        E = 1.0 - S                                       # (N, k)  "distance-like"

        # save matrices X,S,E
        # np.save(out_dir / f"embed_k{k}_features.npy", Xn.toarray())
        # np.save(out_dir / f"embed_k{k}_similarities.npy", S)

        np.save(out_dir / f"embed_k{k}_distances.npy", E)
        

        # ---- PCA (all points) ----
        Zp = PCA(n_components=2, random_state=int(args.seed)).fit_transform(E)
        plt.figure(figsize=(12, 9))
        plt.scatter(Zp[:, 0], Zp[:, 1], s=6, c=labels, cmap="tab20", alpha=0.55)
        plt.scatter(Zp[medoids, 0], Zp[medoids, 1], s=120, c="black", marker="x")
        plt.title(f"PCA of (1 - cosine similarity to medoids), k={k}")
        plt.tight_layout()
        plt.savefig(out_dir / f"embed_k{k}_pca.png", dpi=220)
        plt.close()

        # ---- UMAP (all points) ----
        Zu = UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.10,
            metric="euclidean",           # because E is already a distance-like feature space
            random_state=int(args.seed),
        ).fit_transform(E)

        plt.figure(figsize=(12, 9))
        plt.scatter(Zu[:, 0], Zu[:, 1], s=6, c=labels, cmap="tab20", alpha=0.55)
        plt.scatter(Zu[medoids, 0], Zu[medoids, 1], s=120, c="black", marker="x")
        plt.title(f"UMAP of (1 - cosine similarity to medoids), k={k}")
        plt.tight_layout()
        plt.savefig(out_dir / f"embed_k{k}_umap.png", dpi=220)
        plt.close()

        # ---- t-SNE (SUBSAMPLE only; t-SNE is expensive) ----
        n_plot = min(N, 8000)
        idx = rng.choice(np.arange(N), size=n_plot, replace=False)
        Et = E[idx]

        Zt = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=30,
            random_state=int(args.seed),
        ).fit_transform(Et)

        plt.figure(figsize=(12, 9))
        plt.scatter(Zt[:, 0], Zt[:, 1], s=7, c=labels[idx], cmap="tab20", alpha=0.65)

        # mark medoids if they appear in the subsample
        pos = {int(i): j for j, i in enumerate(idx.tolist())}
        mid_in = [pos[int(m)] for m in medoids.tolist() if int(m) in pos]
        if len(mid_in) > 0:
            plt.scatter(Zt[mid_in, 0], Zt[mid_in, 1], s=140, c="black", marker="x")

        plt.title(f"t-SNE of (1 - cosine similarity to medoids) [n={n_plot}], k={k}")
        plt.tight_layout()
        plt.savefig(out_dir / f"tsne_{k}.png", dpi=220)
        plt.close()

        print(f"[info] Saved plots: embed_k{k}_pca.png, embed_k{k}_umap.png, embed_k{k}_tsne.png")
    # ---------- end plots ----------


        selected_files = [os.path.basename(files[i]) for i in res.medoid_indices]
        payload = {
            "method": "directed_wl_feature_hashing_kmedoids",
            "k": int(k),
            "wl_iterations": int(args.wl_iterations),
            "hash_dim": int(args.hash_dim),
            "node_label_attr": args.node_label_attr,
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

# how to run:
# python src/kernels/kernel_baseline.py --graphs_dir data/sub20/graph