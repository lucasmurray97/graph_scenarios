#!/usr/bin/env python3
"""
Directed WL Graph Kernel + k-Medoids Baseline for Scenario Selection (NetworkX DAG pickles).

"""

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


EPS = 1e-12


def stable_md5(s: str) -> str:
    """Deterministic hash string (constant-length label)."""
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def stable_bucket(s: str, mod: int) -> int:
    """Deterministic bucket id for feature hashing."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % mod


def find_repo_root(start: Optional[Path] = None) -> Path:
    if start is None:
        start = Path(__file__).resolve()
    start = start.resolve()
    for parent in [start] + list(start.parents):
        if (parent / "graphs").exists():
            return parent
    if len(start.parents) >= 2:
        return start.parents[2]
    return start.parent


def discover_files(graphs_dir: str, pattern: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(graphs_dir, pattern)))
    return paths


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
        G = nx.DiGraph()
        G.add_node(0)

    return G


def initial_labels(G: nx.DiGraph, node_label_attr: Optional[str]) -> Dict[object, str]:
    labels: Dict[object, str] = {}
    for v in G.nodes():
        if node_label_attr is not None and node_label_attr in G.nodes[v]:
            labels[v] = str(G.nodes[v][node_label_attr])
        else:
            labels[v] = f"in{G.in_degree(v)}_out{G.out_degree(v)}"
    return labels


class DirectedWLKernelHasher:
    """
    Directed WL subtree "feature hashing" kernel.

    Produces a sparse feature matrix X (N x hash_dim) where features are hashed counts of
    node labels across WL iterations t=0..h.

    Similarity is computed as cosine similarity between row vectors:
      K = Xn * Xn^T where Xn is row-normalized (L2).
    """

    def __init__(self, h: int = 3, hash_dim: int = 2**18, node_label_attr: Optional[str] = None):
        self.h = h
        self.hash_dim = hash_dim
        self.node_label_attr = node_label_attr

    def _wl_features_one_graph(self, G: nx.DiGraph) -> Dict[int, float]:
        # labels are always kept as compressed (fixed-length) strings after t>=1
        labels = initial_labels(G, self.node_label_attr)

        feat_counts: Dict[int, float] = {}

        # Count iteration 0 labels
        for v, lab in labels.items():
            col = stable_bucket(f"0|{lab}", self.hash_dim)
            feat_counts[col] = feat_counts.get(col, 0.0) + 1.0

        # WL refinement iterations
        # Use sorted predecessor/successor label lists to ensure determinism.
        # Carry forward only compressed labels.
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

    def fit_transform(self, graphs: List[nx.DiGraph]) -> sparse.csr_matrix:
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        for i in tqdm(range(len(graphs)), desc="WL features (directed)"):
            fd = self._wl_features_one_graph(graphs[i])
            for c, v in fd.items():
                rows.append(i)
                cols.append(c)
                data.append(v)

        X = sparse.csr_matrix((data, (rows, cols)), shape=(len(graphs), self.hash_dim), dtype=np.float64)
        return X


def row_l2_norms(X: sparse.csr_matrix) -> np.ndarray:
    # Efficient row norms for sparse CSR
    return np.sqrt(X.multiply(X).sum(axis=1)).A1


def row_normalize(X: sparse.csr_matrix, norms: np.ndarray) -> sparse.csr_matrix:
    inv = np.zeros_like(norms, dtype=np.float64)
    inv[norms > 0] = 1.0 / norms[norms > 0]
    return sparse.diags(inv, format="csr") @ X


def cosine_similarity_matrix(Xn: sparse.csr_matrix) -> np.ndarray:
    # Dense NxN (use only when N is moderate)
    K = (Xn @ Xn.T).toarray()
    # numerical clipping
    np.fill_diagonal(K, 1.0)
    return np.clip(K, -1.0, 1.0)


def cosine_similarity_to_medoids(Xn: sparse.csr_matrix, medoids: Sequence[int]) -> np.ndarray:
    # Returns dense (N x k) similarities to medoids
    Xm = Xn[list(medoids)]
    S = (Xn @ Xm.T).toarray()
    return np.clip(S, -1.0, 1.0)


def cosine_distance_from_similarity(S: np.ndarray) -> np.ndarray:
    # For unit-norm vectors: d^2 = 2 - 2*cos
    d2 = 2.0 - 2.0 * S
    return np.sqrt(np.maximum(d2, 0.0))


def distance_matrix_from_kernel(K: np.ndarray) -> np.ndarray:
    diag = np.diag(K)
    d2 = diag[:, None] + diag[None, :] - 2.0 * K
    return np.sqrt(np.maximum(d2, 0.0))


def kmedoidspp_init(D: np.ndarray, k: int, rng: np.random.Generator) -> List[int]:
    n = D.shape[0]
    first = int(rng.integers(0, n))
    medoids = [first]

    if k == 1:
        return medoids

    dmin = D[:, first].copy()

    while len(medoids) < k:
        probs = dmin**2
        s = probs.sum()
        if s <= 0:
            # everything identical; pick any not already chosen
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


def k_medoids_lloyd(D: np.ndarray, k: int, rng: np.random.Generator, max_iter: int = 50) -> Tuple[List[int], np.ndarray, float]:
    """
    Simple k-medoids: assign -> recompute medoid within each cluster.
    (Not full PAM swap; kept intentionally simple and readable.)
    """
    n = D.shape[0]
    if not (1 <= k <= n):
        raise ValueError(f"Invalid k={k} for n={n}")

    medoids = kmedoidspp_init(D, k, rng)

    for _ in range(max_iter):
        dist_to_m = D[:, medoids]                 # (n,k)
        labels = dist_to_m.argmin(axis=1)
        new_medoids = medoids.copy()
        changed = False

        for c in range(k):
            idx = np.where(labels == c)[0]
            if idx.size == 0:
                # pick farthest point from its nearest medoid (to reduce empties)
                nearest = dist_to_m.min(axis=1)
                order = np.argsort(-nearest)
                for cand in order:
                    if int(cand) not in new_medoids:
                        new_medoids[c] = int(cand)
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
    return medoids, labels, inertia


@dataclass
class SelectionResult:
    k: int
    medoid_indices: List[int]
    cluster_labels: List[int]
    cluster_sizes: List[int]
    cluster_weights: List[float]
    inertia: float


def full_kmedoids_from_Xn(Xn: sparse.csr_matrix, k: int, rng: np.random.Generator, max_iter: int) -> SelectionResult:
    K = cosine_similarity_matrix(Xn)
    D = cosine_distance_from_similarity(K)  # (n,n)
    medoids, labels, inertia = k_medoids_lloyd(D, k, rng, max_iter=max_iter)

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
) -> SelectionResult:
    """
    CLARA-style approximate k-medoids without building full NxN.
    - Sample subset
    - Compute subset distance
    - Cluster subset
    - Evaluate on full set by assigning to nearest medoid using distances-to-medoids only
    """
    n = Xn.shape[0]
    subset_size = min(subset_size, n)
    best: Optional[SelectionResult] = None

    for _ in range(n_trials):
        subset = rng.choice(np.arange(n), size=subset_size, replace=False)
        subset = np.array(sorted(map(int, subset.tolist())), dtype=int)

        Xs = Xn[subset]
        Ks = cosine_similarity_matrix(Xs)
        Ds = cosine_distance_from_similarity(Ks)

        med_sub, _, _ = k_medoids_lloyd(Ds, k, rng, max_iter=max_iter)
        med_global = subset[med_sub].tolist()

        # Assign full dataset to nearest medoid (distances-to-medoids only)
        S = cosine_similarity_to_medoids(Xn, med_global)        # (n,k)
        D_to_med = cosine_distance_from_similarity(S)           # (n,k)
        labels = D_to_med.argmin(axis=1)
        inertia = float(D_to_med.min(axis=1).sum())

        sizes = [int(np.sum(labels == c)) for c in range(k)]
        weights = [s / float(n) for s in sizes]

        cand = SelectionResult(
            k=k,
            medoid_indices=[int(m) for m in med_global],
            cluster_labels=labels.astype(int).tolist(),
            cluster_sizes=sizes,
            cluster_weights=weights,
            inertia=inertia,
        )

        if best is None or cand.inertia < best.inertia:
            best = cand

    assert best is not None
    return best


def save_json(out_path: str, payload: dict) -> None:
    out_path = os.fspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Directed WL kernel + k-medoids scenario selection")
    ap.add_argument("--graphs_dir", type=str, default='graphs', help="Directory containing graph pickle files")
    ap.add_argument("--pattern", type=str, default="graph_*.pickle", help="Glob pattern inside graphs_dir")
    ap.add_argument("--out_dir", type=str, default="outputs", help="Output directory")
    ap.add_argument("--k_values", type=int, nargs="+", default=[20, 100], help="K values (e.g., 20 100)")
    ap.add_argument("--wl_iterations", type=int, default=3, help="WL iterations h")
    ap.add_argument("--hash_dim", type=int, default=2**18, help="Feature hashing dimension")
    ap.add_argument("--node_label_attr", type=str, default=None, help="Optional node attribute for initial labels")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--max_full_n", type=int, default=1200, help="If N <= this, compute full NxN; else CLARA")
    ap.add_argument("--subset_size", type=int, default=800, help="CLARA subset size")
    ap.add_argument("--clara_trials", type=int, default=10, help="CLARA trials")
    ap.add_argument("--max_iter", type=int, default=50, help="k-medoids iterations")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit for quick tests")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    repo_root = find_repo_root()
    graphs_dir = Path(args.graphs_dir)
    out_dir = Path(args.out_dir)
    if not graphs_dir.is_absolute():
        graphs_dir = (repo_root / graphs_dir).resolve()
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()

    files = discover_files(str(graphs_dir), args.pattern)
    if args.limit is not None:
        files = files[: args.limit]
    if len(files) == 0:
        raise RuntimeError("No pickle files found. Check --graphs_dir and --pattern.")

    print(f"[info] Found {len(files)} files")

    graphs: List[nx.DiGraph] = []
    for p in tqdm(files, desc="Loading DAG graphs"):
        graphs.append(load_nx_dag(p))

    # WL features -> sparse matrix
    hasher = DirectedWLKernelHasher(h=args.wl_iterations, hash_dim=args.hash_dim, node_label_attr=args.node_label_attr)
    X = hasher.fit_transform(graphs)
    norms = row_l2_norms(X)
    Xn = row_normalize(X, norms)

    N = Xn.shape[0]
    print(f"[info] Feature matrix: {X.shape} nnz={X.nnz}")
    print(f"[info] N={N} | mode={'FULL' if N <= args.max_full_n else 'CLARA'}")

    for k in args.k_values:
        if k > N:
            raise ValueError(f"k={k} > N={N}")

        if N <= args.max_full_n:
            res = full_kmedoids_from_Xn(Xn, k=k, rng=rng, max_iter=args.max_iter)
        else:
            res = clara_kmedoids_from_Xn(
                Xn, k=k, rng=rng,
                subset_size=args.subset_size,
                n_trials=args.clara_trials,
                max_iter=args.max_iter,
            )

        selected_files = [os.path.basename(files[i]) for i in res.medoid_indices]

        payload = {
            "method": "directed_wl_kernel_kmedoids",
            "k": int(k),
            "wl_iterations": int(args.wl_iterations),
            "hash_dim": int(args.hash_dim),
            "node_label_attr": args.node_label_attr,
            "seed": int(args.seed),
            # what you explicitly need
            "selected_pickle_files": selected_files,
            # useful metadata
            "selected_indices": res.medoid_indices,
            "cluster_sizes": res.cluster_sizes,
            "cluster_weights": res.cluster_weights,
            "cluster_labels": res.cluster_labels,
            "inertia": float(res.inertia),
            "total_graphs": int(N),
            "pattern": args.pattern,
        }

        out_path = out_dir / f"selected_graphs_k{k}.json"
        save_json(out_path, payload)

        print(f"[ok] k={k} -> {out_path} | inertia={res.inertia:.4f} | sizes(min/mean/max)={min(res.cluster_sizes)}/{np.mean(res.cluster_sizes):.1f}/{max(res.cluster_sizes)}")

    print("[done]")


if __name__ == "__main__":
    main()
