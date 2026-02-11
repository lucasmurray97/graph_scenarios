#!/usr/bin/env python3
"""
Baseline 2: Scenario Reduction via Fast Forward Selection (FFS).

Pipeline (in order):
  (1) Represent each scenario as a burn scar (set of burned cells/nodes or binary vector).
      Compute a distance metric c_ij between representations (Jaccard or Hamming/L1).
  (2) Greedy Fast Forward Selection to pick K representatives minimizing:
        sum_i p_i * min_{j in S} c_ij
      Track per-scenario best distance D_i and update with each new representative.
  (3) Assign all scenarios to the nearest representative and reweight representatives:
        p~_j = sum_{i: a(i)=j} p_i
      Also store cluster sizes.
  (4) Evaluate/plot: histogram of distances-to-nearest-rep and visualize burn scars of
      representatives. Compare burned-area distribution: full vs weighted reduced.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm



def discover_files(graphs_dir: str, pattern: str) -> List[str]:
    return sorted(glob.glob(os.path.join(graphs_dir, pattern)))


def load_nx_dag(path: str) -> nx.DiGraph:
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, nx.DiGraph):
        G = obj
    elif isinstance(obj, nx.MultiDiGraph):
        G = nx.DiGraph(obj)
    else:
        raise TypeError(f"{path}: expected nx.DiGraph (or nx.MultiDiGraph), got {type(obj)}")

    if G.number_of_nodes() == 0:
        G = nx.DiGraph()
        G.add_node(0)

    return G


def save_json(out_path: str | Path, payload: dict) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


DEFAULT_GRAPH_BURN_KEYS = [
    "burned_nodes",
    "burned_cells",
    "burned",
    "burn_scar",
    "burnscar",
]
DEFAULT_NODE_BURN_KEYS = ["burned", "is_burned", "burning"]
DEFAULT_STATE_KEYS = ["state", "status"]


def _truthy(x: object) -> bool:
    return bool(x) and str(x).lower() not in {"0", "false", "none"}


def extract_burn_set(
    G: nx.DiGraph,
    graph_attr: Optional[str],
    node_attr: Optional[str],
    state_attr: Optional[str],
    burned_values: Optional[Iterable[object]],
    use_nodes_as_burn: bool = True,
) -> set:
    # 1) Graph-level list of burned nodes/cells
    if graph_attr and graph_attr in G.graph:
        return set(G.graph[graph_attr])
    for key in DEFAULT_GRAPH_BURN_KEYS:
        if key in G.graph:
            return set(G.graph[key])

    # 2) Node-level boolean attribute
    if node_attr:
        return {n for n, d in G.nodes(data=True) if _truthy(d.get(node_attr))}
    for key in DEFAULT_NODE_BURN_KEYS:
        if any(key in d for _, d in G.nodes(data=True)):
            return {n for n, d in G.nodes(data=True) if _truthy(d.get(key))}

    # 3) Node-level state attribute
    if state_attr:
        vals = set(burned_values or {1, "burned", "B", "burn", True})
        return {n for n, d in G.nodes(data=True) if d.get(state_attr) in vals}
    for key in DEFAULT_STATE_KEYS:
        if any(key in d for _, d in G.nodes(data=True)):
            vals = set(burned_values or {1, "burned", "B", "burn", True})
            return {n for n, d in G.nodes(data=True) if d.get(key) in vals}

    # 4) Fallback: if use_nodes_as_burn is True, treat all nodes as burned
    #    This is the correct interpretation for fire-spread DAGs where
    #    the graph itself represents the burn scar
    if use_nodes_as_burn:
        return set(G.nodes())

    return set()


def build_global_nodes(burn_sets: Sequence[set]) -> List[object]:
    all_nodes = set()
    for s in burn_sets:
        all_nodes.update(s)
    return sorted(all_nodes)


def build_vectors(burn_sets: Sequence[set], node_list: Sequence[object]) -> List[np.ndarray]:
    idx = {n: i for i, n in enumerate(node_list)}
    vecs: List[np.ndarray] = []
    n = len(node_list)
    for s in burn_sets:
        v = np.zeros(n, dtype=np.uint8)
        for node in s:
            i = idx.get(node)
            if i is not None:
                v[i] = 1
        vecs.append(v)
    return vecs


# -----------------------------
# Distances
# -----------------------------
def jaccard_distance(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return 1.0 - (inter / union)


def hamming_distance_sets(a: set, b: set, total_nodes: Optional[int] = None) -> float:
    sym = len(a ^ b)
    if total_nodes is None or total_nodes <= 0:
        return float(sym)
    return float(sym) / float(total_nodes)


def hamming_distance_vec(a: np.ndarray, b: np.ndarray, normalize: bool = True) -> float:
    diff = np.count_nonzero(a != b)
    if normalize:
        return float(diff) / float(a.size)
    return float(diff)


def make_distance_fn(
    rep_kind: str,
    metric: str,
    total_nodes: Optional[int] = None,
    normalize_hamming: bool = True,
):
    metric = metric.lower()
    rep_kind = rep_kind.lower()

    if rep_kind == "set":
        if metric == "jaccard":
            return lambda a, b: jaccard_distance(a, b)
        if metric in {"hamming", "l1"}:
            return lambda a, b: hamming_distance_sets(a, b, total_nodes=total_nodes if normalize_hamming else None)
        raise ValueError(f"Unknown metric for set: {metric}")

    if rep_kind == "vector":
        if metric in {"hamming", "l1"}:
            return lambda a, b: hamming_distance_vec(a, b, normalize=normalize_hamming)
        if metric == "jaccard":
            # Jaccard on vectors: treat 1's as set
            return lambda a, b: jaccard_distance(set(np.flatnonzero(a)), set(np.flatnonzero(b)))
        raise ValueError(f"Unknown metric for vector: {metric}")

    raise ValueError(f"Unknown representation kind: {rep_kind}")


# -----------------------------
# Fast Forward Selection (FFS)
# -----------------------------
def compute_distance_matrix(reps: Sequence, dist_fn) -> np.ndarray:
    n = len(reps)
    D = np.zeros((n, n), dtype=np.float64)
    for i in tqdm(range(n), desc="Pairwise distances"):
        for j in range(i + 1, n):
            d = dist_fn(reps[i], reps[j])
            D[i, j] = d
            D[j, i] = d
    return D


def fast_forward_selection(
    reps: Sequence,
    probs: np.ndarray,
    k: int,
    dist_fn,
    precomputed: Optional[np.ndarray] = None,
) -> Tuple[List[int], np.ndarray, float]:
    n = len(reps)
    if k < 1 or k > n:
        raise ValueError(f"Invalid k={k} for n={n}")

    selected: List[int] = []
    D = np.full(n, np.inf, dtype=np.float64)
    best_cost = float("inf")

    dist_cache: Dict[int, np.ndarray] = {}

    def get_dists_to(j: int) -> np.ndarray:
        if precomputed is not None:
            return precomputed[:, j]
        if j in dist_cache:
            return dist_cache[j]
        d = np.array([dist_fn(reps[i], reps[j]) for i in range(n)], dtype=np.float64)
        dist_cache[j] = d
        return d

    for _ in tqdm(range(k), desc="FFS selection"):
        best_j = None
        best_j_cost = float("inf")
        best_j_d = None

        for j in range(n):
            if j in selected:
                continue
            d_j = get_dists_to(j)
            new_cost = float(np.sum(probs * np.minimum(D, d_j)))
            if new_cost < best_j_cost:
                best_j_cost = new_cost
                best_j = j
                best_j_d = d_j

        assert best_j is not None and best_j_d is not None
        selected.append(int(best_j))
        D = np.minimum(D, best_j_d)
        best_cost = best_j_cost

    return selected, D, best_cost


def fast_forward_selection_vectorized(
    burn_matrix: np.ndarray,
    sizes: np.ndarray,
    probs: np.ndarray,
    k: int,
    batch_size: int = 1000,
) -> Tuple[List[int], np.ndarray, float]:
    """
    Vectorized FFS using batched numpy operations on binary burn matrix.

    Much faster and more memory-efficient than the set-based version for large N.
    Uses batched matrix operations to compute Jaccard distances for many candidates at once.

    Parameters
    ----------
    burn_matrix : (N, D) uint8 binary matrix where D = number of unique nodes
    sizes : (N,) float64 array of burn scar sizes (row sums)
    probs : (N,) float64 probability weights
    k : number of representatives to select
    batch_size : candidates to evaluate per batch (controls memory)
    """
    N = burn_matrix.shape[0]
    D_best = np.full(N, np.inf, dtype=np.float64)
    selected: List[int] = []
    selected_set: set = set()

    # Pre-convert to float32 for faster matrix ops
    B = burn_matrix.astype(np.float32)
    sizes_f = sizes.astype(np.float64)

    for step in tqdm(range(k), desc="FFS selection (vectorized)"):
        best_j = -1
        best_cost = float("inf")
        best_dj: Optional[np.ndarray] = None

        # Build candidate list (exclude already selected)
        candidates = np.array([j for j in range(N) if j not in selected_set], dtype=np.int64)
        n_cand = len(candidates)

        for b_start in range(0, n_cand, batch_size):
            b_end = min(b_start + batch_size, n_cand)
            batch_idx = candidates[b_start:b_end]

            # Compute intersection: B_all @ B_batch.T -> (N, bs)
            B_batch = B[batch_idx]       # (bs, D)
            inter = B @ B_batch.T        # (N, bs) - dot product gives intersection count

            # Compute union and Jaccard distance
            sizes_batch = sizes_f[batch_idx]  # (bs,)
            union = sizes_f[:, None] + sizes_batch[None, :] - inter  # (N, bs)

            # Jaccard distance = 1 - intersection/union (handle zero union)
            dist_batch = np.where(union > 0, 1.0 - inter / union, 0.0)  # (N, bs)

            # Compute cost for each candidate: sum(probs * min(D_best, dist_j))
            new_D = np.minimum(D_best[:, None], dist_batch)  # (N, bs)
            costs = (probs[:, None] * new_D).sum(axis=0)     # (bs,)

            # Find best in this batch
            local_best = int(np.argmin(costs))
            if costs[local_best] < best_cost:
                best_cost = float(costs[local_best])
                best_j = int(batch_idx[local_best])
                best_dj = dist_batch[:, local_best].copy()

            del B_batch, inter, union, dist_batch, new_D, costs

        assert best_j >= 0 and best_dj is not None
        selected.append(best_j)
        selected_set.add(best_j)
        D_best = np.minimum(D_best, best_dj)

    final_cost = float(np.sum(probs * D_best))
    return selected, D_best, final_cost


# -----------------------------
# Assignment + reweighting
# -----------------------------
def assign_clusters(
    reps: Sequence,
    selected: Sequence[int],
    dist_fn,
    precomputed: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(reps)
    k = len(selected)
    if precomputed is not None:
        Dk = precomputed[:, selected]  # (n, k)
    else:
        Dk = np.zeros((n, k), dtype=np.float64)
        for c, j in enumerate(selected):
            d = np.array([dist_fn(reps[i], reps[j]) for i in range(n)], dtype=np.float64)
            Dk[:, c] = d

    labels = np.argmin(Dk, axis=1).astype(int)
    d_min = Dk[np.arange(n), labels]
    return labels, d_min


def reweight_clusters(labels: np.ndarray, probs: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
    sizes = [int(np.sum(labels == c)) for c in range(k)]
    weights = [float(np.sum(probs[labels == c])) for c in range(k)]
    return sizes, weights


# -----------------------------
# Plotting helpers
# -----------------------------
def infer_grid_shape_from_nodes(nodes: Iterable[object]) -> Optional[Tuple[int, int]]:
    nodes = list(nodes)
    if not nodes:
        return None
    if all(isinstance(n, tuple) and len(n) == 2 for n in nodes):
        rows = max(int(n[0]) for n in nodes) + 1
        cols = max(int(n[1]) for n in nodes) + 1
        return rows, cols
    return None


def plot_burn_scar_set(
    burn_set: set,
    out_path: Path,
    grid_shape: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,
) -> None:
    plt.figure(figsize=(5, 4))
    if grid_shape is not None and burn_set:
        rows, cols = grid_shape
        img = np.zeros((rows, cols), dtype=np.uint8)
        if all(isinstance(n, tuple) and len(n) == 2 for n in burn_set):
            for r, c in burn_set:
                if 0 <= r < rows and 0 <= c < cols:
                    img[int(r), int(c)] = 1
        else:
            for n in burn_set:
                if isinstance(n, int):
                    r, c = divmod(int(n), cols)
                    if 0 <= r < rows and 0 <= c < cols:
                        img[r, c] = 1
        plt.imshow(img, cmap="Reds", interpolation="nearest")
        plt.axis("off")
    else:
        # Fallback: scatter node ids if no grid info
        xs = list(burn_set)
        ys = np.zeros(len(xs))
        plt.scatter(xs, ys, s=6)
        plt.yticks([])
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline 2: Fast Forward Selection (FFS) for scenario reduction")
    ap.add_argument("--graphs_dir", type=str, default="data/sub20/graphs", help="Directory containing graph pickle files")
    ap.add_argument("--pattern", type=str, default="graph_*.pickle", help="Glob pattern inside graphs_dir")
    ap.add_argument("--out_dir", type=str, default="src/kernels/outputs", help="Output directory")

    ap.add_argument("--k_values", type=int, nargs="+", default=[20, 100], help="K values (e.g., 20 100)")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit for quick tests")

    # Representation + distance
    ap.add_argument("--representation", type=str, default="set", choices=["set", "vector"], help="Burn-scar representation")
    ap.add_argument("--distance", type=str, default="jaccard", choices=["jaccard", "hamming", "l1"], help="Distance metric")
    ap.add_argument("--normalize_hamming", action="store_true", help="Normalize Hamming/L1 by total nodes")

    ap.add_argument("--graph_burn_attr", type=str, default=None, help="Graph-level burned nodes attribute")
    ap.add_argument("--node_burn_attr", type=str, default=None, help="Node-level burned attribute")
    ap.add_argument("--state_attr", type=str, default=None, help="Node-level state attribute")
    ap.add_argument(
        "--burned_values",
        type=str,
        nargs="*",
        default=None,
        help="State values treated as burned (e.g., 1 burned B)",
    )

    ap.add_argument("--prob_attr", type=str, default=None, help="Graph-level probability attribute name")
    ap.add_argument("--prob_path", type=str, default=None, help="CSV/JSON with filename->probability mapping")

    ap.add_argument("--precompute_threshold", type=int, default=2000, help="Precompute distances if N <= threshold")
    ap.add_argument("--plots", action="store_true", help="Generate evaluation plots")
    ap.add_argument("--plot_medoids", type=int, default=12, help="Number of medoids to visualize")
    ap.add_argument("--grid_shape", type=int, nargs=2, default=None, help="Grid shape rows cols for burn scar plots")
    ap.add_argument("--nodes_as_burn", action="store_true", default=True,
                    help="Treat all graph nodes as burn scar (default for fire-spread DAGs)")
    ap.add_argument("--no_nodes_as_burn", dest="nodes_as_burn", action="store_false",
                    help="Do not use nodes as burn scar fallback")

    args = ap.parse_args()

    import time

    graphs_dir = Path(args.graphs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=" * 60)
    print("Baseline 2: Fast Forward Selection (FFS)")
    print(f"=" * 60)

    files = discover_files(str(graphs_dir), args.pattern)
    if args.limit is not None:
        files = files[: int(args.limit)]
    if len(files) == 0:
        raise RuntimeError("No pickle files found. Check --graphs_dir and --pattern.")

    print(f"[info] Found {len(files)} files under {graphs_dir}")

    # --- (1) Representation + distances ---
    burn_sets: List[set] = []
    probs: List[float] = []

    prob_map = None
    if args.prob_path:
        ext = os.path.splitext(args.prob_path)[1].lower()
        with open(args.prob_path, "r", encoding="utf-8") as f:
            if ext in {".json"}:
                prob_map = json.load(f)
            else:
                # CSV with "file,prob"
                prob_map = {}
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    name, val = line.split(",")[:2]
                    prob_map[name.strip()] = float(val)

    for p in tqdm(files, desc="Load burn scars"):
        G = load_nx_dag(p)
        s = extract_burn_set(
            G,
            graph_attr=args.graph_burn_attr,
            node_attr=args.node_burn_attr,
            state_attr=args.state_attr,
            burned_values=args.burned_values,
            use_nodes_as_burn=args.nodes_as_burn,
        )
        burn_sets.append(s)

        if args.prob_attr and args.prob_attr in G.graph:
            probs.append(float(G.graph[args.prob_attr]))
        elif prob_map is not None:
            probs.append(float(prob_map.get(os.path.basename(p), 0.0)))
        else:
            probs.append(1.0)

    probs_arr = np.array(probs, dtype=np.float64)
    if probs_arr.sum() <= 0:
        probs_arr = np.ones_like(probs_arr)
    probs_arr = probs_arr / probs_arr.sum()

    rep_kind = args.representation.lower()
    reps: List = burn_sets
    total_nodes = None

    if rep_kind == "vector":
        node_list = build_global_nodes(burn_sets)
        total_nodes = len(node_list)
        reps = build_vectors(burn_sets, node_list)
    else:
        if args.normalize_hamming:
            total_nodes = len(build_global_nodes(burn_sets))

    dist_fn = make_distance_fn(
        rep_kind=rep_kind,
        metric=args.distance,
        total_nodes=total_nodes,
        normalize_hamming=args.normalize_hamming,
    )

    # Precompute distances if N is moderate
    precomputed = None
    if len(reps) <= int(args.precompute_threshold):
        precomputed = compute_distance_matrix(reps, dist_fn)

    # --- (2) Fast Forward Selection ---
    # For large N with set representation and Jaccard, use vectorized FFS
    use_vectorized = (len(reps) > int(args.precompute_threshold) and rep_kind == "set")
    burn_matrix = None
    burn_sizes = None

    if use_vectorized:
        print(f"[info] Using vectorized FFS (N={len(reps)} > threshold={args.precompute_threshold})")
        all_nodes_sorted = sorted(set().union(*burn_sets))
        node_to_idx = {n: i for i, n in enumerate(all_nodes_sorted)}
        D_dim = len(all_nodes_sorted)
        burn_matrix = np.zeros((len(burn_sets), D_dim), dtype=np.uint8)
        for i, s in enumerate(burn_sets):
            for node in s:
                burn_matrix[i, node_to_idx[node]] = 1
        burn_sizes = burn_matrix.sum(axis=1).astype(np.float64)
        print(f"[info] Binary matrix shape: {burn_matrix.shape}, density: {burn_matrix.mean():.3f}")

    for k in [int(v) for v in args.k_values]:
        if use_vectorized:
            selected, D_best, obj = fast_forward_selection_vectorized(
                burn_matrix=burn_matrix,
                sizes=burn_sizes,
                probs=probs_arr,
                k=k,
                batch_size=1000,
            )
            # Compute cluster assignments from selected representatives
            B_sel = burn_matrix[selected].astype(np.float32)
            inter_sel = burn_matrix.astype(np.float32) @ B_sel.T
            sizes_sel = burn_sizes[selected]
            union_sel = burn_sizes[:, None] + sizes_sel[None, :] - inter_sel
            dist_to_sel = np.where(union_sel > 0, 1.0 - inter_sel / union_sel, 0.0)
            labels = np.argmin(dist_to_sel, axis=1).astype(int)
            d_min = dist_to_sel[np.arange(len(dist_to_sel)), labels]
            del B_sel, inter_sel, union_sel, dist_to_sel
        else:
            selected, D_best, obj = fast_forward_selection(
                reps=reps,
                probs=probs_arr,
                k=k,
                dist_fn=dist_fn,
                precomputed=precomputed,
            )
            labels, d_min = assign_clusters(reps, selected, dist_fn, precomputed=precomputed)

        # --- (3) Assignment + reweight ---
        cluster_sizes, weights = reweight_clusters(labels, probs_arr, k)

        selected_files = [os.path.basename(files[i]) for i in selected]

        payload = {
            "method": "fast_forward_selection",
            "k": int(k),
            "graphs_dir": str(graphs_dir),
            "pattern": args.pattern,
            "representation": rep_kind,
            "distance": args.distance,
            "normalize_hamming": bool(args.normalize_hamming),
            "selected_indices": selected,
            "selected_pickle_files": selected_files,
            "cluster_labels": labels.tolist(),
            "cluster_sizes": cluster_sizes,
            "cluster_weights": weights,
            "objective": float(obj),
            "total_graphs": len(files),
        }

        out_path = out_dir / f"fss_selected_graphs_k{k}.json"
        save_json(out_path, payload)

        # Print statistics similar to WL baseline
        print(f"\n{'='*40}")
        print(f"FFS Results for k={k}")
        print(f"{'='*40}")
        print(f"  Objective (weighted avg distance): {obj:.6f}")
        print(f"  Output: {out_path}")
        print(f"\nCluster size statistics:")
        print(f"  Min: {min(cluster_sizes)}")
        print(f"  Max: {max(cluster_sizes)}")
        print(f"  Mean: {np.mean(cluster_sizes):.1f}")
        print(f"  Std: {np.std(cluster_sizes):.1f}")

        # --- (4) Evaluation / plots ---
        if args.plots:
            # Histogram of distance to nearest representative
            plt.figure(figsize=(6, 4))
            plt.hist(d_min, bins=40, color="steelblue", alpha=0.8)
            plt.title(f"Distance to nearest representative (k={k})")
            plt.xlabel("distance")
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(out_dir / f"fss_k{k}_nearest_dist_hist.png", dpi=200)
            plt.close()

            # Burned area distribution: full vs reduced weighted
            areas_all = np.array([len(s) for s in burn_sets], dtype=np.float64)
            areas_sel = np.array([len(burn_sets[i]) for i in selected], dtype=np.float64)
            weights_sel = np.array(weights, dtype=np.float64)
            if weights_sel.sum() > 0:
                weights_sel = weights_sel / weights_sel.sum()

            plt.figure(figsize=(6, 4))
            bins = min(40, max(10, int(np.sqrt(len(areas_all)))))
            plt.hist(areas_all, bins=bins, alpha=0.5, label="all")
            plt.hist(areas_sel, bins=bins, weights=weights_sel, alpha=0.5, label="reduced (weighted)")
            plt.title(f"Burned area distribution (k={k})")
            plt.xlabel("burned cells/nodes")
            plt.ylabel("count / weight")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"fss_k{k}_burned_area_dist.png", dpi=200)
            plt.close()

            # Burn scar visualizations for medoids
            n_plot = min(int(args.plot_medoids), len(selected))
            grid_shape = tuple(args.grid_shape) if args.grid_shape is not None else None
            if grid_shape is None:
                grid_shape = infer_grid_shape_from_nodes(burn_sets[selected[0]])

            for rank, idx in enumerate(selected[:n_plot]):
                out_img = out_dir / f"fss_k{k}_medoid_{rank:02d}.png"
                title = f"Medoid {rank} (idx={idx})"
                plot_burn_scar_set(burn_sets[idx], out_img, grid_shape=grid_shape, title=title)


if __name__ == "__main__":
    main()
