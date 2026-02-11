#!/usr/bin/env python3
"""
Fast Forward Selection (FFS) for Scenario Reduction.

A vectorized, memory-efficient implementation for selecting K representative
scenarios from a large set of stochastic simulations (e.g., fire burn scars).

Key Features:
  - Sparse Matrix representation for memory efficiency.
  - Batched matrix multiplication for speed (Vectorized).
  - Supports both 'Hamming' (Risk/Magnitude) and 'Jaccard' (Shape) metrics.
  - Probability-aware: Prioritizes likely scenarios.
"""

import argparse
import glob
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import sparse
from tqdm import tqdm

# ==========================================
# 1. Data Loading & Matrix Conversion
# ==========================================

class ScenarioDataset:
    """
    Handles loading of graph files and conversion to a Sparse Binary Matrix.
    """
    def __init__(self, file_paths: List[str], prob_map: Optional[Dict[str, float]] = None):
        self.file_paths = sorted(file_paths)
        self.filenames = [os.path.basename(p) for p in self.file_paths]
        self.n_scenarios = len(self.file_paths)
        self.node_to_idx = {}
        self.idx_to_node = []
        self.matrix = None  # Will be (N_scenarios, N_unique_nodes)
        self.probs = None   # Will be (N_scenarios,)
        
        # Load data
        self._build_matrix(prob_map)

    def _build_matrix(self, prob_map):
        print(f"[Dataset] Loading {self.n_scenarios} scenarios...")
        
        rows = []
        cols = []
        data = []
        
        loaded_probs = []
        
        # Pass 1: Load all graphs, build global node index, build sparse matrix data
        for i, path in enumerate(tqdm(self.file_paths, desc="Loading Graphs")):
            # Load Graph
            try:
                with open(path, "rb") as f:
                    G = pickle.load(f)
            except Exception as e:
                print(f"[Warning] Failed to load {path}: {e}")
                loaded_probs.append(0.0)
                continue

            # Extract Burned Nodes
            # Fallback: If no explicit 'burned' attribute, assume all nodes in G are burned
            # (Standard for propagation DAGs)
            burned_nodes = list(G.nodes())
            
            # Map nodes to global indices
            for node in burned_nodes:
                if node not in self.node_to_idx:
                    self.node_to_idx[node] = len(self.idx_to_node)
                    self.idx_to_node.append(node)
                
                col_idx = self.node_to_idx[node]
                rows.append(i)
                cols.append(col_idx)
                data.append(1)

            # Handle Probability
            p = 1.0
            if prob_map and self.filenames[i] in prob_map:
                p = prob_map[self.filenames[i]]
            elif hasattr(G, "graph") and "probability" in G.graph:
                p = float(G.graph["probability"])
            loaded_probs.append(p)

        # Create CSR Sparse Matrix
        n_unique_nodes = len(self.idx_to_node)
        self.matrix = sparse.csr_matrix(
            (data, (rows, cols)), 
            shape=(self.n_scenarios, n_unique_nodes), 
            dtype=np.uint8
        )
        
        # Normalize Probabilities
        self.probs = np.array(loaded_probs, dtype=np.float64)
        if self.probs.sum() > 0:
            self.probs /= self.probs.sum()
        else:
            self.probs = np.full(self.n_scenarios, 1.0 / self.n_scenarios)
            
        print(f"[Dataset] Matrix Shape: {self.matrix.shape} (Scenarios x UniqueNodes)")
        print(f"[Dataset] Matrix Density: {self.matrix.nnz / (self.matrix.shape[0]*self.matrix.shape[1]):.4f}")

# ==========================================
# 2. Vectorized Distance Engine
# ==========================================

class DistanceEngine:
    """
    Computes distances between a 'batch' of candidates and 'all' scenarios
    using vectorized matrix operations.
    """
    @staticmethod
    def compute_batch(
        candidates_mat: sparse.csr_matrix, 
        all_mat: sparse.csr_matrix, 
        metric: str,
        cand_sizes: np.ndarray,
        all_sizes: np.ndarray
    ) -> np.ndarray:
        """
        Returns a distance matrix of shape (N_all, N_candidates).
        """
        # 1. Intersection Count (Dot Product)
        # Result shape: (N_all, N_candidates)
        # We use all_mat @ candidates_mat.T
        intersection = all_mat.dot(candidates_mat.T).toarray()
        
        # 2. Compute Metric
        if metric == "hamming":
            # Hamming = |A| + |B| - 2*|A n B|
            # Broadcasting: (N_all, 1) + (1, N_cand) - (N_all, N_cand)
            dist = all_sizes[:, None] + cand_sizes[None, :] - (2 * intersection)
            
            # Optional: Normalize by max possible size (union of all nodes) could be done here
            # but usually raw Hamming is better for Risk Analysis.
            return np.maximum(dist, 0)
            
        elif metric == "jaccard":
            # Jaccard = 1 - (|A n B| / |A u B|)
            # |A u B| = |A| + |B| - |A n B|
            union = all_sizes[:, None] + cand_sizes[None, :] - intersection
            
            # Avoid divide by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                jaccard_sim = intersection / union
                jaccard_sim[union == 0] = 0.0 # If both empty, sim is 0? or 1? usually 1. 
                                              # But if union=0, distance=0 usually.
                                              # Let's assume dist=0 if both empty.
                
            dist = 1.0 - jaccard_sim
            dist[union == 0] = 0.0 
            return dist
            
        else:
            raise ValueError(f"Unknown metric: {metric}")

# ==========================================
# 3. Fast Forward Selection (The Algo)
# ==========================================

class FastForwardSelector:
    def __init__(self, dataset: ScenarioDataset, metric: str = "hamming"):
        self.ds = dataset
        self.metric = metric
        
        # Precompute scenario sizes (number of burned nodes) for fast distance calc
        self.sizes = np.array(self.ds.matrix.sum(axis=1)).flatten()
        
    def select(self, k: int, batch_size: int = 1000):
        """
        Run the Greedy FFS algorithm.
        """
        n = self.ds.n_scenarios
        selected_indices = []
        selected_set = set()
        
        # Track min distance from every scenario to the *closest selected* rep
        # Initialize to infinity
        min_dists = np.full(n, np.inf, dtype=np.float64)
        
        print(f"\n[FFS] Starting Selection (k={k}, metric={self.metric})...")
        
        # The FFS Loop
        for step in range(k):
            best_cand_idx = -1
            best_cost_reduction = -1.0 # We want to MAXIMIZE reduction (or MINIMIZE total cost)
            best_new_dists = None
            
            # Identify candidates (scenarios not yet selected)
            candidates = [x for x in range(n) if x not in selected_set]
            n_candidates = len(candidates)
            
            # Process candidates in batches to save memory
            current_total_cost = np.sum(self.ds.probs * min_dists) if step > 0 else np.inf
            
            # Loop over batches
            for b_start in range(0, n_candidates, batch_size):
                b_end = min(b_start + batch_size, n_candidates)
                batch_indices = candidates[b_start:b_end]
                
                # Extract sub-matrices
                cand_mat = self.ds.matrix[batch_indices]
                cand_sizes = self.sizes[batch_indices]
                
                # Compute distances: (N_all, Batch_Size)
                dists_batch = DistanceEngine.compute_batch(
                    cand_mat, self.ds.matrix, self.metric, cand_sizes, self.sizes
                )
                
                # Compute "Potential New Cost" for each candidate in batch
                # New_Min_Dist = min(Old_Min_Dist, Dist_to_Candidate)
                # We want the one that minimizes sum(Prob * New_Min_Dist)
                
                # Broadcasting: min( (N,1), (N, Batch) ) -> (N, Batch)
                new_dists_matrix = np.minimum(min_dists[:, None], dists_batch)
                
                # Weighted Sum: (N, 1) * (N, Batch) -> Sum over axis 0 -> (Batch,)
                new_costs = np.sum(self.ds.probs[:, None] * new_dists_matrix, axis=0)
                
                # Find best in batch
                local_best_idx_in_batch = np.argmin(new_costs)
                local_min_cost = new_costs[local_best_idx_in_batch]
                
                if local_min_cost < current_total_cost:
                    # Found a better candidate
                    current_total_cost = local_min_cost
                    best_cand_idx = batch_indices[local_best_idx_in_batch]
                    best_new_dists = new_dists_matrix[:, local_best_idx_in_batch]

            # End of Batches -> Register the winner
            if best_cand_idx == -1:
                # Should only happen if k > N or something weird
                print(f"[FFS] No improvement found at step {step}. Stopping early.")
                break
                
            selected_indices.append(best_cand_idx)
            selected_set.add(best_cand_idx)
            min_dists = best_new_dists # Update global min distances
            
            print(f"  Step {step+1}/{k}: Selected Scenario {best_cand_idx} | Cost: {current_total_cost:.4f}")

        return selected_indices, min_dists

    def reweight(self, selected_indices: List[int], batch_size: int = 1000):
        """
        Assign every scenario to its closest selected representative and compute weights.
        """
        n = self.ds.n_scenarios
        k = len(selected_indices)
        
        # Compute distances from ALL scenarios to SELECTED representatives
        # (N, K)
        sel_mat = self.ds.matrix[selected_indices]
        sel_sizes = self.sizes[selected_indices]
        
        # We can do this in one shot since K is usually small
        dists = DistanceEngine.compute_batch(
            sel_mat, self.ds.matrix, self.metric, sel_sizes, self.sizes
        )
        
        # Assign to closest (argmin over columns)
        # labels[i] = index (0..k-1) of the representative closest to scenario i
        labels = np.argmin(dists, axis=1)
        
        # Sum probabilities
        new_weights = np.zeros(k)
        cluster_sizes = np.zeros(k, dtype=int)
        
        for i in range(n):
            c_idx = labels[i]
            new_weights[c_idx] += self.ds.probs[i]
            cluster_sizes[c_idx] += 1
            
        return labels, new_weights, cluster_sizes

# ==========================================
# 4. Visualization & Output
# ==========================================

def plot_results(
    dataset: ScenarioDataset, 
    selected_indices: List[int], 
    weights: np.ndarray, 
    out_dir: Path,
    grid_width: Optional[int] = None
):
    """
    Generates plots:
    1. Medoids (Visual of selected burn scars)
    2. Probability Map (Weighted average of selection)
    """
    # Helper to plot a single scar
    def _plot_scar(node_indices, title, fpath):
        plt.figure(figsize=(5, 5))
        
        # Get coordinates
        # If grid_width is known, we map 1D index -> 2D (x, y)
        # If not, we map 1D index -> (index, 0) or just scatter
        xs, ys = [], []
        if grid_width:
            for idx in node_indices:
                # Assuming row-major: idx = y * width + x
                y = idx // grid_width
                x = idx % grid_width
                xs.append(x)
                ys.append(y)
            plt.scatter(xs, ys, c='red', s=10, marker='s')
            plt.gca().invert_yaxis()
        else:
            # Fallback: Just scatter the node IDs
            plt.scatter(node_indices, np.zeros_like(node_indices), alpha=0.1)
            plt.yticks([])
            plt.xlabel("Node Index (Linear)")
        
        plt.title(title)
        plt.tight_layout()
        plt.savefig(fpath)
        plt.close()

    # 1. Plot Medoids
    print("[Plotting] Generating Medoid plots...")
    for i, (idx, w) in enumerate(zip(selected_indices, weights)):
        # Reconstruct indices from sparse row
        row = dataset.matrix[idx]
        node_cols = row.indices # These are column indices in matrix
        # Map column indices back to real Node IDs (if different)
        # But for plotting x,y relative to the matrix, we can use col indices directly 
        # if the node universe is dense. If sparse, we might need mapping.
        # Let's use the 'idx_to_node' mapping.
        real_nodes = [dataset.idx_to_node[c] for c in node_cols]
        
        # If real_nodes are ints, use them. 
        # If they are tuples, we might need to unzip them.
        # Given the previous context, let's assume real_nodes are the spatial integers.
        
        fname = out_dir / f"selected_rep_{i:02d}_orig_{dataset.filenames[idx]}.png"
        _plot_scar(real_nodes, f"Rep {i} (p={w:.2f})", fname)

    # 2. Probability Map (Risk Map)
    # Weighted average of the selected binary vectors
    print("[Plotting] Generating Probability Map...")
    
    # We want: Sum( Weight_i * Vector_i )
    # Sparse matrix multiplication: Weights (1, K) @ Selected_Rows (K, Nodes)
    sel_submat = dataset.matrix[selected_indices]
    
    # sparse.csr_matrix does not support direct broadcasting easily,
    # but we can multiply the matrix rows by weights.
    # Convert weights to diag matrix
    w_diag = sparse.diags(weights)
    weighted_sum = (w_diag @ sel_submat).sum(axis=0) # Result is (1, Nodes) matrix
    
    # Convert to dense array for plotting
    risk_array = np.array(weighted_sum).flatten()
    
    # Plotting
    plt.figure(figsize=(6, 5))
    valid_mask = risk_array > 0
    valid_indices = np.where(valid_mask)[0]
    valid_values = risk_array[valid_mask]
    
    if grid_width:
        xs = [dataset.idx_to_node[i] % grid_width for i in valid_indices]
        ys = [dataset.idx_to_node[i] // grid_width for i in valid_indices]
        plt.scatter(xs, ys, c=valid_values, cmap='Reds', s=10, marker='s')
        plt.colorbar(label="Burn Probability")
        plt.gca().invert_yaxis()
    else:
        # Fallback linear plot
        plt.scatter(valid_indices, valid_values, alpha=0.5, s=2)
        plt.xlabel("Node Index")
        plt.ylabel("Probability")
        
    plt.title("Aggregated Risk Map (Reduced Set)")
    plt.tight_layout()
    plt.savefig(out_dir / "aggregated_risk_map.png")
    plt.close()


# ==========================================
# 5. Main CLI
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Vectorized Fast Forward Selection")
    parser.add_argument("--graphs_dir", type=str, required=True, help="Path to folder with .pickle graphs")
    parser.add_argument("--pattern", type=str, default="*.pickle", help="File pattern (default: *.pickle)")
    parser.add_argument("--k", type=int, default=10, help="Number of scenarios to select")
    parser.add_argument("--distance", type=str, default="hamming", choices=["hamming", "jaccard"], help="Distance metric")
    parser.add_argument("--out_dir", type=str, default="output_ffs", help="Output directory")
    parser.add_argument("--plots", action="store_true", help="Generate plots")
    parser.add_argument("--grid_width", type=int, default=None, help="Grid width for 2D visualization (optional)")
    parser.add_argument("--prob_file", type=str, default=None, help="Optional CSV with filename,probability")
    
    args = parser.parse_args()
    
    # Setup
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Discovery
    search_path = os.path.join(args.graphs_dir, args.pattern)
    files = sorted(glob.glob(search_path))
    if not files:
        print(f"No files found at {search_path}")
        sys.exit(1)
        
    # 2. Load Probability Map (if any)
    prob_map = {}
    if args.prob_file:
        with open(args.prob_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    prob_map[parts[0]] = float(parts[1])
    
    # 3. Build Dataset (Matrix)
    dataset = ScenarioDataset(files, prob_map if prob_map else None)
    
    if args.k > dataset.n_scenarios:
        print(f"Requested k={args.k} > N={dataset.n_scenarios}. Adjusting k to N.")
        args.k = dataset.n_scenarios

    # 4. Run FFS
    selector = FastForwardSelector(dataset, metric=args.distance)
    start_t = time.time()
    selected_indices, _ = selector.select(k=args.k)
    end_t = time.time()
    print(f"\n[FFS] Selection complete in {end_t - start_t:.2f}s")
    
    # 5. Re-weight
    labels, weights, sizes = selector.reweight(selected_indices)
    
    # 6. Save Outputs
    output_data = {
        "method": "fast_forward_selection",
        "metric": args.distance,
        "k": args.k,
        "selected_scenarios": [
            {
                "original_index": int(idx),
                "filename": dataset.filenames[idx],
                "new_weight": float(w),
                "cluster_size": int(s)
            }
            for idx, w, s in zip(selected_indices, weights, sizes)
        ],
        "cluster_assignments": labels.tolist()
    }
    
    json_path = out_dir / "selected_scenarios.json"
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"[Output] Saved results to {json_path}")
    
    # 7. Plots
    if args.plots:
        plot_results(dataset, selected_indices, weights, out_dir, args.grid_width)

if __name__ == "__main__":
    main()