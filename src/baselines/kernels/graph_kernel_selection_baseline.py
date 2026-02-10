"""
Weisfeiler-Lehman Graph Kernel + K-Medoids Baseline for Scenario Selection.

This module implements a graph kernel-based approach for selecting representative
fire-spread scenarios using the Weisfeiler-Lehman subtree kernel and k-medoids clustering.

Optimized implementation using sparse matrices for efficient kernel computation.
"""

import os
import glob
import pickle
import json
import argparse
import hashlib
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
import networkx as nx
from scipy import sparse
from tqdm import tqdm


class WeisfeilerLehmanKernel:
    """
    Weisfeiler-Lehman subtree kernel for directed graphs using sparse matrices.

    The WL kernel computes graph similarity by iteratively relabeling nodes
    based on their neighborhood structure (IN and OUT neighbors for directed graphs),
    then comparing label histograms.

    This implementation properly handles directed graphs by treating IN and OUT
    neighbors separately, which is critical for DAG-structured data like wildfire spread.
    """

    def __init__(self, n_iter: int = 3, normalize: bool = True, hash_dim: int = 2**18, directed: bool = True):
        """
        Initialize WL kernel.

        Args:
            n_iter: Number of WL iterations (h parameter)
            normalize: Whether to normalize the kernel matrix
            hash_dim: Dimension for feature hashing (power of 2 recommended)
            directed: Whether to treat graphs as directed (separate IN/OUT neighbors)
        """
        self.n_iter = n_iter
        self.normalize = normalize
        self.hash_dim = hash_dim
        self.directed = directed

    def _hash_label(self, label: str, iteration: int) -> int:
        """
        Hash a label string to a fixed dimension with iteration separation.

        Including iteration index prevents collisions between features from
        different WL iterations.
        """
        label_with_iter = f"{iteration}|{label}"
        return int(hashlib.md5(label_with_iter.encode()).hexdigest(), 16) % self.hash_dim

    def _compute_features_sparse(self, graphs: List[nx.DiGraph]) -> sparse.csr_matrix:
        """
        Compute WL feature vectors as a sparse matrix.

        For directed graphs, uses separate IN and OUT neighbor multisets.
        Labels are properly compressed after each iteration to prevent exponential growth.

        Args:
            graphs: List of NetworkX graphs (directed or undirected)

        Returns:
            Sparse matrix of shape (n_graphs, hash_dim)
        """
        n = len(graphs)

        # Initialize node labels with degree information
        all_labels = []
        for G in graphs:
            if self.directed and isinstance(G, nx.DiGraph):
                # For directed graphs: use (in_degree, out_degree)
                labels = {node: f"{G.in_degree(node)}_{G.out_degree(node)}" for node in G.nodes()}
            else:
                # For undirected: use total degree
                labels = {node: str(G.degree(node)) for node in G.nodes()}
            all_labels.append(labels)

        # Collect all features in COO format
        rows = []
        cols = []
        data = []

        # Process each graph
        for graph_idx in tqdm(range(n), desc="Computing WL features"):
            G = graphs[graph_idx]
            labels = all_labels[graph_idx]
            feature_counter = Counter()

            # Initial features (iteration 0)
            for node in G.nodes():
                label_hash = self._hash_label(labels[node], iteration=0)
                feature_counter[label_hash] += 1

            # WL iterations
            for iteration in range(1, self.n_iter + 1):
                new_labels = {}

                for node in G.nodes():
                    current_label = labels[node]

                    # Build signature based on graph type
                    if self.directed and isinstance(G, nx.DiGraph):
                        # Separate IN and OUT neighbors for directed graphs
                        in_neighbors = sorted([labels[pred] for pred in G.predecessors(node)])
                        out_neighbors = sorted([labels[succ] for succ in G.successors(node)])
                        signature = f"{current_label}|IN:{'_'.join(in_neighbors)}|OUT:{'_'.join(out_neighbors)}"
                    else:
                        # Undirected case
                        neighbor_labels = sorted([labels[neighbor] for neighbor in G.neighbors(node)])
                        signature = f"{current_label}|{'_'.join(neighbor_labels)}"

                    # Compress signature to a fixed-length label (consistent across graphs)
                    compressed_label = hashlib.md5(signature.encode()).hexdigest()
                    new_labels[node] = compressed_label

                    # Hash the compressed label for features
                    label_hash = self._hash_label(compressed_label, iteration=iteration)
                    feature_counter[label_hash] += 1

                labels = new_labels

            # Add to sparse matrix data
            for col, count in feature_counter.items():
                rows.append(graph_idx)
                cols.append(col)
                data.append(count)

        # Build sparse matrix
        feature_matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n, self.hash_dim),
            dtype=np.float64
        )

        return feature_matrix

    def fit_transform(self, graphs: List[nx.DiGraph]) -> np.ndarray:
        """
        Compute the WL kernel matrix for all graphs.

        Args:
            graphs: List of NetworkX graphs (can be directed or undirected)

        Returns:
            Kernel matrix of shape (n, n)

        Warning:
            This method densifies the kernel matrix, requiring O(n²) memory.
            For large datasets (n > 5000), this may cause memory issues.
        """
        n = len(graphs)
        print(f"Computing WL features (h={self.n_iter}, hash_dim={self.hash_dim}, directed={self.directed})...")

        # Get sparse feature matrix
        features = self._compute_features_sparse(graphs)
        print(f"Feature matrix: {features.shape}, nnz={features.nnz}")

        # Estimate memory requirement
        kernel_size_mb = (n * n * 8) / (1024 ** 2)
        if kernel_size_mb > 1000:
            print(f"WARNING: Kernel matrix will require ~{kernel_size_mb:.1f} MB of memory")
            if kernel_size_mb > 10000:
                raise MemoryError(
                    f"Kernel matrix too large ({kernel_size_mb:.1f} MB). "
                    "Consider reducing dataset size or using approximate methods."
                )

        print("Computing kernel matrix via sparse matrix multiplication...")
        # K = X @ X.T (sparse matrix multiplication, then densify)
        K = (features @ features.T).toarray()

        if self.normalize:
            # Normalize: K_ij / sqrt(K_ii * K_jj)
            diag = np.sqrt(np.diag(K))
            diag[diag == 0] = 1  # Avoid division by zero
            K = K / diag[:, np.newaxis]
            K = K / diag[np.newaxis, :]

        return K


class WLKernelBaseline:
    """
    WL Kernel + K-Medoids baseline for scenario selection.

    This class computes pairwise graph similarities using the Weisfeiler-Lehman
    subtree kernel, then applies k-medoids clustering (Lloyd-style) to select
    representative scenarios.

    Note: K-medoids implementation uses Lloyd-style alternating optimization
    (assign → recompute medoids), not the PAM swap-based algorithm.
    """

    def __init__(
        self,
        graphs_dir: str,
        output_dir: str = "outputs",
        wl_iterations: int = 3,
        normalize: bool = True,
        directed: bool = True,
        random_state: int = 42
    ):
        """
        Initialize the WL Kernel baseline.

        Args:
            graphs_dir: Directory containing graph pickle files
            output_dir: Directory for output files
            wl_iterations: Number of WL iterations (h parameter)
            normalize: Whether to normalize the kernel matrix
            directed: Whether to treat graphs as directed (critical for DAGs)
            random_state: Random seed for reproducibility
        """
        self.graphs_dir = graphs_dir
        self.output_dir = output_dir
        self.wl_iterations = wl_iterations
        self.normalize = normalize
        self.directed = directed
        self.random_state = random_state

        # Will be populated during execution
        self.graph_files: List[str] = []
        self.graphs_nx: List[nx.DiGraph] = []
        self.graphs_processed: List[nx.DiGraph] = []
        self.kernel_matrix: Optional[np.ndarray] = None
        self.distance_matrix: Optional[np.ndarray] = None

    def load_graphs(self, limit: Optional[int] = None) -> None:
        """
        Load all graph pickles from the graphs directory.

        Args:
            limit: Optional limit on number of graphs to load (for testing)
        """
        # Find all pickle files
        pattern = os.path.join(self.graphs_dir, "graph_*.pickle")
        self.graph_files = sorted(glob.glob(pattern))

        if limit is not None:
            self.graph_files = self.graph_files[:limit]

        print(f"Found {len(self.graph_files)} graph files")

        # Load graphs
        self.graphs_nx = []
        for fpath in tqdm(self.graph_files, desc="Loading graphs"):
            with open(fpath, 'rb') as f:
                G = pickle.load(f)
                self.graphs_nx.append(G)

        print(f"Loaded {len(self.graphs_nx)} graphs")

        # Preprocess graphs based on directed setting
        self._preprocess_graphs()

    def _preprocess_graphs(self) -> None:
        """
        Preprocess graphs for WL kernel computation.

        If directed=False, converts to undirected. Otherwise keeps as directed.
        """
        self.graphs_processed = []

        if self.directed:
            print("Using directed graphs (IN/OUT neighbors treated separately)")
            for G in tqdm(self.graphs_nx, desc="Preprocessing graphs"):
                if isinstance(G, nx.MultiDiGraph):
                    G = nx.DiGraph(G)
                # Handle empty graphs
                if G.number_of_nodes() == 0:
                    G_processed = nx.DiGraph()
                    G_processed.add_node(0)  # Add dummy node
                else:
                    G_processed = G
                self.graphs_processed.append(G_processed)
        else:
            print("Converting to undirected graphs (direction information discarded)")
            for G in tqdm(self.graphs_nx, desc="Converting to undirected"):
                if isinstance(G, nx.MultiGraph):
                    G = nx.Graph(G)
                # Handle empty graphs
                if G.number_of_nodes() == 0:
                    G_undir = nx.Graph()
                    G_undir.add_node(0)  # Add dummy node
                else:
                    G_undir = G.to_undirected()
                self.graphs_processed.append(G_undir)

        print(f"Preprocessed {len(self.graphs_processed)} graphs")

    def compute_kernel_matrix(self) -> np.ndarray:
        """
        Compute the WL kernel matrix for all graphs.

        Returns:
            Kernel matrix of shape (n_graphs, n_graphs)
        """
        print(f"\nComputing WL kernel matrix (h={self.wl_iterations}, directed={self.directed})...")

        # Initialize WL kernel
        wl_kernel = WeisfeilerLehmanKernel(
            n_iter=self.wl_iterations,
            normalize=self.normalize,
            directed=self.directed
        )

        # Compute kernel matrix
        self.kernel_matrix = wl_kernel.fit_transform(self.graphs_processed)

        # Verify properties
        n = len(self.graphs_processed)
        assert self.kernel_matrix.shape == (n, n), \
            f"Expected ({n}, {n}), got {self.kernel_matrix.shape}"

        # Ensure symmetry
        self.kernel_matrix = (self.kernel_matrix + self.kernel_matrix.T) / 2

        if self.normalize:
            diag = np.diag(self.kernel_matrix)
            print(f"Kernel diagonal range: [{diag.min():.4f}, {diag.max():.4f}]")

        print(f"Kernel matrix computed: shape={self.kernel_matrix.shape}")

        return self.kernel_matrix

    def kernel_to_distance(self) -> np.ndarray:
        """
        Convert kernel matrix to distance matrix.

        Uses the formula: D_ij = sqrt(K_ii + K_jj - 2*K_ij)
        This gives a proper metric when K is a valid kernel.

        Returns:
            Distance matrix of shape (n_graphs, n_graphs)
        """
        if self.kernel_matrix is None:
            raise ValueError("Kernel matrix not computed. Call compute_kernel_matrix first.")

        print("Converting kernel to distance matrix...")

        K = self.kernel_matrix
        diag = np.diag(K)

        # D_ij = sqrt(K_ii + K_jj - 2*K_ij)
        D_squared = diag[:, np.newaxis] + diag[np.newaxis, :] - 2 * K
        D_squared = np.maximum(D_squared, 0)  # Handle numerical issues

        self.distance_matrix = np.sqrt(D_squared)

        print(f"Distance matrix computed: shape={self.distance_matrix.shape}")
        nonzero_distances = self.distance_matrix[self.distance_matrix > 1e-10]
        if len(nonzero_distances) > 0:
            print(f"  Min distance: {nonzero_distances.min():.4f}")
            print(f"  Max distance: {self.distance_matrix.max():.4f}")
            print(f"  Mean distance: {nonzero_distances.mean():.4f}")

        return self.distance_matrix

    def kmedoids(
        self,
        k: int,
        max_iter: int = 100,
        init: str = "kmeans++"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        K-medoids clustering using Lloyd-style alternating optimization.

        Note: This uses assign → recompute medoid steps, not PAM swap optimization.

        Args:
            k: Number of clusters/medoids
            max_iter: Maximum iterations
            init: Initialization method ("random" or "kmeans++")

        Returns:
            Tuple of (medoid_indices, cluster_labels)
        """
        if self.distance_matrix is None:
            raise ValueError("Distance matrix not computed. Call kernel_to_distance first.")

        np.random.seed(self.random_state)
        D = self.distance_matrix
        n = D.shape[0]

        print(f"\nRunning k-medoids with k={k}...")

        # Initialize medoids
        if init == "kmeans++":
            medoid_indices = self._kmedoids_plusplus_init(D, k)
        else:
            medoid_indices = np.random.choice(n, k, replace=False)

        best_medoids = medoid_indices.copy()
        best_cost = np.inf

        for iteration in tqdm(range(max_iter), desc=f"K-medoids (k={k})"):
            # Assign each point to nearest medoid
            distances_to_medoids = D[:, medoid_indices]
            labels = np.argmin(distances_to_medoids, axis=1)

            # Compute total cost (vectorized)
            cost = distances_to_medoids[np.arange(n), labels].sum()

            if cost < best_cost:
                best_cost = cost
                best_medoids = medoid_indices.copy()

            # Update medoids
            new_medoids = []
            for cluster_id in range(k):
                cluster_indices = np.where(labels == cluster_id)[0]

                if len(cluster_indices) == 0:
                    new_medoids.append(medoid_indices[cluster_id])
                    continue

                # Find point minimizing sum of distances within cluster
                cluster_distances = D[np.ix_(cluster_indices, cluster_indices)]
                total_distances = cluster_distances.sum(axis=1)
                best_local_idx = np.argmin(total_distances)
                new_medoids.append(cluster_indices[best_local_idx])

            new_medoids = np.array(new_medoids)

            # Check convergence
            if np.array_equal(np.sort(new_medoids), np.sort(medoid_indices)):
                print(f"  Converged at iteration {iteration}")
                break

            medoid_indices = new_medoids

        medoid_indices = best_medoids
        distances_to_medoids = D[:, medoid_indices]
        labels = np.argmin(distances_to_medoids, axis=1)

        # Compute final cost (vectorized)
        final_cost = distances_to_medoids[np.arange(n), labels].sum()
        print(f"  Final cost: {final_cost:.4f}")

        return medoid_indices, labels

    def _kmedoids_plusplus_init(self, D: np.ndarray, k: int) -> np.ndarray:
        """
        K-medoids++ initialization.

        Prevents duplicate medoids by setting probability to 0 for already chosen points.
        """
        n = D.shape[0]
        medoids = [np.random.randint(n)]

        for _ in range(1, k):
            min_distances = np.min(D[:, medoids], axis=1)
            probs = min_distances ** 2

            # Set probability to 0 for already-chosen medoids to prevent duplicates
            probs[medoids] = 0

            probs_sum = probs.sum()
            if probs_sum > 0:
                probs /= probs_sum
            else:
                # Fallback: choose uniformly from unchosen points
                probs = np.ones(n)
                probs[medoids] = 0
                probs /= probs.sum()

            next_medoid = np.random.choice(n, p=probs)
            medoids.append(next_medoid)

        return np.array(medoids)

    def save_results(
        self,
        k: int,
        medoid_indices: np.ndarray,
        labels: np.ndarray
    ) -> str:
        """Save clustering results to JSON and numpy files."""
        os.makedirs(self.output_dir, exist_ok=True)

        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))

        selected_scenarios = []
        for cluster_id, medoid_idx in enumerate(medoid_indices):
            filename = os.path.basename(self.graph_files[medoid_idx])
            selected_scenarios.append({
                "filename": filename,
                "graph_index": int(medoid_idx),
                "cluster_id": int(cluster_id),
                "cluster_size": int(cluster_sizes.get(cluster_id, 0))
            })

        selected_scenarios.sort(key=lambda x: x["cluster_id"])

        output = {
            "method": "wl_kernel_kmedoids",
            "k": k,
            "wl_iterations": self.wl_iterations,
            "normalize": self.normalize,
            "directed": self.directed,
            "selected_scenarios": selected_scenarios,
            "total_graphs": len(self.graph_files)
        }

        json_path = os.path.join(self.output_dir, f"selected_scenarios_k{k}.json")
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved selected scenarios to {json_path}")

        npy_path = os.path.join(self.output_dir, f"cluster_assignments_k{k}.npy")
        np.save(npy_path, labels)
        print(f"Saved cluster assignments to {npy_path}")

        return json_path

    def save_matrices(self) -> None:
        """Save kernel and distance matrices to disk."""
        os.makedirs(self.output_dir, exist_ok=True)

        if self.kernel_matrix is not None:
            kernel_path = os.path.join(self.output_dir, "wl_kernel_matrix.npy")
            np.save(kernel_path, self.kernel_matrix)
            print(f"Saved kernel matrix to {kernel_path}")

        if self.distance_matrix is not None:
            dist_path = os.path.join(self.output_dir, "wl_distance_matrix.npy")
            np.save(dist_path, self.distance_matrix)
            print(f"Saved distance matrix to {dist_path}")

    def load_matrices(self) -> bool:
        """Load cached kernel and distance matrices if they exist."""
        kernel_path = os.path.join(self.output_dir, "wl_kernel_matrix.npy")
        dist_path = os.path.join(self.output_dir, "wl_distance_matrix.npy")

        if os.path.exists(kernel_path) and os.path.exists(dist_path):
            print("Loading cached matrices...")
            self.kernel_matrix = np.load(kernel_path)
            self.distance_matrix = np.load(dist_path)
            print(f"Loaded kernel matrix: {self.kernel_matrix.shape}")
            print(f"Loaded distance matrix: {self.distance_matrix.shape}")
            return True

        return False

    def run(
        self,
        k_values: List[int] = [20, 100],
        use_cache: bool = True,
        limit: Optional[int] = None
    ) -> Dict[int, str]:
        """Run the full pipeline."""
        print("=" * 60)
        print("WL Kernel + K-Medoids Baseline")
        print("=" * 60)

        self.load_graphs(limit=limit)

        cached = False
        if use_cache:
            cached = self.load_matrices()
            if cached and self.kernel_matrix.shape[0] != len(self.graph_files):
                print("Cache mismatch, recomputing...")
                cached = False

        if not cached:
            self.compute_kernel_matrix()
            self.kernel_to_distance()
            self.save_matrices()

        results = {}
        for k in k_values:
            print(f"\n{'='*40}")
            print(f"Running k-medoids with k={k}")
            print(f"{'='*40}")

            medoids, labels = self.kmedoids(k, init="kmeans++")
            json_path = self.save_results(k, medoids, labels)
            results[k] = json_path

            unique, counts = np.unique(labels, return_counts=True)
            print(f"\nCluster size statistics:")
            print(f"  Min: {counts.min()}")
            print(f"  Max: {counts.max()}")
            print(f"  Mean: {counts.mean():.1f}")
            print(f"  Std: {counts.std():.1f}")

        print("\n" + "=" * 60)
        print("Done!")
        print("=" * 60)

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="WL Kernel + K-Medoids Baseline for Scenario Selection"
    )
    parser.add_argument(
        "--graphs_dir", type=str, default="graphs",
        help="Directory containing graph pickle files"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Output directory for results"
    )
    parser.add_argument(
        "--wl_iterations", type=int, default=3,
        help="Number of WL iterations"
    )
    parser.add_argument(
        "--k_values", type=int, nargs="+", default=[20, 100],
        help="K values for k-medoids clustering"
    )
    parser.add_argument(
        "--directed", action="store_true", default=True,
        help="Use directed WL kernel (default: True). Critical for DAG data."
    )
    parser.add_argument(
        "--undirected", dest="directed", action="store_false",
        help="Convert to undirected graphs (loses direction information)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of graphs (for testing)"
    )
    parser.add_argument(
        "--no_cache", action="store_true",
        help="Disable cache usage"
    )
    parser.add_argument(
        "--random_state", type=int, default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    baseline = WLKernelBaseline(
        graphs_dir=args.graphs_dir,
        output_dir=args.output_dir,
        wl_iterations=args.wl_iterations,
        directed=args.directed,
        random_state=args.random_state
    )

    baseline.run(
        k_values=args.k_values,
        use_cache=not args.no_cache,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
