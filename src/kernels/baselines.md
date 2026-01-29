# Baselines

## Baseline 1: Directed WL + k-Medoids


This baseline selects a representative subset of wildfire scenarios from a large collection of directed acyclic graphs (DAGs) using a deterministic, non-parametric pipeline. It is meant as a strong, interpretable reference to compare against learned embeddings (e.g., VGAE/GraphVAE). The output is a set of **medoid graphs**—actual scenarios from the dataset—so the selected subset can be used directly in the downstream optimization model.

#### Directed WL features (hashed sparse vectors)

Each scenario graph $G_i=(V_i,E_i)$ is mapped to a sparse vector $x_i\in\mathbb{R}^d$ using a directed Weisfeiler–Lehman (WL) refinement. Nodes start with an initial label $\ell^{(0)}(v)$, taken from node attributes when available, or otherwise from a simple structural signature such as $(\deg^{-}(v),\deg^{+}(v))$. For iterations $t=1,\dots,h$, labels are updated by aggregating predecessor and successor labels separately to preserve directionality:
$$
\ell^{(t)}(v) = \text{hash}\Big(\ell^{(t-1)}(v), \{\!\{\ell^{(t-1)}(u): u \in \text{Pred}(v)\}\!\}, \{\!\{\ell^{(t-1)}(u): u \in \text{Succ}(v)\}\!\}\Big).
$$
Graph-level features are then built by counting hashed node labels across nodes and iterations (feature hashing). Vectors are L2-normalized, and scenario similarity is measured with cosine similarity $K_{ij}=\langle x_i,x_j\rangle$; clustering uses the equivalent distance $d_{ij}=\sqrt{2-2K_{ij}}$.

#### Scenario selection via k-medoids (CLARA for large $N$)

Given distances $d_{ij}$, we select $k$ representative scenarios using k-medoids, which minimizes total distance to the chosen medoids under nearest-medoid assignment, with the constraint that medoids are real dataset graphs. For large $N$, we use a CLARA-style approximation: run k-medoids on several random subsets, evaluate candidate medoids on the full dataset via nearest-medoid assignment, keep the best set, and optionally refine with full-data assignments. The baseline outputs the selected medoid indices (or graph IDs/filenames) and a cluster label for each graph.

## Baseline 2: Scenario Reduction Fast Forward Selection



## Implementation Notes (Project Use)
The script consumes pickled NetworkX DAGs (e.g., `data/sub20/graphs/graph_*.pickle`) and writes selection outputs to JSON, including the chosen medoid filenames and cluster statistics (default location: `src/kernels/outputs/final_selected_graphs_k{K}.json`). To run:
```bash
python src/kernels/kernel_baseline.py --graphs_dir data/sub20/graphs --k_values 20 100
```
