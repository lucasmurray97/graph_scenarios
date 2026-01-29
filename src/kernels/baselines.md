# Baselines

## Baseline 1: Directed WL + k-Medoids
This baseline provides a deterministic, non-parametric pipeline for scenario selection from directed acyclic graphs (DAGs). It serves as a reference point for evaluating learned approaches in this project (e.g., GraphVAE/VGAE latent-space clustering) by offering a fast, interpretable selector with minimal tuning.

Method. Let a dataset of DAGs be denoted by ${G_i}_{i=1}^N$, with $G_i=(V_i,E_i)$. Each node $v \in V_i$ is initialized with a label $\ell^{(0)}(v)$, taken from a node attribute when available, or otherwise from the structural signature $\text{in}(v)$ and $\text{out}(v)$ degrees. For $t=1,\dots,h$, labels are updated by combining the current label with multisets of incoming and outgoing neighbor labels, i.e.,
$$
\ell^{(t)}(v) = \text{hash}\Big(\ell^{(t-1)}(v), \{\ell^{(t-1)}(u): u \in \text{Pred}(v)\}, \{\ell^{(t-1)}(u): u \in \text{Succ}(v)\}\Big).
$$
At each iteration $t$, label counts are accumulated and mapped into a fixed-dimensional vector via feature hashing, yielding a sparse representation $x_i \in \mathbb{R}^d$ for each graph. The vectors are L2-normalized and compared with cosine similarity $K_{ij}=\langle x_i, x_j\rangle$, with cosine distance $d_{ij}=\sqrt{2-2K_{ij}}$.

Scenario selection. Given $k$, the algorithm selects indices $M \subset \{1,\dots,N\}$ to minimize the sum of distances from each graph to its assigned medoid in cosine-distance space. The implementation uses k-medoids++ initialization and Lloyd-style updates; for large $N$, it switches to a CLARA-style approximation that clusters subsets, lifts medoids to the full set, and refines them using a lightweight full-data polishing step based on distances to current medoids.

Implementation notes (project use). The script consumes pickled NetworkX DAGs (e.g., `data/sub20/graphs/graph_*.pickle`) and writes selection outputs to JSON, including the chosen medoid filenames and cluster statistics (default location: `src/kernels/outputs_final/final_selected_graphs_k{K}.json`). How to run:
```bash
python src/kernels/kernel_baseline.py --graphs_dir data/sub20/graphs --k_values 20 100
```

## Baseline 2: [Title TBD]
Placeholder for the second baseline (in progress).
