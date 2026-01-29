# Directed WL + k-medoids baseline

This baseline represents each directed DAG with a Weisfeiler–Lehman (WL) subtree
feature map that separates IN/OUT neighborhoods and uses hashing to keep feature
size fixed. Graphs are L2-normalized and compared with cosine similarity. A
k-medoids clustering step selects representative graphs (medoids) as scenarios.

Outputs include:
- `selected_graphs_k{K}.json` with selected pickle filenames and cluster stats
- optional cached WL features for reuse

Run:
```bash
python src/kernels/kernel_baseline.py --graphs_dir data/sub20/graph
```

We represent each directed acyclic graph with a directed Weisfeiler–Lehman (WL) subtree feature map of height $h$: nodes are initialized by an attribute (or $\text{in}-\text{out}$ degree), then iteratively relabeled by the multiset of incoming and outgoing neighbor labels; at each iteration $t$ we count labels and hash them into a fixed $d$-dimensional vector using a stable MD5 bucketer (feature hashing). The resulting sparse matrix $X\in\mathbb{R}^{N\times d}$ is row‑normalized, and similarity is cosine: $K_{ij}=\langle x_i,x_j\rangle$, with distance $d_{ij}=\sqrt{2-2K_{ij}}$.

We select representatives via k‑medoids with k‑medoids++ initialization and Lloyd updates; for large $N$ we use a CLARA‑style scheme (subset clustering, lift medoids, full‑data assignment) followed by a lightweight full‑data refinement that evaluates a small candidate set per cluster. 

The end‑to‑end cost is linear in the number of graph edges for WL feature extraction and approximately $O(Nkd)$ for assignment plus $O(S^2k)$ for subset clustering (subset size $S$), avoiding full $N^2$ storage except when $N$ is small. 

[CITATION: WL] 
[CITATION: feature hashing] 
[CITATION: k‑medoids/CLARA]

