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
python src/kernels/kernel_baseline.py --graphs_dir graphs --out_dir outputs
```
