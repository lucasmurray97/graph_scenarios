import torch
from torch_geometric.data import Dataset
from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils import to_dense_adj
import glob
import networkx as nx
import pickle


class GraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        """
        This class is a Dataset for PyTorch Geometric that reads a list of pickle files
        and returns a PyG Graph object. The pickle files are NetworkX DiGraph objects and are
        sampled from the directory root. The class also stores the original ids of the nodes
        args: root: str, path to the directory containing the pickle files
        output: None
        """
        ids = {} # dictionary to store the ids of the files
        n = 0
        # iterate over the files in the directory
        for fname in glob.glob(f'{root}/*.pickle'):
            ids[n] = fname
            n += 1
        self.ids = ids

    def __len__(self):
        # return the number of files in the directory
        return len(self.ids)

    def __getitem__(self, idx):
        # load the pickle file and return a PyG Graph object corresponding to the idx file
        G = pickle.load(open(f'{self.ids[idx]}', 'rb'))
        for node in G.nodes():
            G.nodes[node]['original_ids'] = node - 1
        pyg_graph = from_networkx(G)
        pyg_graph.original_graph = torch.tensor(float(self.ids[idx].split("_")[1].split(".pickle")[0]))
        # List nodes in pyg_graph:
        pyg_graph.x = torch.tensor([1 for _ in range(len(G))], dtype=torch.float).unsqueeze(1)
        return pyg_graph
    
    
def reconstruct_matrix(graph_list):
    global_matrix = []
    for graph in graph_list:
        node_mapping = graph.original_ids
        mapped_edge_index = torch.tensor(
        [[node_mapping[int(i)] for i in edge_pair] for edge_pair in graph.edge_index.T],
            dtype=torch.long).T
        matrix = to_dense_adj(mapped_edge_index, max_num_nodes=400)[0]
        global_matrix.append(matrix)
    # We concatenate the list of matrices into a tensor:
    return torch.cat(global_matrix)
    
def generate_grid_edges(grid_size):
    edges = []
    for i in range(grid_size):
        for j in range(grid_size):
            node = i * grid_size + j
            neighbors = [
                (i - 1, j),  # up
                (i + 1, j),  # down
                (i, j - 1),  # left
                (i, j + 1),  # right
                (i - 1, j - 1),  # top-left diagonal
                (i - 1, j + 1),  # top-right diagonal
                (i + 1, j - 1),  # bottom-left diagonal
                (i + 1, j + 1),  # bottom-right diagonal
            ]
            for ni, nj in neighbors:
                if 0 <= ni < grid_size and 0 <= nj < grid_size:
                    neighbor_node = ni * grid_size + nj
                    edges.append((node, neighbor_node))
    return edges
    
