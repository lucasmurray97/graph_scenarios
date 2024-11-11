import torch
from torch_geometric.data import Dataset
from torch_geometric.utils.convert import from_networkx
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

    def len(self):
        # return the number of files in the directory
        return len(self.ids)

    def get(self, idx):
        # load the pickle file and return a PyG Graph object corresponding to the idx file
        G = pickle.load(open(f'{self.ids[idx]}', 'rb'))
        pyg_graph = from_networkx(G)
        pyg_graph.original_ids = torch.tensor(list(G.nodes()), dtype=torch.long)
        return pyg_graph
