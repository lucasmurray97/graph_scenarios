import numpy as np
import networkx as nx
from networkx import DiGraph
import pickle
import glob


def digraph_from_messages(afile):
    """Not checking if file exists or if size > 0
    This is done previously on read_files
    """
    data = np.loadtxt(
        afile, delimiter=",", dtype=[("i", np.int32), ("j", np.int32), ("time", np.int16)], usecols=(0, 1, 2)
    )
    root = data[0][0]  # checkar que el primer valor del message sea el punto de ignición
    G = nx.DiGraph()
    G.add_weighted_edges_from(data)
    return G, root

def generate_graphs(directory):
    for fname in glob.glob(f'{directory}/*.csv'):
        scn_n = fname.split("MessagesFile")[1].split(".")[0]
        try:
            G, root = digraph_from_messages(fname)
            pickle.dump(G, open(f'{directory}/../graphs/graph_{scn_n}.pickle', 'wb+'))
        except:
            print("No ignition")
        


generate_graphs("../data/sub20/Messages")




