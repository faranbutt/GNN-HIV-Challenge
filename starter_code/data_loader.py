# working/Molecular Graph/starter_code/data_loader.py
import torch
import pickle
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops

class HIVPyGDataset(Dataset):
    def __init__(self, csv_file, struct_file, feat_file, is_test=False):
        self.df = pd.read_csv(csv_file)
        self.is_test = is_test
        with open(struct_file, 'rb') as f: self.structs = pickle.load(f)
        with open(feat_file, 'rb') as f: self.feats = pickle.load(f)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        gid = row['graph_id']
        x = torch.from_numpy(self.feats[gid]).float()
        edges = self.structs[gid]['edge_list']
        edges = edges + [(j, i) for (i, j) in edges]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        if self.is_test:
            return Data(x=x, edge_index=edge_index, graph_id=gid)
        y = torch.tensor([row['target']], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, y=y)


def get_dataloaders(batch_size=32):
    train_ds = HIVPyGDataset('data/train.csv', 'data/graph_structures.pkl', 'data/node_features.pkl')
    test_ds = HIVPyGDataset('data/test.csv', 'data/graph_structures.pkl', 'data/node_features.pkl', is_test=True)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
