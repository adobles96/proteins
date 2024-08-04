from datasets import load_dataset
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def get_dataset() -> list[Data]:
    dataset_hf = load_dataset("graphs-datasets/PROTEINS")
    return [
        Data(
            x=torch.tensor(graph["node_feat"]),
            edge_index=torch.tensor(graph["edge_index"]),
            y=torch.tensor(graph["y"]),
        ) for graph in dataset_hf["train"]
    ]


def get_dataloaders(train_size, val_size):
    test_size = 1 - train_size - val_size
    assert train_size + val_size + test_size == 1
    pass
