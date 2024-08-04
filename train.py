import argparse

from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import VirtualNode
from tqdm import tqdm

from model import GNN, parameter_count, GATLayer


TRAIN_SIZE = 0.9
N_FEATS = 3


class ProteinDset(torch.utils.data.Dataset):
    def __init__(self, add_vnodes: bool = False):
        transform = VirtualNode() if add_vnodes else torch.nn.Identity()
        self.protein_graphs = [
            transform(
                Data(
                    x=torch.tensor(graph["node_feat"]),
                    edge_index=torch.tensor(graph["edge_index"]),
                    y=torch.tensor(graph["y"], dtype=torch.float),
                )
            ) for graph in load_dataset("graphs-datasets/PROTEINS")["train"]
        ]

    def __getitem__(self, idx):
        return self.protein_graphs[idx]

    def __len__(self):
        return len(self.protein_graphs)


def eval(model, val_loader, loss_fn, device):
    model.eval()
    val_loss = 0
    val_acc = 0
    num_samples = len(val_loader.dataset)
    with torch.no_grad():
        for data in val_loader:
            data.to(device)
            pred = model(data)
            val_acc += (torch.sigmoid(pred) > 0.5).int().eq(data.y).sum().item()
            val_loss += loss_fn(model(data), data.y).item() * val_loader.batch_size
        return val_loss / num_samples, val_acc / num_samples


def train(model, train_loader, val_loader, optimizer, n_epochs, device):
    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    loss_fn = torch.nn.BCEWithLogitsLoss()
    num_samples = len(train_loader.dataset)
    epoch_pbar = tqdm(range(n_epochs), unit='epoch', position=1)
    for _ in epoch_pbar:
        model.train()
        n_correct = 0
        cum_loss = 0
        batch_pbar = tqdm(train_loader, unit='batch', position=2, leave=False)
        for i, data in enumerate(batch_pbar, start=1):
            data.to(device)
            optimizer.zero_grad()
            pred = model(data)
            loss = loss_fn(pred, data.y)
            loss.backward()
            optimizer.step()
            cum_loss += loss.detach().item() * train_loader.batch_size
            n_correct += (torch.sigmoid(pred.detach()) > 0.5).int().eq(data.y).sum().item()
            batch_pbar.set_description(f'Avg Loss: {cum_loss / (i * train_loader.batch_size):.4f}')
        train_loss.append(cum_loss / num_samples)
        train_acc.append(n_correct / num_samples)
        l, a = eval(model, val_loader, loss_fn, device)
        val_loss.append(l)
        val_acc.append(a)
        epoch_pbar.set_description(f'Train Acc: {n_correct / num_samples:.1%} -- Val Acc: {a:.1%}')
    return train_loss, train_acc, val_loss, val_acc


def plot_results(tloss, tacc, vloss, vacc):
    # Plotting training and validation loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(tloss, label='Training Loss')
    plt.plot(vloss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    # Plotting training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(tacc, label='Training Accuracy')
    plt.plot(vacc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    # Adjusting subplot spacing
    plt.tight_layout()

    # Display the plot
    plt.show()


def run_cv(args, k=10):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device: ', device)
    stats = pd.DataFrame(columns=['train_loss', 'train_acc', 'val_loss', 'val_acc'])
    dataset = ProteinDset(args.vnodes)
    shuffled_order = torch.randperm(len(dataset)).detach()
    fold_size = len(dataset) // k
    for i in tqdm(range(k), position=0, unit='fold'):
        val_indices = shuffled_order[fold_size * i: fold_size * (i + 1)]
        train_indices = shuffled_order[~torch.isin(shuffled_order, val_indices)]
        val_data = torch.utils.data.Subset(dataset, val_indices)
        train_data = torch.utils.data.Subset(dataset, train_indices)
        # loaders
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                  num_workers=0)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

        model = GNN(
            n_layers=args.n_layers, input_features=N_FEATS, channels=args.channels,
            heads=args.heads, vnode=args.vnodes, conv_layer=pyg.nn.GATv2Conv
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        results = train(model, train_loader, val_loader, optimizer, args.epochs, device)
        stats.loc[len(stats)] = [min(results[0]), max(results[1]), min(results[2]), max(results[3])]
    print(f'Mean Train Loss: {stats["train_loss"].mean():.4f}')
    print(f'Mean Validation Loss: {stats["val_loss"].mean():.4f}')
    print(f'Mean Train Accuracy: {stats["train_acc"].mean():.1%}')
    print(f'Mean Validation Accuracy: {stats["val_acc"].mean():.1%}')


def single_fold(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device: ', device)

    # loaders
    dataset = ProteinDset(args.vnodes)
    train_data, val_data = torch.utils.data.random_split(dataset, [TRAIN_SIZE, 1 - TRAIN_SIZE])
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = GNN(
        n_layers=args.n_layers, input_features=N_FEATS, channels=args.channels, heads=args.heads,
        vnode=args.vnodes, conv_layer=GATLayer
    ).to(device)
    # model = torch.compile(model)
    print(f'Initialized GNN with {parameter_count(model):,} parameters.')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    results = train(model, train_loader, val_loader, optimizer, args.epochs, device)
    plot_results(*results)
    print(f'Best Train Loss: {min(results[0]):.4f}')
    print(f'Best Validation Loss: {min(results[2]):.4f}')
    print(f'Best Train Accuracy: {max(results[1]):.1%}')
    print(f'Best Validation Accuracy: {max(results[3]):.1%}')


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning Rate')
    parser.add_argument('-c', '--channels', type=int, default=32, help='Channels for GAT layers')
    parser.add_argument('-n', '--n_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--heads', type=int, default=8,
                        help='Number of attn heads for GAT layers')
    parser.add_argument('-v', '--vnodes', action='store_true', default=False,
                        help='Whether to use virtual nodes')
    parser.add_argument('-k', '--k_fold', type=int, default=1, help='How many CV folds to run')
    args = parser.parse_args()
    pyg.seed.seed_everything(42)

    if args.k_fold == 1:
        single_fold(args)
    else:
        run_cv(args)
