import torch
import torch_geometric as pyg


class GATHead(pyg.nn.MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1, **kwargs):
        super().__init__(aggr="add")
        self.C = out_channels
        self.lin = torch.nn.Linear(in_channels, self.C, bias=False)
        self.attn = torch.nn.Sequential(
            torch.nn.Linear(2 * self.C, 1, bias=False),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x, edge_index):
        # x: [N, in_channels]
        # edge_index: [2, E]
        # Add self loops (save for last)
        edge_index, _ = pyg.utils.add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)  # [N, C]
        return self.propagate(edge_index, x=x)  # [N, C]

    def message(self, x_j, x_i, edge_index_i):
        # x_i, x_j: [E, C]
        # edge_index_i: [E]
        # if flow is "source_to_target", x_j is the source node and x_i is the target node
        attn_input = torch.cat([x_j, x_i], dim=-1)  # [E, 2C]
        attn_logits = self.attn(attn_input)  # [E, 1]
        alpha = pyg.utils.softmax(attn_logits, edge_index_i)  # [E, 1]
        return alpha * x_j  # [E, C]


class GATLayer(pyg.nn.MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, heads: int, concat=False, dropout=0,
                 **kwargs):
        super().__init__(aggr="add", node_dim=0)
        self.H = heads
        self.C = out_channels
        self.concat = concat
        self.lin = torch.nn.Linear(in_channels, self.C * self.H, bias=False)
        self.attn_w = torch.nn.Parameter(torch.empty(self.H, 2 * self.C))
        self.attn_relu = torch.nn.LeakyReLU(negative_slope=0.2)
        self.dropout = torch.nn.Dropout(dropout)
        self.reset_parameters()

    def forward(self, x, edge_index):
        # x: [N, in_channels]
        # edge_index: [2, E]
        # Add self loops (save for last)
        edge_index, _ = pyg.utils.add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x).view(-1, self.H, self.C)  # [N, H, C]
        out = self.propagate(edge_index, x=x)  # [N, H, C]
        if self.concat:
            return out.view(x.size(1), -1)  # [N, H * C]
        return out.mean(dim=1)  # [N, C]

    def message(self, x_j, x_i, edge_index_i):
        # x_i, x_j: [E, H, C]
        # edge_index_i: [E]
        # if flow is "source_to_target", x_j is the source node and x_i is the target node
        attn_input = torch.cat([x_j, x_i], dim=-1)  # [E, H, 2C]
        attn_logits = self.attn_relu((self.attn_w * attn_input).sum(dim=-1, keepdim=True))
        # ^-- [E, H, 1]
        alpha = self.dropout(pyg.utils.softmax(attn_logits, edge_index_i))  # [E, H, 1]
        return alpha * x_j  # [E, H, C]

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        pyg.nn.inits.glorot(self.attn_w)
        # pyg.inits.zeros(self.bias)


class GNN(torch.nn.Module):
    def __init__(self, n_layers: int, input_features: int, channels: int = 64, heads: int = 4,
                 vnode: bool = False, conv_layer: pyg.nn.MessagePassing = pyg.nn.GATv2Conv):
        super(GNN, self).__init__()
        self.vnode = vnode
        self.vnode_transform = pyg.transforms.VirtualNode() if vnode else None
        self.input_features = input_features
        self.channels = channels
        self.initial_proj = (
            torch.nn.Linear(input_features, channels, bias=False) if input_features != channels
            else torch.nn.Identity()
        )
        self.convs = torch.nn.ModuleList([
            conv_layer(self.channels, self.channels, heads, concat=False, dropout=0.1)
            for _ in range(n_layers)
        ])
        self.norms = torch.nn.ModuleList(
            [pyg.nn.LayerNorm(self.channels) for _ in range(n_layers)]
        )
        self.head = torch.nn.Linear(self.channels, 1)

    def forward(self, data):
        # add vnode
        if self.vnode:
            # TODO won't worked with batched data
            data = self.vnode_transform(data)
        x, edge_index = data.x, data.edge_index
        x = self.initial_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            x = torch.selu(norm(conv(x, edge_index)))
        # predict on virtual node embedding
        if self.vnode:
            vnodes = x[data.ptr[1:] - 1]  # last node of each graph
            return self.head(vnodes).squeeze(dim=-1)
        return self.head(pyg.nn.global_mean_pool(x, data.batch)).squeeze(dim=-1)


def parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_gat_head():
    N = 10
    in_size = 4
    C = 16
    edge_index = pyg.utils.random.erdos_renyi_graph(N, 0.2, directed=False)
    x = torch.randn(N, in_size)
    layer = GATHead(in_size, C)
    with torch.no_grad():
        out = layer(x, edge_index)
    assert out.size() == (N, C)
    # compare with pyg
    gat = pyg.nn.GATConv(in_size, C, heads=1, bias=False)
    with torch.no_grad():
        gat.lin.weight.copy_(layer.lin.weight)
        # map attn_src, attn_dst to layer.attn.weight[:16], layer.attn.weight[16:]
        gat.att_src.copy_(layer.attn[0].weight[:, :C].unsqueeze(0))
        gat.att_dst.copy_(layer.attn[0].weight[:, C:].unsqueeze(0))
        out_pyg = gat(x, edge_index)
    assert torch.allclose(out, out_pyg)


def test_gat_layer():
    N = 10
    in_size = 4
    C = 16
    H = 8
    edge_index = pyg.utils.random.erdos_renyi_graph(N, 0.2, directed=False)
    x = torch.randn(N, in_size)
    layer = GATLayer(in_size, C, H)
    with torch.no_grad():
        out = layer(x, edge_index)
    assert out.size() == (N, C)
    # compare with pyg
    gat = pyg.nn.GATConv(in_size, C, heads=H, bias=False, concat=False)
    with torch.no_grad():
        gat.lin.weight.copy_(layer.lin.weight)
        # map attn_src, attn_dst to layer.attn.weight[:16], layer.attn.weight[16:]
        gat.att_src.copy_(layer.attn_w[:, :C].unsqueeze(0))
        gat.att_dst.copy_(layer.attn_w[:, C:].unsqueeze(0))
        out_pyg = gat(x, edge_index)
    assert torch.allclose(out, out_pyg)


if __name__ == "__main__":
    pyg.seed_everything(42)
    test_gat_head()
    test_gat_layer()
    print("All tests passed ✅")
