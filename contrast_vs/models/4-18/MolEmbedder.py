import torch
from torch_geometric.nn import (
    GATv2Conv,
    Sequential as PyG_Sequential,
)
from torch_geometric.nn.norm import BatchNorm

from torch.nn import Sequential, ReLU, Linear
from torch_geometric.nn.aggr import (
    AttentionalAggregation,
    MaxAggregation,
    MeanAggregation,
)
from torch_geometric.utils import subgraph
from torch_geometric.nn.models import MLP


class ResLayer(torch.nn.Module):
    def __init__(self, dims, act=ReLU, dr=0.25, heads=6, first=False):
        super().__init__()
        self.act = act()
        self.layer = Linear(dims, dims)
        self.first = first

        if self.first:
            self.layer = GATv2Conv(dims, dims, heads=heads)
        else:
            self.layer = GATv2Conv(dims * heads, dims, heads=heads)

        self.dropout = torch.nn.Dropout(dr)

    def forward(self, x, edge_index):
        if self.first:
            x = self.layer(x, edge_index)
        else:
            x = x + self.layer(x, edge_index)

        x = self.act(x)

        return self.dropout(x)
        # return x


class MolEmbedder(torch.nn.Module):
    """
    Generates molecular atom embeddings.
    Replaces 'substructure nodes' with single 'context node', if provided
    """

    def __init__(self, feature_dim, hidden_dim, out_dim, **kwargs):
        super().__init__()

        self.heads = 4
        self.block_depth = 2

        self.projection_layer = torch.nn.Linear(feature_dim, hidden_dim * self.heads)

        # self.pre_block = ResLayer(hidden_dim, heads=self.heads, first=True)

        self.res_block1 = PyG_Sequential(
            "x, edge_index",
            [
                (ResLayer(hidden_dim, heads=self.heads), "x, edge_index -> x")
                for _ in range(self.block_depth)
            ],
        )

        self.res_block2 = PyG_Sequential(
            "x, edge_index",
            [
                (ResLayer(hidden_dim, heads=self.heads), "x, edge_index -> x")
                for _ in range(self.block_depth)
            ],
        )

        self.res_block3 = PyG_Sequential(
            "x, edge_index",
            [
                (ResLayer(hidden_dim, heads=self.heads), "x, edge_index -> x")
                for _ in range(self.block_depth)
            ],
        )

        self.out_layer = Sequential(
            torch.nn.Linear(3 * hidden_dim * self.heads, out_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(out_dim, out_dim),
        )

        self.act = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = (data.x, data.edge_index, data.batch)

        h = self.projection_layer(x.float())

        h1 = self.res_block1(h, edge_index)
        h2 = self.res_block2(h1, edge_index)
        h3 = self.res_block3(h2, edge_index)

        enc = self.out_layer(torch.hstack((h1, h2, h3)))

        return enc
