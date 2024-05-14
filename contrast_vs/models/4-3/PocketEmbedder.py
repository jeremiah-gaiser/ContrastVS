import torch
from torch_geometric.nn import GATv2Conv, Sequential
from torch_geometric.nn.models import MLP
from torch_geometric.nn.aggr import AttentionalAggregation, MaxAggregation
from torch.nn import ReLU, LayerNorm, Dropout
from torch_geometric.utils import to_undirected
from copy import deepcopy
import math
import sys


class ResLayer(torch.nn.Module):
    def __init__(self, dims, act=ReLU, dr=0.25, heads=6, first=False):
        super().__init__()
        self.act = act()
        self.first = first

        if self.first:
            self.layer = GATv2Conv(dims, dims, edge_dim=1, heads=heads)
        else:
            self.layer = GATv2Conv(dims * heads, dims, edge_dim=1, heads=heads)

        self.dropout = Dropout(dr)

    def forward(self, x, edge_index, edge_weights):
        if self.first:
            x = self.layer(x, edge_index, edge_weights)
        else:
            x = x + self.layer(x, edge_index, edge_weights)

        x = self.act(x)
        return self.dropout(x)
        # return x


class PocketEmbedder(torch.nn.Module):
    """
    Encodes voxels (pocket space points connected to proximal protein atoms) into 'out_dim'-dimensional encoding.
    """

    def __init__(self, feature_dim, hidden_dim, out_dim):
        super().__init__()

        self.edge_MLP1 = MLP([1, 32, 1])

        self.heads = 4
        self.block_depth = 2

        self.linear_in = torch.nn.Linear(feature_dim, hidden_dim * self.heads)
        # self.pre_block = ResLayer(hidden_dim, heads=self.heads, first=True)

        self.res_block1 = Sequential(
            "x, edge_index, edge_weights",
            [
                (
                    ResLayer(hidden_dim, heads=self.heads),
                    "x, edge_index, edge_weights -> x",
                )
                for _ in range(self.block_depth)
            ],
        )

        self.res_block2 = Sequential(
            "x, edge_index, edge_weights",
            [
                (
                    ResLayer(hidden_dim, heads=self.heads),
                    "x, edge_index, edge_weights -> x",
                )
                for _ in range(self.block_depth)
            ],
        )

        self.res_block3 = Sequential(
            "x, edge_index, edge_weights",
            [
                (
                    ResLayer(hidden_dim, heads=self.heads),
                    "x, edge_index, edge_weights -> x",
                )
                for _ in range(self.block_depth)
            ],
        )

        self.aggregate_voxels1 = AttentionalAggregation(
            MLP([hidden_dim * self.heads, 1], act="relu"),
            torch.nn.Linear(hidden_dim * self.heads, hidden_dim * self.heads),
        )

        self.aggregate_voxels2 = AttentionalAggregation(
            MLP([hidden_dim * self.heads, 1], act="relu"),
            torch.nn.Linear(hidden_dim * self.heads, hidden_dim * self.heads),
        )

        self.aggregate_voxels3 = AttentionalAggregation(
            MLP([hidden_dim * self.heads, 1], act="relu"),
            torch.nn.Linear(hidden_dim * self.heads, hidden_dim * self.heads),
        )

        self.out_layer = torch.nn.Sequential(
            torch.nn.Linear(3 * hidden_dim * self.heads, out_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(out_dim, out_dim),
        )

        self.act = ReLU()

    def forward(self, data):
        x, pos, edge_index, batch = (
            data.x,
            data.pos,
            data.edge_index,
            data.sub_batch_index,
        )

        # Generate voxel embed values based on neighboring protein atoms
        edge_weights = torch.norm(
            pos[edge_index[0]] - pos[edge_index[1]], dim=1
        ).unsqueeze(dim=1)

        edge_weights = self.act(self.edge_MLP1(edge_weights)).squeeze()

        h = self.linear_in(x)

        h = self.res_block1(h, edge_index, edge_weights)
        o1 = self.aggregate_voxels1(h, index=batch)

        h = self.res_block2(h, edge_index, edge_weights)
        o2 = self.aggregate_voxels2(h, index=batch)

        h = self.res_block3(h, edge_index, edge_weights)
        o3 = self.aggregate_voxels3(h, index=batch)

        enc = self.out_layer(torch.hstack((o1, o2, o3)))

        return h, enc
