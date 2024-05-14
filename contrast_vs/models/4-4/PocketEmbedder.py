import torch
from torch_geometric.nn import GATv2Conv, Sequential
from torch_geometric.nn.models import MLP
from torch_geometric.nn.aggr import AttentionalAggregation, MaxAggregation
from torch.nn import ReLU, LayerNorm, Dropout
from torch_geometric.utils import to_undirected
from torch_geometric.nn import knn_graph
from torch_geometric.nn.pool import knn
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

    def __init__(self, feature_dim, hidden_dim, out_dim, atom_k=10, vox_k=20):
        super().__init__()
        self.atom_edge_MLP = MLP([1, 32, 1])
        self.vox_edge_MLP = MLP([1, 32, 1])
        self.atom_k = atom_k
        self.vox_k = vox_k

        self.heads = 4
        self.block_depth = 2
        self.vox_block_depth = 4

        self.linear_in = torch.nn.Linear(feature_dim, hidden_dim * self.heads)
        self.vox_init = torch.nn.Linear(1, hidden_dim * self.heads)
        self.atom_out = torch.nn.Linear(
            4 * hidden_dim * self.heads, hidden_dim * self.heads
        )

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

        self.res_block4 = Sequential(
            "x, edge_index, edge_weights",
            [
                (
                    ResLayer(hidden_dim, heads=self.heads),
                    "x, edge_index, edge_weights -> x",
                )
                for _ in range(self.vox_block_depth)
            ],
        )

        self.out_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * self.heads, out_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(out_dim, out_dim),
        )

        self.act = ReLU()

    def forward(self, data):
        x, pos, vox_mask, batch = (data.x, data.pos, data.vox_mask, data.batch)

        # -- STEP 1 - ATOM MODEL ----------|
        atom_x = x[~vox_mask]
        atom_pos = pos[~vox_mask]
        atom_batch = batch[~vox_mask]

        prox_atom_indices = torch.unique(knn(atom_pos, pos[vox_mask], 30)[1])

        atom_x = atom_x[prox_atom_indices]
        atom_pos = atom_pos[prox_atom_indices]
        atom_batch = atom_batch[prox_atom_indices]

        atom_ei = knn_graph(atom_pos, self.atom_k, atom_batch)

        # Generate voxel embed values based on neighboring protein atoms
        atom_edge_weights = torch.norm(
            atom_pos[atom_ei[0]] - atom_pos[atom_ei[1]], dim=1
        ).unsqueeze(dim=1)

        atom_edge_weights = self.act(self.atom_edge_MLP(atom_edge_weights)).squeeze()

        h0 = self.linear_in(atom_x.float())

        h1 = self.res_block1(h0, atom_ei, atom_edge_weights)
        h2 = self.res_block2(h1, atom_ei, atom_edge_weights)
        h3 = self.res_block3(h2, atom_ei, atom_edge_weights)

        atom_out = self.atom_out(torch.hstack((h0, h1, h2, h3)))
        # --------------------------------|

        # -- STEP 2 - VOX MODEL ----------|
        vox_pos = pos[vox_mask]
        vox_batch = batch[vox_mask]

        vox_ei = knn(atom_pos, vox_pos, self.vox_k, atom_batch, vox_batch)

        # Swap edge index order so messages are passed FROM atoms TO voxels
        vox_ei = torch.vstack((vox_ei[1], vox_ei[0] + atom_out.size(0)))

        vox_block = self.vox_init(torch.ones(vox_pos.size(0)).unsqueeze(1).to(x.device))
        vox_x = torch.vstack((atom_out, vox_block))

        vox_graph_pos = torch.vstack((atom_pos, vox_pos))

        vox_edge_weights = torch.norm(
            vox_graph_pos[vox_ei[0]] - vox_graph_pos[vox_ei[1]], dim=1
        ).unsqueeze(dim=1)

        vox_edge_weights = self.act(self.vox_edge_MLP(vox_edge_weights)).squeeze()

        v_h = self.res_block4(vox_x, vox_ei, vox_edge_weights)
        v_out = self.out_layer(v_h)

        return v_out[atom_out.size(0) :]
