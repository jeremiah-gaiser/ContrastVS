import re
from utils import to_onehot, get_k_hop_neighborhoods
import torch
import numpy as np
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem, AddHs
from torch_geometric.utils import subgraph, to_undirected

mol_atom_vocab = [
    "As",
    "B",
    "Be",
    "Br",
    "C",
    "Cl",
    "Co",
    "Cu",
    "F",
    "Fe",
    "H",
    "I",
    "Ir",
    "Mg",
    "N",
    "O",
    "Os",
    "P",
    "Pt",
    "Re",
    "Rh",
    "Ru",
    "S",
    "Sb",
    "Se",
    "Si",
    "Te",
    "V",
    "Zn",
]

H_count_vocab = [0, 1, 2, 3]
bond_type_vocab = ["AROMATIC", "DOUBLE", "K", "SINGLE", "TRIPLE"]


def mol2graph(m, get_pos=False, k=1):
    k_hop_vocab = [x + 1 for x in range(k)]

    try:
        m = AddHs(m)
    except:
        return None

    atom_count = m.GetNumAtoms()
    am = Chem.GetAdjacencyMatrix(m)

    if get_pos:
        conformer = m.GetConformer()
        pos = [[] for _ in range(atom_count)]

    x = [[] for _ in range(atom_count)]
    atom_H_count = [0 for _ in range(atom_count)]
    heavy_atoms = [0 for _ in range(atom_count)]

    edge_index = [[], []]
    edge_attr = []

    for b in m.GetBonds():
        edge_index[0].append(b.GetBeginAtomIdx())
        edge_index[1].append(b.GetEndAtomIdx())

        edge_attr.append(to_onehot(b.GetBondType().name, bond_type_vocab))

    for atom in m.GetAtoms():
        a_i = atom.GetIdx()

        if get_pos:
            ap = conformer.GetAtomPosition(a_i)
            pos[a_i] = [ap.x, ap.y, ap.z]

        x[a_i] = to_onehot(atom.GetSymbol(), mol_atom_vocab)

        if atom.GetSymbol() == "H":
            atom_H_count[list(am[a_i]).index(1)] += 1
        else:
            heavy_atoms[a_i] = 1

    for a_i in range(atom_count):
        x[a_i] += to_onehot(atom_H_count[a_i], H_count_vocab)

    heavy_atoms = torch.where(torch.tensor(heavy_atoms) == 1)[0]

    x = torch.tensor(x)[heavy_atoms]

    edge_index, edge_attr = subgraph(
        heavy_atoms,
        torch.tensor(edge_index),
        torch.tensor(edge_attr),
        relabel_nodes=True,
    )

    edge_index, edge_attr = to_undirected(edge_index, edge_attr)

    # For now, throw away bond type information on edges.
    final_edge_index = torch.tensor([[], []])
    final_edge_attr = []

    for atom_idx in torch.unique(edge_index.flatten()):
        for k_i, k_neighborhood in enumerate(
            get_k_hop_neighborhoods(atom_idx.item(), k, edge_index)
        ):
            final_edge_attr += [
                to_onehot(k_i + 1, k_hop_vocab) for _ in range(k_neighborhood.size(0))
            ]
            final_edge_index = torch.hstack(
                (
                    final_edge_index,
                    torch.vstack(
                        (atom_idx.repeat(k_neighborhood.size(0)), k_neighborhood)
                    ),
                )
            )

    final_edge_attr = torch.tensor(final_edge_attr)

    final_edge_index, final_edge_attr = to_undirected(
        final_edge_index, final_edge_attr, reduce="max"
    )

    g = Data(x=x, edge_index=final_edge_index, edge_attr=final_edge_attr)

    if get_pos:
        g.pos = torch.tensor(pos)[heavy_atoms]

    return g
