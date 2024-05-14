import re
from utils import to_onehot
import torch
from torch_geometric.data import Data

res_vocab = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "VOX",
]

atom_vocab = [
    "C",
    "CA",
    "CB",
    "CD",
    "CD1",
    "CD2",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "CG",
    "CG1",
    "CG2",
    "CH2",
    "CZ",
    "CZ2",
    "CZ3",
    "N",
    "ND1",
    "ND2",
    "NE",
    "NE1",
    "NE2",
    "NH1",
    "NH2",
    "NZ",
    "O",
    "OD1",
    "OD2",
    "OE1",
    "OE2",
    "OG",
    "OG1",
    "OH",
    "OXT",
    "SD",
    "SG",
    "VOX",
]


def vox_onehot():
    return torch.tensor(to_onehot("VOX", atom_vocab) + to_onehot("VOX", res_vocab))


def get_labels(node):
    feature_labels = (None, None)

    atom_onehot = node[: len(atom_vocab)]
    res_onehot = node[len(atom_vocab) :]

    for i in range(len(atom_onehot)):
        if atom_onehot[i] == 1:
            feature_labels[0] = atom_vocab[i]
            break

    for i in range(len(res_onehot)):
        if res_onehot[i] == 1:
            feature_labels[1] = res_vocab[i]
            break

    return feature_labels


def featurize_pdb_line(l):
    atom_name = (12, 16)
    res_name = (17, 20)
    x_pos = (30, 38)
    y_pos = (38, 46)
    z_pos = (46, 54)

    onehot = []
    pos = []

    for f, v in zip([atom_name, res_name], [atom_vocab, res_vocab]):
        feature_val = l[f[0] : f[1]].strip()
        onehot += to_onehot(feature_val, v)

    for p in [x_pos, y_pos, z_pos]:
        pos_val = float(l[p[0] : p[1]].strip())
        pos.append(pos_val)

    return onehot, pos


def pdb2graph(pdb_path):
    graph_x = []
    graph_pos = []

    with open(pdb_path) as pdb_in:
        for line in pdb_in:
            if line[0:4] == "ATOM":
                if re.match(r"^(\d+H|H)", line[12:16].strip()):
                    continue
                else:
                    x, pos = featurize_pdb_line(line)
                    graph_x.append(x)
                    graph_pos.append(pos)

    g = Data(x=torch.tensor(graph_x), pos=torch.tensor(graph_pos))
    return g
