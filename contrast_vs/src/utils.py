import importlib
import torch
from torch.nn import CosineSimilarity
from torch_geometric.utils import k_hop_subgraph


def to_onehot(value, vocab):
    onehot = [0 for _ in vocab]

    if value in vocab:
        onehot[vocab.index(value)] = 1

    return onehot


def get_n_closest(a, b, n, dist_func):
    """
    For each row in a, get n closest rows in b.
    expects a dist_func like torch.cdist.
    """
    d_vals, d_index = torch.sort(dist_func(a, b))
    return d_vals[:, :n], d_index[:, :n]


def all_pairs_cos(a, b, device="cpu"):
    """
    cosine-similarity that behaves like torch.cdist
    """
    cos = CosineSimilarity(dim=2)
    a = a.unsqueeze(1).to(device)
    b = b.unsqueeze(0).to(device)
    return cos(a, b).cpu()


def get_k_hop_neighborhoods(atom_index, max_k, ei):
    met_neighbors = torch.tensor([atom_index])
    k_hop_neighborhoods = []

    for k_val in range(1, max_k + 1):
        k_neighbors = k_hop_subgraph(atom_index, k_val, ei)[0]
        k_neighbors = k_neighbors[
            ~torch.any(k_neighbors.unsqueeze(1) == met_neighbors.unsqueeze(0), 1)
        ]
        met_neighbors = torch.unique(torch.hstack((met_neighbors, k_neighbors)))

        k_hop_neighborhoods.append(k_neighbors)

    return k_hop_neighborhoods


def cos_dist(a, b, device="cpu"):
    return 1 - all_pairs_cos(a, b, device)


def load_class_from_file(file_path):
    class_name = file_path.split("/")[-1].split(".")[0]
    spec = importlib.util.spec_from_file_location(class_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)
