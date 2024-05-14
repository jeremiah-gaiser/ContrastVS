from copy import deepcopy
from utils import load_class_from_file
from pdb_utils import pdb2graph, vox_onehot
from utils import get_n_closest, cos_dist
import torch
from torch_geometric.nn import radius
from torch_geometric.data import Data


def add_voxels(pg, vox_coords):
    pg = deepcopy(pg)
    vox_block = vox_onehot().unsqueeze(0).repeat(vox_coords.size(0), 1)

    pg.vox_mask = torch.hstack(
        (torch.zeros(pg.pos.size(0)), torch.ones(vox_block.size(0)))
    ).bool()

    pg.x = torch.vstack((pg.x, vox_block))
    pg.pos = torch.vstack((pg.pos, vox_coords))

    return pg


def get_voxel_sphere(center, rad):
    square_dim = rad + 1

    min_pos = torch.tensor(
        [center[0] - square_dim, center[1] - square_dim, center[2] - square_dim]
    )

    max_pos = torch.tensor(
        [center[0] + square_dim, center[1] + square_dim, center[2] + square_dim]
    )

    vox_grid = torch.cartesian_prod(
        torch.arange(min_pos[0], max_pos[0]),
        torch.arange(min_pos[1], max_pos[1]),
        torch.arange(min_pos[2], max_pos[2]),
    )

    voxel_sphere = radius(
        vox_grid, torch.tensor(center).unsqueeze(0), r=rad, max_num_neighbors=99999
    )[1]

    return vox_grid[voxel_sphere].float()


def load_models(
    model_path, pe_hparams=None, me_hparams=None, load_weights=False, weight_suffix=""
):
    PocketEmbedder = load_class_from_file(model_path + "PocketEmbedder.py")
    MolEmbedder = load_class_from_file(model_path + "MolEmbedder.py")

    if pe_hparams is None:
        pe_hparams = torch.load(model_path + "pe_hparams.pt")
    if me_hparams is None:
        me_hparams = torch.load(model_path + "me_hparams.pt")

    pe = PocketEmbedder(*pe_hparams)
    me = MolEmbedder(*me_hparams)

    if load_weights:
        pe_w_path = model_path + "pe_w%s.pt" % weight_suffix
        me_w_path = model_path + "me_w%s.pt" % weight_suffix

        pe.load_state_dict(torch.load(pe_w_path, map_location=torch.device("cpu")))
        me.load_state_dict(torch.load(me_w_path, map_location=torch.device("cpu")))

    return pe, me


def generate_pocket_embed(protein_graph, pocket_specs, model):
    model.eval()
    voxel_pos = get_voxel_sphere(*pocket_specs)
    pg = add_voxels(protein_graph, voxel_pos)
    pg.batch = torch.zeros(pg.x.size(0))

    with torch.no_grad():
        out = model(pg)

    embed_g = Data(x=out, pos=voxel_pos, name=pg.name)
    return embed_g


def generate_mol_embed(mol_graph, model):
    model.eval()
    mg = deepcopy(mol_graph)
    mg.batch = torch.zeros(mg.x.size(0))

    with torch.no_grad():
        mg.x = model(mg)

    return mg


def generate_interaction_profile(
    pocket_embed_graph, mol_embed_loader, N=20, dist_func=cos_dist
):
    voxel_score = torch.zeros(pocket_embed_graph.x.size(0))
    min_neighbor_dist = torch.ones(pocket_embed_graph.x.size(0))
    mean_neighbor_dist = torch.ones(pocket_embed_graph.x.size(0))
    neighbor_atom_vals = torch.zeros_like(pocket_embed_graph.x)

    for bi, batch in enumerate(mol_embed_loader):
        print(bi)
        batch_x = batch.x[batch.proximity_mask]

        voxel_neighbor_dist, voxel_neighbor_idx = get_n_closest(
            batch_x, pocket_embed_graph.x, N, dist_func
        )

        for atom_idx in torch.arange(batch_x.size(0)):
            voxel_idx = voxel_neighbor_idx[atom_idx]
            voxel_score[voxel_idx] += 1
            min_neighbor_dist[voxel_idx] = torch.minimum(
                min_neighbor_dist[voxel_idx], voxel_neighbor_dist[atom_idx]
            )
            mean_neighbor_dist[voxel_idx] += voxel_neighbor_dist[atom_idx]
            neighbor_atom_vals[voxel_idx] += batch_x[atom_idx]

    mean_neighbor_dist[voxel_score > 0] /= voxel_score[voxel_score > 0]
    neighbor_atom_vals[voxel_score > 0] /= voxel_score[voxel_score > 0].unsqueeze(1)

    return Data(
        count=voxel_score,
        mean_dist=mean_neighbor_dist,
        min_dist=min_neighbor_dist,
        mean_embed=neighbor_atom_vals,
        pos=pocket_embed_graph.pos,
    )
