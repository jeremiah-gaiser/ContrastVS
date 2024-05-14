import sys
import json
import math
from torch_geometric.nn import radius
import torch
from glob import glob

pairs_file = sys.argv[1]
json_out = sys.argv[2]

validation_pairs = torch.load(pairs_file)
complex_pocket_data = {}

for pg, mg in validation_pairs:
    proximal_m_atoms = torch.unique(radius(pg.pos, mg.pos, 4)[0])

    if proximal_m_atoms.size(0) == 0:
        continue

    center = mg.pos[proximal_m_atoms].mean(0).tolist()
    complex_pocket_data[pg.name] = [center, 7]

with open(json_out, "w") as jf:
    json.dump(complex_pocket_data, jf, indent=4)
