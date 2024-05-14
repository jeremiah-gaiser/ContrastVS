import sys
import json
from glob import glob
import os
from pathlib import Path
import torch

sys.path.append("/xdisk/twheeler/jgaiser/ContrastVS/contrast_vs/src/")
from app_utils import generate_pocket_embed, get_voxel_sphere, load_models

graph_dir = sys.argv[1]
embed_dir = sys.argv[2]
pocket_specs = sys.argv[3]
model_path = sys.argv[4]

with open(pocket_specs) as config_in:
    pocket_specs = json.load(config_in)

if os.path.exists(embed_dir) == False:
    os.makedirs(embed_dir)

pyg_files = glob(graph_dir + "/*.pyg")
files_total = len(pyg_files)

for f_i, pyg_f in enumerate(pyg_files):
    print("%s of %s" % (f_i, files_total))
    pocket_model, _ = load_models(model_path, load_weights=True)
    filename = pyg_f.split("/")[-1].split(".")[0]

    if filename not in pocket_specs:
        continue

    pg = torch.load(pyg_f)
    eg = generate_pocket_embed(pg, pocket_specs[filename], pocket_model)
    print(embed_dir + "/%s.pyg" % filename)
    torch.save(eg, embed_dir + "/%s.pyg" % filename)
