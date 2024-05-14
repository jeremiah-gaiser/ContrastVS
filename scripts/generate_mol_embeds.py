import sys
import json
from glob import glob
import os
from pathlib import Path
import torch

sys.path.append("/xdisk/twheeler/jgaiser/ContrastVS/contrast_vs/src/")
from app_utils import generate_mol_embed, load_models

graph_dir = sys.argv[1]
embed_dir = sys.argv[2]
model_path = sys.argv[3]

if os.path.exists(embed_dir) == False:
    os.makedirs(embed_dir)

pyg_files = glob(graph_dir + "/*.pyg")
files_total = len(pyg_files)

for f_i, pyg_f in enumerate(pyg_files):
    print("%s of %s" % (f_i, files_total))
    filename = pyg_f.split("/")[-1].split(".")[0]
    _, mol_model = load_models(model_path, load_weights=True)

    mg = torch.load(pyg_f)
    eg = generate_mol_embed(mg, mol_model)
    torch.save(eg, embed_dir + "/%s.pyg" % filename)
