import sys
from glob import glob
import os
from pathlib import Path
from rdkit import Chem
import torch

sys.path.append("/xdisk/twheeler/jgaiser/ContrastVS/contrast_vs/src/")
from mol_utils import mol2graph

sdf_dir = sys.argv[1]
graph_dir = sys.argv[2]

if os.path.exists(graph_dir) == False:
    os.makedirs(graph_dir)

atom_types = []

sdf_files = glob(sdf_dir + "/*.sdf")
files_total = len(sdf_files)

symbols = []

for f_i, sdf_f in enumerate(sdf_files):
    print("%s of %s" % (f_i, files_total))
    filename = sdf_f.split("/")[-1].split(".")[0]

    m = Chem.SDMolSupplier(sdf_f, sanitize=False)[0]
    g = mol2graph(m, get_pos=True, k=3)

    if g == None:
        continue

    g.name = filename
    torch.save(g, graph_dir + "/%s.pyg" % filename)
