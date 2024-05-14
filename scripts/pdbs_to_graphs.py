import sys
from glob import glob
import os
from pathlib import Path
import torch

sys.path.append("/xdisk/twheeler/jgaiser/ContrastVS/contrast_vs/src/")
from pdb_utils import pdb2graph

pdb_dir = sys.argv[1]
graph_dir = sys.argv[2]

if os.path.exists(graph_dir) == False:
    os.makedirs(graph_dir)

pdb_files = glob(pdb_dir + "/*.pdb")
files_total = len(pdb_files)

for f_i, pdb_f in enumerate(pdb_files):
    print("%s of %s" % (f_i, files_total))
    filename = pdb_f.split("/")[-1].split(".")[0]
    g = pdb2graph(pdb_f)
    g.name = filename
    torch.save(g, graph_dir + "/%s.pyg" % filename)
