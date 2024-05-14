import sys
from pathlib import Path
import os
import torch
from glob import glob

sys.path.append("/xdisk/twheeler/jgaiser/ContrastVS/contrast_vs/src/")
from app_utils import generate_interaction_profile

embed_dir = sys.argv[1]
ip_dir = sys.argv[2]
train_mol_loader = sys.argv[3]

if os.path.exists(ip_dir) == False:
    os.makedirs(ip_dir)

train_embed_loader = torch.load(train_mol_loader)

for embed_f in glob(embed_dir + "*.pyg"):
    fname = embed_f.split("/")[-1].split(".")[0]
    ip_f = ip_dir + "/%s.pyg" % fname

    if os.path.exists(ip_f):
        continue
    else:
        Path(ip_f).touch()

    pocket_embed = torch.load(embed_f)
    ipg = generate_interaction_profile(pocket_embed, train_embed_loader)
    print(ip_f)
    torch.save(ipg, ip_f)
