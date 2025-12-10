import gseapy as gp
import pandas as pd
import numpy as np

shared_genes = np.load("shared_genes.npy", allow_pickle=True)

###########################################
# 1. Load KEGG HUMAN pathways via Enrichr #
###########################################
kegg = gp.get_library(name="KEGG_2021_Human")

# kegg is a dict: pathway_name → list_of_genes
print(len(kegg), "KEGG pathways loaded")

################################
# 2. Build mask based on genes #
################################

gene2idx = {g:i for i,g in enumerate(shared_genes)}

filtered_pathways = {
    pname: [g for g in genes if g in gene2idx]
    for pname, genes in kegg.items()
}

filtered_pathways = {k:v for k,v in filtered_pathways.items() if len(v) >= 3}

print("Pathways with ≥3 shared genes:", len(filtered_pathways))


latent_dim = len(filtered_pathways) # want latent dim = one pathway (so can say AD and T2D differ strongly in pathways A, B, C.)

mask = np.zeros((latent_dim, len(shared_genes)), dtype=np.float32)

for j, (pname, genes) in enumerate(filtered_pathways.items()):
    for g in genes:
        mask[j, gene2idx[g]] = 1.0

mask.shape

np.save("pathway_mask.npy", mask)
np.save("filtered_pathways.npy", filtered_pathways)