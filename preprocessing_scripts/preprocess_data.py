from typing import *

import numpy as np
import pandas as pd

import os
import glob

import GEOparse
import mygene


from sklearn.model_selection import train_test_split

##############################################
# Step 1: Load AD microarray data (GSE15222) #
##############################################

gse_ad = GEOparse.get_GEO("GSE15222", destdir=".")

# Pivot samples to get a gene × sample matrix
expr_ad = gse_ad.pivot_samples('VALUE')  # columns = samples (patient/tissue), rows = probes (specific gene transcript), values = measured expression

# Convert from probe x sample matrix -> gene x sample matrix (some probes may not map to anything, some probes map to multiple genes)
# Get platform annotation
platform = gse_ad.gpls[list(gse_ad.gpls.keys())[0]]

mg = mygene.MyGeneInfo()

# Extract RefSeq IDs from platform
refseq_ids = platform.table['GB_ACC'].dropna().unique().tolist()

# Query mygene for gene symbols
query_res = mg.querymany(refseq_ids, scopes='refseq', fields='symbol', species='human')

# Convert to DataFrame
mapping_df = pd.DataFrame(query_res)[['query','symbol']].dropna()
mapping_df.rename(columns={'query':'GB_ACC','symbol':'Gene Symbol'}, inplace=True)# Merge platform table with gene symbols


platform_map = platform.table[['ID','GB_ACC']].merge(mapping_df, on='GB_ACC', how='left')

# Create probe:gene dictionary
probe_to_gene = dict(zip(platform_map['ID'], platform_map['Gene Symbol']))


# expr_ad: probes x samples
expr_ad['Gene Symbol'] = expr_ad.index.map(lambda x: probe_to_gene.get(x, None))

# Drop probes that didn’t map
expr_ad = expr_ad.dropna(subset=['Gene Symbol'])

# Collapse multiple probes mapping to the same gene (mean expression)
expr_ad_gene = expr_ad.groupby('Gene Symbol').mean() # gene x sample matrix

# Standardizing (for smoother data)
expr_ad_log = np.log2(expr_ad_gene + 1)

# Z-score per gene (rows)
expr_ad_z = (expr_ad_log.sub(expr_ad_log.mean(axis=1), axis=0)
                        .div(expr_ad_log.std(axis=1).replace(0, 1e-8), axis=0))


############################################
# Step 2: Load T2D RNA-seq data (GSE38642) #
############################################

gse_t2d = GEOparse.get_GEO("GSE38642", destdir=".")

# Pivot samples to get a gene × sample matrix
expr_t2d = gse_t2d.pivot_samples('VALUE')  # columns = samples (patient/tissue), rows = genes (specific gene transcript), values = measured expression

# Helper function to extract gene symbols from the gene_assignment available for T2D dataset
def extract_gene_symbol(s):
  if s == "---":
    return None
  
  mappings = s.split("///")
  for mapping in mappings:
    parts = mapping.split("//")
    if parts[1] == "---":
      return None
    return parts[1].strip()
  
  return None

platform_t2d = gse_t2d.gpls[list(gse_t2d.gpls.keys())[0]]
gene_map = platform_t2d.table[['ID', 'gene_assignment']].copy()
gene_map['GeneSymbol'] = gene_map['gene_assignment'].apply(extract_gene_symbol)

probe_to_gene = gene_map.set_index('ID')['GeneSymbol'].to_dict()

expr_t2d['GeneSymbol'] = expr_t2d.index.map(probe_to_gene)
expr_t2d = expr_t2d.dropna(subset=['GeneSymbol'])
expr_t2d = expr_t2d.groupby('GeneSymbol').mean()

# Normalize to CPM
expr_t2d_cpm = expr_t2d.div(expr_t2d.sum(axis=0), axis=1) * 1e6

# Log2 transform
expr_t2d_log = np.log2(expr_t2d_cpm + 1)

# Z-score per gene
expr_t2d_z = (
    expr_t2d_log
        .sub(expr_t2d_log.mean(axis=1), axis=0)
        .div(expr_t2d_log.std(axis=1).replace(0, 1e-8), axis=0)
)

#############################################
# Step 3: Get genes shared between AD + T2D #
#############################################

shared_genes = expr_ad_z.index.intersection(expr_t2d_z.index)
print(f"Number of shared genes: {len(shared_genes)}")

expr_ad_final = expr_ad_z.loc[shared_genes]
expr_t2d_final = expr_t2d_z.loc[shared_genes]


####################################
# Step 4: Prepare data for PyTorch #
####################################

# Convert to samples x genes
X_ad = expr_ad_final.T.astype(np.float32)    # shape: samples_AD x genes
X_t2d = expr_t2d_final.T.astype(np.float32)  # shape: samples_T2D x genes

# Combine datasets
X_combined = pd.concat([X_ad, X_t2d], axis=0)
y_combined = np.array([0]*X_ad.shape[0] + [1]*X_t2d.shape[0])  # 0=AD, 1=T2D

# Saving to disk
np.save("X_ad.npy", X_ad)
np.save("X_t2d.npy", X_t2d)
np.save("X_combined.npy", X_combined)
np.save("y_combined.npy", y_combined)
np.save("shared_genes.npy",  shared_genes)


##############################################################
# Step 5: Load AD-T2D Mouse Dataset (for testing, GSE262426) #
##############################################################
os.chdir('GSE262426')
files = glob.glob("GSM*.csv.gz")
pseudobulk_list = []
sample_metadata = []

for f in files:
    df = pd.read_csv(f, index_col=0)
    pseudobulk = df.sum(axis=0)

    basename = os.path.basename(f)
    sample_id = basename.split(".")[0]
    condition = sample_id.split("_")[1]
    mouse_id = sample_id.split("_")[2]

    pseudobulk.name = sample_id
    
    pseudobulk_list.append(pseudobulk)
    sample_metadata.append({"sample_id": sample_id,
                        "condition": condition,
                        "mouse": mouse_id})

mouse_expr = pd.concat(pseudobulk_list, axis=1)

mouse_meta = pd.DataFrame(sample_metadata)


##############################################################
# Step 6: Preprocess mouse dataset #
##############################################################
# Counts per million
counts = mouse_expr
cpm = (counts / counts.sum(axis=0)) * 1e6

# Log-transform
mouse_expr_log = np.log1p(cpm)

cond_map = {
    "WTCtrD": "WT_Control",
    "WTHfD": "WT_HighFat",
    "dCtrD": "AD_Control",
    "dHfD": "AD_HighFat"
}

mouse_meta["label"] = mouse_meta["condition"].map(cond_map)

label_mapping = {
    "WT_Control": 0,
    "WT_HighFat": 1,
    "AD_Control": 2,
    "AD_HighFat": 3
}

mouse_meta["label_int"] = mouse_meta["label"].map(label_mapping)

####################################
# Step 7: Prepare data for PyTorch #
####################################
# Convert to numpy
X_mouse = mouse_expr_log.T.astype(np.float32)   # samples × genes
np.save("X_mouse.npy", X_mouse)

mouse_labels = mouse_meta["label_int"].values
np.save("y_mouse.npy", mouse_labels)