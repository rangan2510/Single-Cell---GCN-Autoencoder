#%%
from pandas.io.pytables import ClosedFileError
import scvi
import scanpy as sc
import torch
import numpy as np
import pandas as pd

sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')

results_file = 'write/pbmc3k.h5ad'

adata = sc.read_10x_mtx(
    'data/',  # the directory with the `.mtx` file
    var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
    cache=False)     
# %%
adata.var_names_make_unique()
adata
# %%
#PREPROCESSING
sc.pl.highest_expr_genes(adata, n_top=20, )
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

# %%
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True)
# %%
sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')
# %%
adata = adata[adata.obs.n_genes_by_counts < 2500, :]
adata = adata[adata.obs.pct_counts_mt < 5, :]
# %%
adata.layers["counts"] = adata.X.copy() # preserve counts
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata # freeze the state in `.raw`

# %%
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
# %%
scvi.model.SCVI.setup_anndata(
    adata,
    layer="counts",
    # categorical_covariate_keys=["cell_source", "donor"],
    # continuous_covariate_keys=["percent_mito", "percent_ribo"]
)

# %%
model = scvi.model.SCVI(adata)
model
model.train(max_epochs=50, plan_kwargs={'lr':5e-3}, check_val_every_n_epoch=5)
# %%
latent = model.get_latent_representation()
adata.obsm["X_scVI"] = latent

# %%
sc.pp.neighbors(adata, use_rep="X_scVI")
sc.tl.umap(adata, min_dist=0.3)
sc.tl.leiden(adata, key_added="leiden_scVI", resolution=0.5)
# %%
sc.pl.umap(
    adata,
    color=["leiden_scVI"],
    frameon=False,
)
# %%
sc.tl.rank_genes_groups(adata, 'leiden_scVI', method='t-test')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)
# %%
clusters = list(set(adata.obs['leiden_scVI']))
dat = [list(ele) for ele in list(adata.uns['rank_genes_groups']['names'])] 
ranked_genes_df = pd.DataFrame(data=dat, columns=clusters)
# %%
cell_type_df = pd.read_csv("data/markers.tsv", sep="\t")
cell_type_df_ = cell_type_df[["official gene symbol", "cell type","organ"]]
# %%
gene2celltype = {}
gene2organ = {}
annotation = {}
k = 1
for i in clusters:
    i = int(i)
    cluster_i = pd.DataFrame(list(ranked_genes_df.iloc[:, i]), columns=["official gene symbol"])
    cell_type_i = pd.merge(cluster_i,cell_type_df_, how ='inner', on =["official gene symbol"])
    print(cell_type_i)
    top_k_cells = list(cell_type_i["cell type"][:k])
    top_k_organs = list(cell_type_i["organ"][:k])
    gene2celltype[i] = ', '.join(top_k_cells)
    gene2organ[i] = ', '.join(top_k_organs)


# %%
