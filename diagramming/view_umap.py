#%%
import numpy as np
import pandas as pd
from umap import UMAP
import plotly.express as px
np.random.seed(0)

#%%
#path = "../sc-gcn-dat_3/GNN_clustered.csv"
path = "../sc-gcn-dat_4/input_to_GNN.csv"
# path = "../sc-gcn-dat_4/GNN_clustered.csv"

df_plot = pd.read_csv(path)
df_plot['cluster'] = df_plot['cluster'].astype('str')

#%%

umap_2d = UMAP(n_components=2, init='random', random_state=0)
umap_3d = UMAP(n_components=3, init='random', random_state=0)

proj_2d = umap_2d.fit_transform(df_plot)
proj_3d = umap_3d.fit_transform(df_plot)


fig_2d = px.scatter(
    proj_2d, x=0, y=1,
    color=df_plot.cluster, labels={'color': 'cluster'}
)
fig_3d = px.scatter_3d(
    proj_3d, x=0, y=1, z=2,
    color=df_plot.cluster, labels={'color': 'cluster'}
)
fig_3d.update_traces(marker_size=5)

fig_2d.show()
fig_3d.show()
# %%
