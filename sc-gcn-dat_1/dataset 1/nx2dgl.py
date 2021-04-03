#%%
import pandas as pd
import networkx as nx

df = pd.read_csv('node_feat.csv')
df = df.sort_values(by=['Term']) 
G = nx.read_gexf('134.gexf')
# %%
import dgl
g = dgl.from_networkx(G, edge_attrs=['weight','group','cluster'])
# %%
import torch
# %%
ndata = torch.tensor(df.iloc[:,1:].values)
# %%
g.ndata['feat'] = ndata

from dgl.data.utils import save_graphs
save_graphs('scgraph.bin',[g])
# %%
node_idx = pd.DataFrame(data=sorted(G.nodes()), columns=['Term'])
node_idx.to_csv('node_idx.csv',index=True)
# %%
