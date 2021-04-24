#%%
import dgl

path = "../sc-gcn-dat_4/scgraph.bin"
G = dgl.data.utils.load_graphs(path)
G = G[0][0]
print(G)
# %%
