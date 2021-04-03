#%%
import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

#%%
# step 1: load the graph
G = dgl.data.utils.load_graphs('scgraph.bin')
G = G[0][0]
print(G)

#%%
# step 2: move edge data to nodes
# edge weights
G.update_all(fn.copy_e('weight','m'),fn.sum('m','e_weight_sum'))
G.ndata['e_weight_sum'] = G.ndata['e_weight_sum'].unsqueeze(1).float()
# edge cluster (multiclass needs one-hot encoding)
edata = G.edata['cluster']
G.edata['cluster'] = torch.nn.functional.one_hot(edata).float()
G.update_all(fn.copy_e('cluster','m'),fn.mean('m','e_cluster'))

# network group (multiclass)
edata = G.edata['group']
G.edata['group'] = torch.nn.functional.one_hot(edata).float()
G.update_all(fn.copy_e('group','m'),fn.mean('m','e_group'))
# %%
# concatenate all features into a single tensor
cfeat = torch.cat((G.ndata['feat'],G.ndata['e_weight_sum'],G.ndata['e_cluster'],G.ndata['e_group']),1).float()
G.ndata['cfeat'] = cfeat/cfeat.max()
features = G.ndata['cfeat'].float()
G = dgl.add_self_loop(G)
G
# %%
# define simple AE network
from dgl.nn import GraphConv, SAGEConv

class Block(nn.Module):
    def __init__(self, in_feats, out_feats, stages=3, func_type='encoder',aggregator_type='gcn', feat_drop=0.1):
        super(Block, self).__init__()
        delta = int(abs(in_feats-out_feats)/stages)
        self.layers  = nn.ModuleList()
        if (func_type=='encoder'):
            h_feats = in_feats - delta
            for _ in range(stages):
                self.layers.append(SAGEConv(in_feats, h_feats, aggregator_type, feat_drop))
                self.layers.append(GraphConv(h_feats,h_feats))
                in_feats = h_feats
                h_feats -= delta
        else: #(func_type==decoder)
            h_feats = in_feats + delta
            for _ in range(stages):
                self.layers.append(SAGEConv(in_feats, h_feats, aggregator_type, feat_drop))
                self.layers.append(GraphConv(h_feats,h_feats))
                in_feats = h_feats
                h_feats += delta
        self.fc = nn.Linear(in_feats, out_feats)
        
    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = F.relu(layer(g,h))
        h = self.fc(h)
        return h


class AEGCN(nn.Module):
    def __init__(self, in_feats, out_feats, encoder_stages=3, decoder_stages=3):
        super(AEGCN, self).__init__()
        self.enc = Block(in_feats, out_feats, stages=encoder_stages, func_type='encoder', feat_drop=0.1)
        self.dec = Block(out_feats, in_feats, stages=decoder_stages, func_type='decoder', feat_drop=0.1)
        self.bn = torch.nn.BatchNorm1d(out_feats)
    def forward(self, g, inputs):
        h = self.enc(g, inputs)
        h = self.bn(h)
        h = self.dec(g, h)
        return h

IN_FEATS = 197
OUT_FEATS = 90
E_STAGES = 3
D_STAGES = 3
net = AEGCN(IN_FEATS, OUT_FEATS, E_STAGES, D_STAGES)
# %%
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

for epoch in range(5000+1):
    logits = net(G, features)
    logp = F.log_softmax(logits, 1)
    criterion = nn.KLDivLoss()
    loss = criterion(logp, features)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch%1000)==0:
        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
# %%
encoded_form = net.bn.forward(net.enc.forward(G,features))

#%%
import numpy as np
import pandas as pd
import collections
from sklearn.preprocessing import MinMaxScaler
import hdbscan
from umap import UMAP
import plotly.express as px
np.random.seed(0)

mat = encoded_form.detach().numpy()
scaler = MinMaxScaler()
scaler.fit(mat)
mat_sc = scaler.transform(mat)

clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
clusterer.fit(mat_sc)
labels = clusterer.labels_
print("# of clusters:", max(labels))
print(collections.Counter(labels))

cols = ['feat' + str(x) for x in range(OUT_FEATS)]
df_plot = pd.DataFrame(data=mat_sc, columns=cols)

umap_2d = UMAP(n_components=2, init='random', random_state=0)
umap_3d = UMAP(n_components=3, init='random', random_state=0)

proj_2d = umap_2d.fit_transform(df_plot)
proj_3d = umap_3d.fit_transform(df_plot)

df_plot['cluster'] = labels
df_plot.to_csv('GNN_clustered.csv', index=False)
max(labels)

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
# PLOTTING SOURCE GRAPH #####################################################
mat = G.ndata['cfeat'].numpy()
scaler = MinMaxScaler()
scaler.fit(mat)
mat_sc = scaler.transform(mat)

clusterer = hdbscan.HDBSCAN(min_cluster_size=3, gen_min_span_tree=True)
clusterer.fit(mat_sc)
labels = clusterer.labels_
print("# of clusters:", max(labels))
collections.Counter(labels)

cols = ['feat' + str(x) for x in range(IN_FEATS)]
df_plot = pd.DataFrame(data=mat_sc, columns=cols)

umap_2d = UMAP(n_components=2, init='random', random_state=0)
umap_3d = UMAP(n_components=3, init='random', random_state=0)

proj_2d = umap_2d.fit_transform(df_plot)
proj_3d = umap_3d.fit_transform(df_plot)

df_plot['cluster'] = labels
df_plot.to_csv('input_to_GNN.csv', index=False)

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

