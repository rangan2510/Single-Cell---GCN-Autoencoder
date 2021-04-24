#%%
import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
from dgl.nn import GraphConv
class AEGCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(AEGCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, in_feats)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return h

net = AEGCN(192, 20, 192)
# %%
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

for epoch in range(5000):
    logits = net(G, features)
    logp = F.log_softmax(logits, 1)
    criterion = nn.KLDivLoss()
    loss = criterion(logp, features)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch%100)==0:
        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
# %%
encoded_form = net.conv1.forward(G,features)

#%%
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

mat = encoded_form.detach().numpy()
scaler = MinMaxScaler()
scaler.fit(mat)
mat_sc = scaler.transform(mat)

optics = DBSCAN(min_samples=3).fit(mat_sc)
labels = optics.labels_
# %%
pca = PCA(n_components=2)
pca.fit(mat_sc)
X=pca.transform(mat_sc)
print(np.sum(pca.explained_variance_ratio_))
df_pca = pd.DataFrame(data=X, columns=['x1', 'x2'])
df_pca["labels"] = labels

fig = px.scatter(df_pca, x="x1", y="x2", color="labels", color_discrete_sequence=px.colors.qualitative.Dark24)
fig.show()
# %%
df_pca.to_csv("clust.csv")
fig = px.histogram(df_pca, x="labels", nbins=len(set(list(df_pca['labels']))))
fig.show()
# %%
