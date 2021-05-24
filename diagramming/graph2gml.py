#%%
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt

# path = "../sc-gcn-dat_2/dat2.gexf"
path = "../sc-gcn-dat_1/134.gexf"

G = nx.read_gexf(path)

#%%
pos = nx.spring_layout(G, seed=0)

M = G.number_of_edges()
edge_colors = range(2, M + 2)
edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
cmap = plt.cm.viridis
nodes = nx.draw_networkx_nodes(G, pos, node_size=20, node_color="cornflowerblue")
edges = nx.draw_networkx_edges(
    G,
    pos,
    node_size=20,
    arrowstyle="-",
    arrowsize=10,
    edge_color=edge_colors,
    edge_cmap=cmap,
    width=2,
)
labels = nx.draw_networkx_labels(G, pos, font_size="5", font_color="midnightblue")
# set alpha value for each edge
for i in range(M):
    edges[i].set_alpha(edge_alphas[i])

# pc = mpl.collections.PatchCollection(edges, cmap=cmap)
# pc.set_array(edge_colors)
# plt.colorbar(pc)

ax = plt.gca()
ax.set_axis_off()
plt.show()

# %%
