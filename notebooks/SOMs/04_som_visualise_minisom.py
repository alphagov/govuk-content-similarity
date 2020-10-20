import pickle
from math import sqrt

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm, colorbar

# from bokeh.palettes import viridis
# from bokeh.plotting import figure, output_file, show

# import data, normed document vectors and som
df_sample = pd.read_pickle(filepath_or_buffer='data/df_sample.pkl')
normed_array_doc_vec = np.load(file='data/document_vectors_norm_sample.npy')
with open('data/som.p', 'rb') as infile:
    som = pickle.load(infile)

# compute parameters
len_vector = normed_array_doc_vec.shape[1]
# compute number of neurons and how many make up each side
# where this is approximately the ratio of two largest eigenvalues of training data's covariance matrix
# https://python-data-science.readthedocs.io/en/latest/unsupervised.html
# rule of thumb for setting grid is 5*sqrt(N) where N is sample size
# example must be transpose of our case:
# https://stats.stackexchange.com/questions/282288/som-grid-size-suggested-by-vesanto
total_neurons = 5 * sqrt(normed_array_doc_vec.shape[0])
# calculate eigenvalues
normal_cov = np.cov(normed_array_doc_vec.T)
eigen_values = np.linalg.eigvals(normal_cov)
# get two largest eigenvalues
result = sorted([i.real for i in eigen_values])[-2:]
x = result[1] / result[0]
y = total_neurons / x
x = int(round(x, 0))
y = int(round(y, 0))

# will consider all the sample mapped into a specific neuron as a cluster.
# to identify each cluster more easily, will translate the bi-dimensional indices
# of the neurons on the SOM into mono-dimensional indices.
# each neuron represents a cluster
winner_coordinates = np.array([som.winner(x) for x in normed_array_doc_vec]).T
# with np.ravel_multi_index, we convert the bi-dimensional coordinates to a mono-dimensional index
cluster_index = np.ravel_multi_index(multi_index=winner_coordinates, dims=(x, y))


# bring cluster indices back to original data to tie them with base_path
df_sample['cluster_index'] = cluster_index

# plot hexagonal topology
f = plt.figure(figsize=(10, 10))
ax = f.add_subplot(111)
ax.set_aspect('equal')
xx, yy = som.get_euclidean_coordinates()
umatrix = som.distance_map()
weights = som.get_weights()
for i in range(weights.shape[0]):
    for j in range(weights.shape[1]):
        wy = yy[(i, j)] * 2 / np.sqrt(3) * 3 / 4
        hex = RegularPolygon(xy=(xx[(i, j)], wy),
                             numVertices=6,
                             radius=.95 / np.sqrt(3),
                             facecolor=cm.Blues(umatrix[i, j]),
                             alpha=.4,
                             edgecolor='gray')
        ax.add_patch(hex)

for cnt, x in enumerate(normed_array_doc_vec):
    w = som.winner(x)
    wx, wy = som.convert_map_to_euclidean(xy=w)
    wy = wy * 2 / np.sqrt(3) * 3 / 4
    plt.plot(wx, wy, markerfacecolor='none', markeredgecolor='black', markersize=12, markeredgewidth=2)

xrange = np.arange(weights.shape[0])
yrange = np.arange(weights.shape[1])
plt.xticks(xrange - .5, xrange)
plt.yticks(yrange * 2 / np.sqrt(3) * 3 / 4, yrange)

divider = make_axes_locatable(plt.gca())
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
cbl = colorbar.ColorbarBase(ax=ax_cb, cmap=cm.Blues,
                            orientation='vertical', alpha=.4)
cbl.ax.get_yaxis().labelpad = 16
cbl.ax.set_ylabel('distance from neurons in the neighbourhood', rotation=270, fontsize=16)
plt.gcf().add_axes(ax_cb)

plt.show()
plt.savefig('outputs/minisom_hex.png')
plt.close('all')
