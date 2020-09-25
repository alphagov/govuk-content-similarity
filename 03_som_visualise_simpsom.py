import SimpSOM as sps

from math import sqrt

import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

df_output = pd.read_pickle(filepath_or_buffer='data/df_document_vectors.pkl')
array_doc_vec = np.load(file='data/document_vectors.npy')

labels = df_output['base_path'][:500]

# normalise array using l2 normalisation
# so sum of squares is 1 for each vector
normed_array_doc_vec = normalize(X=array_doc_vec, axis=1, norm='l2')
# take smaller subset to work with
normed_array_doc_vec = normed_array_doc_vec[:500]

# train the SOM

# build network

# activating PCA for setting initial weights, PCI=True
# activating periodic boundary conditions
net = sps.somNet(netHeight=8, netWidth=14, data=normed_array_doc_vec, PCI=True, PBC=True)
# train with 0.1 learning rate
net.train(startLearnRate=0.1, epochs=5000)
# print map of weight differences between nodes,
# which will help identify cluster centres
net.diff_graph(show=True, printout=True)

# save the network dimensions, PBC and node weights to file
net.save(fileName='somNet_trained', path='././data')

# project our data onto the map to see where pages are being mapped
plot_data = net.project(normed_array_doc_vec)
