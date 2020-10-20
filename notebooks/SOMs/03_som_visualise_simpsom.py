import SimpSOM as sps

import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize

import plotly.offline as py
import plotly.graph_objs as go
import plotly.express as px

from PIL import Image
from skimage import io


df_output = pd.read_pickle(filepath_or_buffer='data/df_document_vectors.pkl')
array_doc_vec = np.load(file='data/document_vectors.npy')

labels = df_output['base_path'][:500].tolist()

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

# save the som
with open('././data/som_net_train.pkl', 'wb') as outfile:
    pickle.dump(net, outfile)

# load trained model
with open('././data/som_net_train.pkl', 'rb') as infile:
    net = pickle.load(infile)

# project our data onto the map to see where pages are being mapped
plot_data = net.project(array=normed_array_doc_vec)

# get (x,y) coordinates to crop
img = io.imread(fname='outputs/nodesDifference.png')
fig = px.imshow(img)
py.plot(figure_or_data=fig, filename='outputs/ss_nodes_difference.html')

# interactive plotting to see base_paths
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=[x for x, y in plot_data],
        y=[y for x, y in plot_data],
        hovertext=[str(n) for n in labels],
        text=[str(n) for n in labels],
        mode='markers',
        marker=dict(
            size=14,
            color='black',
            opacity=1),
        showlegend=False)
)

# add hex plot
im = Image.open(fp='outputs/nodesDifference.png')
left = 195
top = 150
right = 1135
bottom = 620
im = im.crop(box=(left, top, right, bottom))
fig.add_layout_image(
        dict(
            source=im,
            xref="x",
            yref="y",
            x=-0.5,
            y=6.5,
            sizex=14,
            sizey=8,
            opacity=0.5,
            layer="below")
)

py.plot(figure_or_data=fig, filename='outputs/ss_nodes_difference.html')
