import SimpSOM as sps

import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize

import plotly.offline as py
import plotly.graph_objs as go

from PIL import Image, ImageChops


def auto_crop(filename):
    im = Image.open(filename)
    im = im.crop((0, 100, 2900, im.size[1]))
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


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

# save the network dimensions, PBC and node weights to file
net.save(fileName='somNet_trained', path='././data')

# project our data onto the map to see where pages are being mapped
plot_data = net.project(array=normed_array_doc_vec)

# interactive plotting to see base_paths
img_crop = auto_crop('nodesDifference.png')
img_crop.save('reports/figures/som_nodes_difference_crop.png')

# prepare plotly graph
trace = go.Scatter(
    x=[x for x, y in plot_data],
    y=[y for x, y in plot_data],
    hovertext=[str(n) for n in labels],
    text=[str(n) for n in labels],
    mode='markers',
    marker=dict(
        size=8,
        color='white',
        opacity=1
    ),
    showlegend=False
)

data = [trace]

layout = go.Layout(
    images=[dict(source='reports/figures/som_nodes_difference_crop.png',
                 xref='x',
                 yref='y',
                 x=-0.5,
                 y=39.5*2/np.sqrt(3)*3/4,
                 sizex=40.5,
                 sizey=40*2/np.sqrt(3)*3/4,
                 sizing='stretch',
                 opacity=0.5,
                 layer='below')],
    width=800,
    height=800,
    hovermode='closest',
    xaxis=dict(range=[-1, 41], zeroline=False, showgrid=False, ticks='', showticklabels=False),
    yaxis=dict(range=[-1, 41], zeroline=False, showgrid=False, ticks='', showticklabels=False),
    showlegend=True
)
fig = dict(data=data, layout=layout)
py.plot(figure_or_data=fig, filename='styled-scatter.html')
