import SimpSOM as sps

import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize

import plotly.offline as py
import plotly.graph_objs as go

from PIL import Image, ImageChops


def auto_crop(filename, left, right, top):
    """
    Crop image passed in and overlay a blank canvas

    param filename: Name of image file to crop.
    param left: The amount of image you want to keep from left-side
    param right: The amount of image you want to keep from the right-side
    param top: The amount of image you want to keep from the top
    return: Cropped empty canvas for interactive plotting

    """
    im = Image.open(filename)
    im = im.crop(box=(left, right, top, im.size[1]))
    bg = Image.new(mode=im.mode, size=im.size, color=im.getpixel((0, 0)))
    diff = ImageChops.difference(image1=im, image2=bg)
    diff = ImageChops.add(image1=diff, image2=diff, scale=2.0, offset=-100)
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

# save the som
with open('././data/som_net_train.pkl', 'wb') as outfile:
    pickle.dump(net, outfile)

# load trained model
with open('././data/som_net_train.pkl', 'rb') as infile:
    test = pickle.load(infile)

# project our data onto the map to see where pages are being mapped
plot_data = net.project(array=normed_array_doc_vec)

# interactive plotting to see base_paths
img_crop = auto_crop(filename='nodesDifference.png',
                     left=0, right=0, top=2900)
img_crop.save('reports/figures/som_nodes_difference_crop.png')

# prepare plotly graph
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=[x for x, y in plot_data],
        y=[y for x, y in plot_data],
        hovertext=[str(n) for n in labels],
        text=[str(n) for n in labels],
        mode='markers',
        marker=dict(
            size=8,
            color='white',
            opacity=1),
        showlegend=False)
)

# add images
im = Image.open(fp='nodesDifference.png')
fig.add_layout_image(
        dict(
            source=im,
            xref="x",
            yref="y",
            x=0,
            y=3,
            sizex=2,
            sizey=2,
            sizing="stretch",
            opacity=0.5,
            layer="below")
)

py.plot(figure_or_data=fig, filename='styled-scatter.html')
