from minisom import MiniSom
import pandas as pd
import numpy as np

from matplotlib import cm
from bokeh.colors import RGB

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt',
                   names=['area', 'perimeter', 'compactness', 'length_kernel', 'width_kernel',
                          'asymmetry_coefficient', 'length_kernel_groove', 'target'],
                   sep='\t+')

data = data[data.columns[:-1]]
# data normalisation
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
data = data.values

# Initialization and training
som = MiniSom(15, 15, data.shape[1], sigma=1.5, learning_rate=.7, activation_distance='euclidean',
              topology='hexagonal', neighborhood_function='gaussian', random_seed=10)

som.train(data, 1000, verbose=True)

# create variables for plotting
xx, yy = som.get_euclidean_coordinates()
umatrix = som.distance_map()
weights = som.get_weights()

# store coordinates for hexes, colours and dots
hex_centre_x, hex_centre_y, hex_colour = [], [], []
for i in range(weights.shape[0]):
    for j in range(weights.shape[1]):
        wy = yy[(i, j)] * 2 / np.sqrt(3) * 3 / 4
        hex_centre_x.append(xx[(i, j)])
        hex_centre_y.append(wy)
        hex_colour.append(cm.Blues(umatrix[i, j]))

dot_x, dot_y = [], []
for i in data:
    w = som.winner(i)
    wx, wy = som.convert_map_to_euclidean(xy=w)
    wy = wy * 2 / np.sqrt(3) * 3 / 4
    dot_x.append(wx)
    dot_y.append(wy)

# clear environment
del som, xx, yy, w, data, i, j, weights

# convert matplotlib colour palette of 3-tuple RGB floats to 1-element hex strings
blues_plt = (255 * cm.Blues(range(256))).astype('int')
blues_bokeh = [RGB(*tuple(rgb)).to_hex() for rgb in blues_plt]

blues_plt_hex = [(255 * np.array(i)).astype(int) for i in hex_colour]
blues_bokeh_hex = [RGB(*tuple(rgb)).to_hex() for rgb in blues_plt_hex]
