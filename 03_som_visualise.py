from minisom import MiniSom

from math import sqrt
from math import ceil
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm, colorbar


def is_prime(n):
    """
    Checks if a number is prime.
    https://stackoverflow.com/a/17377939/13416265

    :param n: Number to check if prime
    :return: Boolean to state whether n is prime or not
    """
    if n == 2:
        return True
    if n % 2 == 0 or n <= 1:
        return False

    root = int(sqrt(n)) + 1
    for divisor in range(3, root, 2):
        if n % divisor == 0:
            return False
    return True


def get_minimal_distance_factors(n):
    """
    Get the factors of a number which have the smallest difference between them.
    This is so we specify the gridsize for our SOM to be relatively balanced in height and width.
    Note: If have prime number, naively takes nearest even number.
            This is so we don't end up in situation where have 1xm gridsize.
            Might be better ways to do this.

    :param n: Integer or float we want to extract the closest factors from.
    :return: The factors of n whose distance from each other is minimised.
    """
    try:
        n = int(n)
        if isinstance(n, float) or is_prime(n):
            # gets next largest even number
            n = ceil(n/2) * 2
            return get_minimal_distance_factors(n)
        else:
            root = np.floor(sqrt(n))
            while n % root > 0:
                root -= 1
            return int(root), int(n/root)
    except TypeError:
        print("Input '{}' is not a integer nor float, please enter one!".format(n))


df_output = pd.read_pickle(filepath_or_buffer='data/df_document_vectors.pkl')
array_doc_vectors = np.load(file='data/document_vectors.npy')

# compute parameters
len_vector = array_doc_vectors.shape[1]
# compute number of neurons and how many make up each side
# where this is approximately the ratio of two largest eigenvalues of training data's covariance matrix
# https://python-data-science.readthedocs.io/en/latest/unsupervised.html
# rule of thumb for setting grid is 5*sqrt(N) where N is sample size
# example must be transpose of our case:
# https://stats.stackexchange.com/questions/282288/som-grid-size-suggested-by-vesanto
total_neurons = 5 * sqrt(array_doc_vectors.shape[0])
# calculate eigenvalues
normal_cov = np.cov(array_doc_vectors.T)
eigen_values = np.linalg.eigvals(normal_cov)
# get two largest eigenvalues
result = sorted([i.real for i in eigen_values])[-2:]
x = result[1]/result[0]
y = total_neurons/x
x = int(round(x, 0))
y = int(round(y, 0))


# values of x and y are too far from each other
x, y = get_minimal_distance_factors(total_neurons)


del total_neurons, normal_cov, eigen_values, result

# initialization and training of SOM
som = MiniSom(x=x, y=y, input_len=len_vector,
              activation_distance='cosine', topology='hexagonal',
              sigma=0.3, learning_rate=0.5, random_seed=42)
# initialise weights to map
som.pca_weights_init(data=array_doc_vectors)
som.train_batch(data=array_doc_vectors, num_iteration=100)

# basic plot
plt.figure(figsize=(10, 10))
plt.pcolor(som.distance_map().T, cmap='bone_r')
plt.show()

# hexagonal plotting
f = plt.figure(figsize=(10, 10))
ax = f.add_subplot(111)
ax.set_aspect('equal')

xx, yy = som.get_euclidean_coordinates()
umatrix = som.distance_map()
weights = som.get_weights()

for i in range(weights.shape[0]):
    for j in range(weights.shape[1]):
        wy = yy[(i, j)]*2/np.sqrt(3)*3/4
        hex = RegularPolygon((xx[(i, j)], wy), numVertices=6, radius=.95/np.sqrt(3),
                             facecolor=cm.Blues(umatrix[i, j]), alpha=.4, edgecolor='gray')
        ax.add_patch(hex)

for cnt, x in enumerate(array_doc_vectors):
    w = som.winner(x)  # getting the winner
    # place a marker on the winning position for the sample xx
    wx, wy = som.convert_map_to_euclidean(w)
    wy = wy*2/np.sqrt(3)*3/4
    plt.plot(wx, wy, markerfacecolor='None', markersize=12, markeredgewidth=2)

xrange = np.arange(weights.shape[0])
yrange = np.arange(weights.shape[1])
plt.xticks(xrange-.5, xrange)
plt.yticks(yrange*2/np.sqrt(3)*3/4, yrange)

divider = make_axes_locatable(plt.gca())
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
cb1 = colorbar.ColorbarBase(ax_cb, cmap=cm.Blues,
                            orientation='vertical', alpha=.4)
cb1.ax.get_yaxis().labelpad = 16
cb1.ax.set_ylabel('distance from neurons in the neighbourhood',
                  rotation=270, fontsize=16)
plt.gcf().add_axes(ax_cb)

plt.show()
