from minisom import MiniSom

from math import sqrt
from math import ceil
import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt
from bokeh.palettes import viridis
from bokeh.plotting import figure, output_file, show


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
            n = ceil(n / 2) * 2
            return get_minimal_distance_factors(n)
        else:
            root = np.floor(sqrt(n))
            while n % root > 0:
                root -= 1
            return int(root), int(n / root)
    except TypeError:
        print("Input '{}' is not a integer nor float, please enter one!".format(n))


# load doc2vec embedding vectors
df_output = pd.read_pickle(filepath_or_buffer='data/df_document_vectors.pkl')
array_doc_vec = np.load(file='data/document_vectors.npy')

# load rest of df - same as 01_load_clean.py
# PLACEHOLDER CODE

# normalise array using l2 normalisation
# so sum of squares is 1 for each vector
# https://stats.stackexchange.com/a/218729/276516
# https://stackoverflow.com/questions/53971240/normalize-vectors-in-gensim-model
normed_array_doc_vec = normalize(X=array_doc_vec, axis=0, norm='l2')

# bring data together
df_output['base_path'] = 'www.gov.uk' + df_output['base_path']
df_output['document_vectors_norm'] = normed_array_doc_vec.tolist()

# focus on specific subset of data, these are:
# - Brexit/Transition
# - Coronavirus
# For now, just random sample
df_output = df_output.sample(n=50000, random_state=42)
normed_array_doc_vec = df_output['document_vectors_norm'].tolist()
normed_array_doc_vec = np.array(normed_array_doc_vec)

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


# values of x and y are too far from each other
x, y = get_minimal_distance_factors(total_neurons)

del total_neurons, normal_cov, eigen_values, result

# initialization and training of SOM
som = MiniSom(x=x, y=y, input_len=len_vector,
              activation_distance='cosine', topology='rectangular', neighborhood_function='gaussian',
              sigma=0.8, learning_rate=0.8, random_seed=42)
som.train_batch(data=normed_array_doc_vec, num_iteration=1000, verbose=True)

# plot distance map, U-Matrix, using pseudocolour
# where neurons of maps are displayed as an array of cells
# and colour represents the weights/distance from neighbour neurons
plt.figure(figsize=(9, 9))
plt.pcolor(som.distance_map().T, cmap='bone_r')
plt.colorbar()

# will consider all the sample mapped into a specific neuron as a cluster.
# to identify each cluster more easily, will translate the bi-dimensional indices
# of the neurons on the SOM into mono-dimensional indices.
# each neuron represents a cluster
winner_coordinates = np.array([som.winner(x) for x in normed_array_doc_vec]).T
# with np.ravel_multi_index, we convert the bi-dimensional coordinates to a mono-dimensional index
cluster_index = np.ravel_multi_index(multi_index=winner_coordinates, dims=(x, y))

# via applying SOMs,
# have gone from mapping space X of dimensions
len(normed_array_doc_vec)
# to a mapping space Y of dimensions
len(np.unique(cluster_index))

# bring cluster indices back to original data to tie them with base_path
df_output['cluster_index'] = cluster_index

# plot each cluster with a different colour
# plotting clusters using the first 2-dimensions of the data
for cluster in np.unique(cluster_index)[10:12]:
    plt.scatter(x=normed_array_doc_vec[cluster_index == cluster, 0],
                y=normed_array_doc_vec[cluster_index == cluster, 1],
                label='cluster = ' + str(cluster),
                alpha=0.7)
# plotting centroids found here: https://github.com/JustGlowing/minisom/blob/master/examples/Clustering.ipynb
# exclude because it's causing noisy plot

plt.legend()
plt.savefig(fname='reports/figures/som_cluster_scatter.png')
plt.close('all')


# experiment with bokeh plotting
TOOLS = "hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,"

p = figure(tools=TOOLS)
list_cluster_index = np.unique(cluster_index).tolist()
palette_colours = viridis(n=len(list_cluster_index))
dict_plot = {list_cluster_index[i]: palette_colours[i] for i in range(len(list_cluster_index))}
dict_plot = {list_cluster_index[i]: palette_colours[i] for i in range(12)}

for key, value in dict_plot.items():
    p.scatter(x=normed_array_doc_vec[cluster_index == key, 0],
              y=normed_array_doc_vec[cluster_index == key, 1],
              fill_color=value)
output_file("color_scatter.html", title="SOMs visualised on two-dimensions")
show(p)

del p
