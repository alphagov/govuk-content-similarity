from minisom import MiniSom

from math import sqrt
from math import ceil
import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt


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

# load USE embedding vectors
# from: https://drive.google.com/drive/u/0/folders/1xnAiSEkQY2P6uTNgax4miZaISn6sVeu4
df_output = pd.read_csv(filepath_or_buffer='data/text_df.csv')
df_doc_vec = pd.read_csv(filepath_or_buffer='data/embeddings_df.csv')
array_doc_vec = df_doc_vec.drop(columns='content_id').to_numpy()

# normalise array using l2 normalisation
# so sum of squares is 1 for each vector
# https://stats.stackexchange.com/a/218729/276516
# https://stackoverflow.com/questions/53971240/normalize-vectors-in-gensim-model
normed_array_doc_vec = normalize(X=array_doc_vec, axis=0, norm='l2')

# alternative normalisation
normed_array_doc_vec = (array_doc_vec - np.mean(array_doc_vec, axis=0)) / np.std(array_doc_vec, axis=0)

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
som.train_batch(data=normed_array_doc_vec, num_iteration=500, verbose=True)

# will consider all the sample mapped into a specific neuron as a cluster.
# to identify each cluster more easily, will translate the bi-dimensional indices
# of the neurons on the SOM into mono-dimensional indices.
# each neuron represents a cluster
winner_coordinates = np.array([som.winner(x) for x in normed_array_doc_vec]).T
# with np.ravel_multi_index, we convert the bi-dimensional coordinates to a mono-dimensional index
cluster_index = np.ravel_multi_index(multi_index=winner_coordinates, dims=(x, y))

# plot each luster with a different colour
# plotting clusters using the first 2-dimensions of the data
for cluster in np.unique(cluster_index):
    plt.scatter(x=normed_array_doc_vec[cluster_index == cluster, 0],
                y=normed_array_doc_vec[cluster_index == cluster, 1],
                label='cluster = ' + str(cluster),
                alpha=0.7)
# plotting centroids
for centroid in som.get_weights():
    plt.scatter(x=centroid[:, 0],
                y=centroid[:, 1],
                marker='x',
                s=80,
                linewidths=3,
                color='k',
                label='centroid')
plt.legend()
plt.savefig(fname='reports/figures/som_cluster_scatter_use_l2.png')
plt.close('all')
