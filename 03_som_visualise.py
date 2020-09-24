from minisom import MiniSom

from math import sqrt
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

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

del total_neurons, normal_cov, eigen_values, result

# initialization and training of SOM
som = MiniSom(x=x, y=y, input_len=len_vector, sigma=0.3, learning_rate=0.5, random_seed=42)
# initialise weights to map
som.pca_weights_init(data=array_doc_vectors)
som.train_batch(data=array_doc_vectors, num_iteration=100)

# basic plot
plt.figure(figsize=(10, 10))
plt.pcolor(som.distance_map().T, cmap='bone_r')
plt.show()
