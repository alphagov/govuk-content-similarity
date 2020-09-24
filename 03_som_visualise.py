from minisom import MiniSom
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

df_output = pd.read_pickle(filepath_or_buffer='data/df_document_vectors.pkl')
array_doc_vectors = np.load(file='data/document_vectors.npy')

len_vector = array_doc_vectors.shape[1]

# initialization and training of 6x6 SOM
som = MiniSom(x=6, y=6, input_len=len_vector, sigma=0.3, learning_rate=0.5)
som.train(data=array_doc_vectors, num_iteration=100)

# basic plot
plt.figure(figsize=(6, 5))
plt.pcolor(som.distance_map().T, cmap='bone_r')
plt.show()
