from minisom import MiniSom

# from src.utils.helper_som import get_minimal_distance_factors
from src.utils.helper_som import get_som_dimensions
import pickle

import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize


# load doc2vec embedding vectors
df_output = pd.read_pickle(filepath_or_buffer='data/df.pkl')
array_doc_vec = np.load(file='data/doc_vec.npy')

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
x, y = get_som_dimensions(arr=normed_array_doc_vec)
# values of x and y are too far from each other
# x, y = get_minimal_distance_factors(total_neurons)


# initialization and training of SOM
som = MiniSom(x=x, y=y, input_len=len_vector,
              activation_distance='cosine', topology='hexagonal', neighborhood_function='gaussian',
              sigma=0.8, learning_rate=0.8, random_seed=42)
som.train_batch(data=normed_array_doc_vec, num_iteration=1000, verbose=True)

# save the SOM
with open('som.p', 'wb') as outfile:
    pickle.dump(som, outfile)

pd.to_pickle(obj=df_output, filepath_or_buffer='data/df_sample.pkl')
np.save(file='data/doc_vec_norm_sample.npy', arr=normed_array_doc_vec)
