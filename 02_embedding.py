from time import time
from tqdm import tqdm
import multiprocessing

import pandas as pd
import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


n_cores = multiprocessing.cpu_count() - 1

df = pd.read_csv(filepath_or_buffer='data/df.csv')

# create TaggedDocument for doc2vec DBOW
# will use indices of sentences as tags
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(df['text_clean'])]

# instantiate and train a doc2vec DBOW model
model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample=0, workers=n_cores)
model_dbow.build_vocab([x for x in tqdm(documents)])
t = time()
model_dbow.train(documents, total_examples=model_dbow.corpus_count, epochs=1)
print('Time to train: {} minutes'.format(round((time() - t) / 60, 2)))

# save trained model
model_dbow.save('data/model_dbow')
#model_dbow = Doc2Vec.load(file_name)

# store and save document vectors in numpy array for minisom
array_doc_vectors = model_dbow.docvecs.vectors_docs
np.save(file='data/document_vectors.npy', arr=array_doc_vectors)
#array_doc_vectors = np.load(file='data/document_vectors.npy')

# associate base_path with document vectors
list_doc_vectors = array_doc_vectors.tolist()
series_doc_vectors = pd.Series(list_doc_vectors)
df_output = pd.DataFrame({'base_path': df['base_path'],
                          'document_vectors': series_doc_vectors})

# save dataframe
pd.to_pickle(obj=df_output, filepath_or_buffer='data/df_document_vectors.pkl')
