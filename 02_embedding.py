from tqdm import tqdm
import multiprocessing

import pandas as pd
import numpy as np
from gensim.test.utils import get_tmpfile
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


n_cores = multiprocessing.cpu_count() - 1

df = pd.read_csv(filepath_or_buffer='data/df.csv')

# create TaggedDocument for doc2vec DBOW
# will use indices of sentences as tags
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(df['text_clean'])]

# instantiate a doc2vec DBOW model
model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample=0, workers=n_cores)
model_dbow.build_vocab([x for x in tqdm(documents)])
model_dbow.train(documents, total_examples=model_dbow.corpus_count, epochs=1)

# save trained model
file_name = get_tmpfile("my_doc2vec_model")
model_dbow.save(file_name)
#model_dbow = Doc2Vec.load(file_name)

# store document vectors in numpy array for minisom
# note: can't use list comprehension as it goes from 1-509,529
array_doc_vectors = np.array([])
for i in range(len(model_dbow.docvecs) - 1):
    np.append(array_doc_vectors, model_dbow.docvecs[i])
