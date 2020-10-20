# aim: illustrate using the Facebook FAISS package
# on conda forge there are various versions of this library
# pytorch/faiss-cpu, pytorch/faiss-gpu are going to be explored here
# using vectors created using the google universal sentence encoder DAN v4 model
# instructions on using here https://github.com/facebookresearch/faiss/wiki
# note that the Facebook index is built using L2 distances, but if you normalise vectors
# before hand, so they have unit length, then L2**2 = 2-2*cosine_distance
# L2 is no use for document similarity

import pandas as pd
import numpy as np
import os
import faiss
from universal_sentence_encoder import document_embedding

os.chdir('/home/james/Documents/gds_nlp/search_documents/data')
embeddings_df = pd.read_csv('embeddings_df.csv')
text_df = pd.read_csv('text_df.csv')

embeddings = np.ascontiguousarray(embeddings_df.iloc[:, :512], dtype=np.float32)
content_id = embeddings_df['content_id'].to_list()

embedding_dimension = 512
index = faiss.IndexFlatL2(embedding_dimension)   # build the index
faiss.normalize_L2(embeddings)
index.add(embeddings)                  # add vectors to the index
print(index.ntotal)
# del(index)

test_text = ['The first person cured of HIV - Timothy Ray Brown - has died from cancer.',
             'Mr Brown, who was also known as "the Berlin patient", was given a bone marrow transplant from a donor \
             who was naturally resistant to HIV in 2007.',
             'It meant he no longer needed anti-viral drugs and he remained free of the virus, which can lead to Aids, \
             for the rest of his life.',
             'The International Aids Society said Mr Brown gave the world hope that an HIV cure was possible.',
             'Mr Brown, 54, who was born in the US, was diagnosed with HIV while he lived in Berlin in 1995. Then in \
             2007 he developed a type of blood cancer called acute myeloid leukaemia.',
             'His treatment involved destroying his bone marrow, which was producing the cancerous cells, and then \
             having a bone marrow transplant.',
             'The transfer came from a donor that had a rare mutation in part of their DNA called the CCR5 gene.']

embedding = document_embedding(test_text)
query = embedding.reshape(1, 512)
faiss.normalize_L2(query)
index.search(query, 5)

text_df['doc_text'][21951]
text_df['doc_text'][1934]
text_df['doc_text'][24591]
