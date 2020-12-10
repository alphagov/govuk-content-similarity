'''
use Approximate Nearest Neighbours to see whether the doc vectors work
requires the Spotify Annoy package from Conda
depends on previous creation of two data frames:
- embeddings_use_large_2000_df.csv
- text_use_large_2000_df.csv
or whatever you have chosen to call them in the previous step of creating embedding vectors
it makes sense to create various differently named flavours of embedding vectors and adding
them to the settings json beneath, so that you can easily compare different vectors
'''

import pandas as pd
import numpy as np
import os
from annoy import AnnoyIndex

import src.utils.embedding_utils as utils

os.chdir('data/processed')

# BUILDING THE INDEX FOR APPROXIMATE NEAREST NEIGHBOURS

# json of settings for the testing
settings = {'dilbert': {'embeddings': 'embeddings_distilbert_base_df.csv',
                        'text': 'text_distilbert_base_df.csv',
                        'embedding_dim': 768,
                        'ann_index': 'sbert.ann'},
            'use_large': {'embeddings': 'embeddings_use_large_df.csv',
                          'text': 'text_use_large_df.csv',
                          'embedding_dim': 512,
                          'ann_index': 'use.ann'},
            'use_2000': {'embeddings': 'embeddings_use_large_2000_df.csv',
                         'text': 'text_use_large_2000_df.csv',
                         'embedding_dim': 512,
                         'ann_index': 'use_2000.ann'},
            'use_20201014': {'embeddings': 'embeddings_20201014_df.csv',
                             'text': 'text_use_20201014_df.csv',
                             'embedding_dim': 512,
                             'ann_index': 'use_20201014.ann'},
            'use_20201014_v2': {'embeddings': 'embeddings_20201014_df_v2.csv',
                                'text': 'text_use_20201014_df_v2.csv',
                                'embedding_dim': 512,
                                'ann_index': 'use_20201014_v2.ann'}
            }

# select which embeddings you are going to analyse
embedding_type = 'use_20201014_v2'

# pass in the settings above for importing the embeddings data and matching text
embeddings_df = pd.read_csv(settings[embedding_type]['embeddings'])
text_df = pd.read_csv(settings[embedding_type]['text'])
embeddings = np.asarray(embeddings_df.iloc[:, :settings[embedding_type]['embedding_dim']])
content_id = embeddings_df['content_id'].to_list()

# build the Annoy Index, f is the length of embedding vector to be used
# f is the length of the vectors, aka 'embedding dimension'
f = settings[embedding_type]['embedding_dim']
# declare an empty index which is going to be based on cosine similarity, aka 'angular'
t = AnnoyIndex(f, 'angular')
# add embedding vectors into the index, converting the values into a list before hand
for i in range(embeddings_df.shape[0]):
    v = [x for x in embeddings[i, :]]
    t.add_item(i, v)

# number of trees determines how refined the groupings are, in practice, it makes little
# difference to performance, in theory, it increases performance with greater number of trees
# you can specify thousands of trees -  no advantage seen in doing this though
t.build(10)
t.save(settings[embedding_type]['ann_index'])

# TESTING AN ANNOY INDEX

# load a previously trained Annoy Index (the one you saved two lines up)
u = AnnoyIndex(f, 'angular')
u.load(settings[embedding_type]['ann_index'])
# u.unload()


# in future might want to put this in a separate module, or look at get_similar_docs from embedding_utils
def print_cosine_and_texts(text_idx, max_length_string=1000, verbose=True):
    '''
    function for printing out details of query document and the best match
    '''
    results = np.array(u.get_nns_by_item(text_idx, 2, include_distances=True))
    cosine_angle = utils.get_cosine_from_similarity(results[1, 1])
    if verbose:
        print('cosine angle: ' + '%s' % float('%.2g' % cosine_angle))
        print("----")
        print('Index: ' + str(text_idx))
        print("----")
        print('query doc: ' + text_df['content_id'][results[0, 0]])
        print('date: ' + text_df['first_published_at'][results[0, 0]][:10])
        print("----")
        print(text_df['doc_text'][results[0, 0]][:max_length_string])
        print("----")
        print('best match: ' + text_df['content_id'][results[0, 1]])
        print('date: ' + text_df['first_published_at'][results[0, 1]][:10])
        print("----")
        print(text_df['doc_text'][results[0, 1]][:max_length_string])
    else:
        return(cosine_angle)


# pick a random document, and search the index, printing to screen
print_cosine_and_texts(np.random.randint(0, 100000))
# this is really how you can compare the ability of different embeddings
print_cosine_and_texts(3779, max_length_string=10000)

# collect a list of potentially interesting documents, inspecting results
list_interesting_indices = []
n = 1000
while n > 0:
    n -= 1
    text_idx = np.random.randint(0, 3000)
    cosine_angle = print_cosine_and_texts(text_idx, verbose=False)
    if cosine_angle > 0.9 and cosine_angle < 0.95:
        list_interesting_indices.append(text_idx)

# check count of collected docs which match your criteria above
len(list_interesting_indices)

# set counter
i = 0
# if you highlight the two lines beneath, and hit 'shift + enter' in vs code
# you can manually inspect the results
print_cosine_and_texts(list_interesting_indices[i])
i += 1


# LOOKING AT GUIDANCE ON VIRUS CONTENT
# this is just an illustration of finding content based on keywords and then looking for similar
# content

# search by date and content type:
doc_types = ['press_release', 'news_story', 'speech', 'world_news_story', 'guidance']
doc_types = ['guidance']
doc_mask = text_df['document_type'].isin(doc_types)
date_mask = text_df['first_published_at'].str[:4].fillna('2000').astype(int) >= 2020
text_mask = text_df['doc_text'].str.lower().str.contains('virus')
content_mask = date_mask & doc_mask & text_mask
cols_keep = ['document_type', 'content_id', 'first_published_at', 'doc_text']
subset_text_df = text_df.loc[content_mask, cols_keep].copy()
subset_text_df.shape

collected_guidance_ids = []
for i in subset_text_df.index.to_list():
    results = np.array(u.get_nns_by_item(i, 2, include_distances=True))
    cosine_angle = utils.get_cosine_from_similarity(results[1, 1])
    if cosine_angle > 0.8:
        collected_guidance_ids.append(i)

print_cosine_and_texts(96897)
i = 0
print_cosine_and_texts(collected_guidance_ids[i])
i += 1


# search by vector
# this illustrates how you can search for similar documents, based on any text
# being convered into an embedding schema
# requires the loading of the embedding model, which can take time to load, as its 1GB
# flake complains about the import of libraries here, but it will add to the run time of the script
# happy for this to be chopped and moved etc
'''
from universal_sentence_encoder import document_embedding

test_text = ['Britain will roll out COVID-19 vaccinations when they are ready based on clinical advice about who \
    should be prioritised, health minister Matt Hancock said on Monday, after a report that half the population \
    could miss out on the jabs.',
    'Asked about comments by the chair of the government vaccine taskforce to the Financial Times that vaccines \
    would probably only be available to less than half the UK population, Hancock said the taskforce had done \
    good work in procuring vaccines but that deployment was his department responsibility.',
    '“We will take the advice on the deployment of the vaccine, based on clinical advice from the Joint \
    Committee on vaccinations and immunizations,” Hancock told parliament.']

embedding = document_embedding(test_text)
results = np.array(u.get_nns_by_vector(embedding, 4, include_distances=True))

utils.get_cosine_from_similarity(results[1,0])
text_df['doc_text'][results[0,0]]
'''
