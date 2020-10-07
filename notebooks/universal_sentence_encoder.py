'''
demo of the Universal Sentence Encoder (USE) on some gov docs
to run this, you must first download:
- preprocessed_content_store_210920.csv from Google Drive
- USE models, which come in two flavours, if you do not have a graphics card, go for DAN
https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder
https://tfhub.dev/google/universal-sentence-encoder-large/5

'''
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import pandas as pd
from ast import literal_eval
import os

# tests presence of GPUs, to run the Transformer, GPU is necessary
# otherwise looking at a run time of order of days for scoring the whole corpus

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# DAN model, lighter A stands for averaging
# model = hub.load('/home/james/Downloads/universal-sentence-encoder_4')
# Transformer model, more performant, runs on GPU, if available
model = hub.load('/home/james/Downloads/universal-sentence-encoder-large_5')


def embed(input):
    return model(input)


# import data from the content store
content_store_df = pd.read_csv("/home/james/Downloads/preprocessed_content_store_210920.csv",
                               compression='gzip', delimiter="\t", low_memory=False)
# filter document types
doc_types = ['press_release', 'news_story', 'speech', 'world_news_story', 'guidance']
doc_mask = content_store_df['document_type'].isin(doc_types)
# filter dates
date_mask = content_store_df['first_published_at'].str[:4].fillna('2000').astype(int) > 2000
# filter live documents
live_mask = content_store_df['withdrawn'] == 'False'
content_mask = live_mask & date_mask & doc_mask
cols_keep = ['document_type', 'content_id', 'first_published_at', 'details']
subset_content_df = content_store_df.loc[content_mask, cols_keep].copy()
subset_content_df['details'] = subset_content_df['details'].map(literal_eval)


def cos_sim(a, b):
    """Takes 2 vectors a, b and returns the cosine similarity
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def clean_xml(original_text):
    ''' strips out xml tagging from string'''
    extracted_sentence = []
    start_idx = 1
    end_idx = 1
    while (start_idx > 0) and (end_idx > 0):
        end_idx = original_text.find('<')
        if end_idx >= 0:
            extracted_sentence.append(original_text[:end_idx])
            start_idx = original_text.find('>')
            if (start_idx >= 0):
                original_text = original_text[start_idx + 1:]
    if len(original_text) > 0:
        extracted_sentence.append(original_text)
    return str(''.join(extracted_sentence))


def extract_paragraphs(original_text):
    ''' takes raw string text from gov uk and returns extracted paragraphs
    still contains xml tags'''
    extracted_paragraphs = []
    start_idx = 1
    end_idx = 1
    while (start_idx >= 0) and (end_idx >= 0):
        start_idx = original_text.find('<p>')
        end_idx = original_text.find('</p>')
        if (start_idx >= 0) and (end_idx >= 0):
            if (end_idx - start_idx) > 3:
                cleaned_text_segment = clean_xml(original_text[start_idx + 3:end_idx])
                extracted_paragraphs.append(cleaned_text_segment)
            original_text = original_text[end_idx + 3:]
    return extracted_paragraphs


def document_embedding(paragraphs):
    '''
    average embeddings across sentences
    '''
    embedding = embed(paragraphs)
    average_embedding = tf.math.reduce_mean(embedding, axis=0).numpy()
    return average_embedding


# initialise an empty array for embeddings
collected_doc_embeddings = np.zeros((subset_content_df.shape[0], 512))

# fill array with embeddings for all docs
for i in range(subset_content_df.shape[0]):
    try:
        doc = subset_content_df.iloc[i]['details']['body']
    except KeyError:
        continue
    extracted_paragraphs = extract_paragraphs(doc)
    if len(extracted_paragraphs) > 0:
        doc_embedding = document_embedding(extracted_paragraphs)
        collected_doc_embeddings[i, :] = doc_embedding
    if i % 1000 == 0:
        progress = i / subset_content_df.shape[0]
        print('%s' % float('%.2g' % progress))

# converts embeddings array into dataframe, with content id as unique key
embeddings_df = pd.DataFrame(collected_doc_embeddings)
embeddings_df['content_id'] = subset_content_df['content_id'].to_list()

# initialise list for storing raw text
collected_doc_text = []

# store the original raw text extracted from the documents
for i in range(subset_content_df.shape[0]):
    try:
        doc = subset_content_df.iloc[i]['details']['body']
    except KeyError:
        collected_doc_text.append('')
        continue
    extracted_paragraphs = extract_paragraphs(doc)
    if len(extracted_paragraphs) > 0:
        collected_doc_text.append(' '.join(extracted_paragraphs))
    else:
        collected_doc_text.append('')
    if i % 1000 == 0:
        progress = i / subset_content_df.shape[0]
        print('%s' % float('%.2g' % progress))

# converts the raw text into a dataframe
text_df = pd.DataFrame({'content_id': subset_content_df['content_id'].to_list(),
                        'doc_text': collected_doc_text,
                        'document_type': subset_content_df['document_type'].to_list(),
                        'first_published_at': subset_content_df['first_published_at'].to_list()})


# output dataframes
os.chdir('/home/james/Documents/gds_nlp/search_documents/data')
text_df.to_csv('text_use_large_2000_df.csv', index=False, header=True, mode='w')
embeddings_df.to_csv('embeddings_use_large_2000_df.csv', index=False, header=True, mode='w')
