# aim: test sentence_transformers
import numpy as np
import pandas as pd
from ast import literal_eval
import os
from sentence_transformers import SentenceTransformer

# load model: specified cpu as the data batches are small
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens', device='cuda')

# import data from the content store
content_store_df = pd.read_csv("/home/james/Downloads/preprocessed_content_store_210920.csv",
                               compression='gzip', delimiter="\t", low_memory=False)
doc_types = ['press_release', 'news_story', 'speech', 'world_news_story']
doc_mask = content_store_df['document_type'].isin(doc_types)
date_mask = content_store_df['first_published_at'].str[:4].fillna('2000').astype(int) > 2015
live_mask = content_store_df['withdrawn'] == 'False'
content_mask = live_mask & date_mask & doc_mask
cols_keep = ['document_type', 'content_id', 'first_published_at', 'details']
subset_content_df = content_store_df.loc[content_mask, cols_keep].copy()
subset_content_df['details'] = subset_content_df['details'].map(literal_eval)


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
    returns the average of sentence embeddings for a document
    note this a different design to the USE embeddings
    '''
    embedding = model.encode(paragraphs)
    average_embedding = embedding.mean(axis=0)
    return average_embedding


# declare empty array, note the embedding dimension of 768 for sbert
collected_doc_embeddings = np.zeros((subset_content_df.shape[0], 768))

# create the document embeddings
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

# convert the numpy array to dataframe for the embeddings
embeddings_df = pd.DataFrame(collected_doc_embeddings)
embeddings_df['content_id'] = subset_content_df['content_id'].to_list()

# declare empty list for doc text
collected_doc_text = []
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

# convert doc text to dataframe with content id as key
text_df = pd.DataFrame({'content_id': subset_content_df['content_id'].to_list(),
                        'doc_text': collected_doc_text})


# output data
# distilbert-base-nli-stsb-mean-tokens
os.chdir('/home/james/Documents/gds_nlp/search_documents/data')
text_df.to_csv('text_distilbert_base_df.csv', index=False, header=True, mode='w')
embeddings_df.to_csv('embeddings_distilbert_base_df.csv', index=False, header=True, mode='w')
