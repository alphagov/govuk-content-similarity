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
import os

# tests presence of GPUs, to run the Transformer, GPU is necessary
# otherwise looking at a run time of order of days for scoring the whole corpus

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# DAN model is lighter, A stands for averaging
# downloaded from https://tfhub.dev/google/universal-sentence-encoder/4
# moved to data/external, and unzipped
model = hub.load('data/external/universal-sentence-encoder_4')

# Transformer model, more performant, runs on GPU, if available
# model = hub.load('/home/james/Downloads/universal-sentence-encoder-large_5')


def embed(input):
    return model(input)


# import data from the content store
content_store_file_path = os.path.join('data/raw/preprocessed_content_store_141020.csv')
content_store_df = pd.read_csv(content_store_file_path, compression='gzip', delimiter="\t", low_memory=False)

# filter document types
doc_types_to_remove = [
    'aaib_report',
    'answer',
    'asylum_support_decision',
    'business_finance_support_scheme',
    'cma_case',
    'countryside_stewardship_grant',
    'drug_safety_update',
    'employment_appeal_tribunal_decision',
    'employment_tribunal_decision',
    'esi_fund',
    'export_health_certificate',
    'help_page',
    'html_publication',
    'international_development_fund',
    'maib_report',
    'manual',
    'manual_section',
    'medical_safety_alert',
    'ministerial_role',
    'person',
    'protected_food_drink_name',
    'raib_report',
    'research_for_development_output',
    'residential_property_tribunal_decision',
    'service_standard_report',
    'simple_smart_answer',
    'statutory_instrument',
    'tax_tribunal_decision',
    'utaac_decision'
]
doc_type_mask = content_store_df['document_type'].isin(doc_types_to_remove)

# filter dates
date_mask = content_store_df['first_published_at'].str[:4].fillna('2000').astype(int) > 2000

# filter live documents
live_mask = content_store_df['withdrawn'] == 'False'

# combine masks
content_mask = live_mask & date_mask & ~doc_type_mask

cols_keep = ['document_type', 'base_path', 'content_id', 'first_published_at', 'text', 'title']
subset_content_df = content_store_df.loc[content_mask, cols_keep].copy()


def cos_sim(a, b):
    """Takes 2 vectors a, b and returns the cosine similarity
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


# sentences may be more performant
def extract_sentences(original_text):
    ''' takes raw string text from gov uk and returns extracted paragraphs
    still contains xml tags'''
    return original_text.split('. ')


def document_embedding(paragraphs):
    """
    average embeddings across sentences
    """
    embedding = embed(paragraphs)
    average_embedding = tf.math.reduce_mean(embedding, axis=0).numpy()
    return average_embedding


# initialise an empty array for embeddings
collected_doc_embeddings = np.zeros((subset_content_df.shape[0], 512))
collected_doc_text = []

# fill array with embeddings for all docs
# and store the original raw text extracted from the documents
for i in range(subset_content_df.shape[0]):
    # try:
    doc = subset_content_df.iloc[i]['text']
    # except KeyError:
    #     collected_doc_text.append('')
    #     continue
    try:
        extracted_sentences = extract_sentences(doc)
    except AttributeError:
        collected_doc_text.append('')
        continue
    if len(extracted_sentences) > 0:
        doc_embedding = document_embedding(extracted_sentences)
        collected_doc_embeddings[i, :] = doc_embedding
        collected_doc_text.append(doc)
    else:
        collected_doc_text.append('')
    if i % 1000 == 0:
        progress = i / subset_content_df.shape[0]
        print('%s' % float('%.2g' % progress))


# converts embeddings array into dataframe, with content id as unique key
embeddings_df = pd.DataFrame(collected_doc_embeddings)
embeddings_df['content_id'] = subset_content_df['content_id'].to_list()

# converts the raw text into a dataframe
text_df = pd.DataFrame({'content_id': subset_content_df['content_id'].to_list(),
                        'base_path': subset_content_df['base_path'].to_list(),
                        'title': subset_content_df['title'].to_list(),
                        'doc_text': collected_doc_text,
                        'document_type': subset_content_df['document_type'].to_list(),
                        'first_published_at': subset_content_df['first_published_at'].to_list()})

# output dataframes
os.chdir('data/processed')
text_df.to_csv('text_use_20201014_df_v2.csv', index=False, header=True, mode='w')
embeddings_df.to_csv('embeddings_20201014_df_v2.csv', index=False, header=True, mode='w')
