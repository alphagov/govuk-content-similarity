# aim: test sentence_transformers
import pandas as pd
from sentence_transformers import SentenceTransformer

import src.utils.embedding_utils as utils

# load model: specified cpu as the data batches are small, can use cuda as device if cuda is available
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens',
                            device='cpu')

# import data from the content store
content_store_file_path = 'data/raw/preprocessed_content_store_141020.csv'
content_store_df = pd.read_csv(content_store_file_path,
                               compression='gzip', delimiter="\t", low_memory=False)

subset_content_df = utils.get_relevant_subset_of_content_store(
    content_store_df)


def document_embedding(paragraphs):
    '''
    returns the average of sentence embeddings for a document
    note this a different design to the USE embeddings
    '''
    embedding = model.encode(paragraphs)
    average_embedding = embedding.mean(axis=0)
    return average_embedding


# declare empty array, note the embedding dimension of 768 for sbert
collected_doc_embeddings, collected_doc_text = utils.get_lists_of_embeddings_and_text(
    subset_content_df, document_embedding, 768)


embeddings_df, text_df = utils.embeddings_and_text_to_dfs(
    subset_content_df, collected_doc_embeddings, collected_doc_text)

# output data
# distilbert-base-nli-stsb-mean-tokens
text_df.to_csv('data/processed/text_distilbert_base_df_v3_small.csv', index=False, header=True, mode='w')
embeddings_df.to_csv('data/processed/embeddings_distilbert_base_df_v3_small.csv', index=False, header=True, mode='w')
