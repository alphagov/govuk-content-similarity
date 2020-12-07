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
import pandas as pd

import src.utils.embedding_utils as utils

# tests presence of GPUs, to run the Transformer, GPU is necessary
# otherwise looking at a run time of order of days for scoring the whole corpus

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# NB: the model for GPUs seemed more performant for us

# DAN model is lighter, A stands for averaging
# downloaded from https://tfhub.dev/google/universal-sentence-encoder/4
# moved to data/external, and unzipped
model = hub.load('data/external/universal-sentence-encoder_4')

# Transformer model, more performant, runs on GPU, if available
# model = hub.load('/home/james/Downloads/universal-sentence-encoder-large_5')


def embed(input_text):
    return model(input_text)


# import data from the content store
# if you download the content store data using the aws-cli instead of the browser the file will end .csv.gz
content_store_file_path = 'data/raw/preprocessed_content_store_141020.csv'
content_store_df = pd.read_csv(content_store_file_path, compression='gzip', delimiter="\t", low_memory=False)

subset_content_df = utils.get_relevant_subset_of_content_store(
    content_store_df)


def get_use_document_embedding(sentences):
    """
    average embeddings across sentences using universal sentence encoder
    """
    embedding = embed(sentences)
    average_embedding = tf.math.reduce_mean(embedding, axis=0).numpy()
    return average_embedding


collected_doc_embeddings, collected_doc_text = utils.get_lists_of_embeddings_and_text(
    subset_content_df, get_use_document_embedding, 512)

embeddings_df, text_df = utils.embeddings_and_text_to_dfs(
    subset_content_df, collected_doc_embeddings, collected_doc_text)

# output dataframes
text_df.to_csv('data/processed/text_use_20201014_df_v3_small.csv', index=False, header=True, mode='w')

# It's worth investigating if it's better to save the embeddings as a pandas.DataFrame() or a numpy.array().
# As these embeddings are typically quite big, it might be better to store as a .npy file?
embeddings_df.to_csv('data/processed/embeddings_20201014_df_v3_small.csv', index=False, header=True, mode='w')
