from src.utils.constants import CONTENT_STORE_HEADER, CONTENT_STORE_DATE

from time import time
import os
import multiprocessing

import pandas as pd
import spacy


def clean_text(doc):
    """
    Lemmatises and remove stopwords.
    Note: Have not moved to a separate functions script as this function depends on creation of spacy Doc object \n
          haven't quite figured out a way to isolate that appropriately if moved to a separate script.

    :param doc: spacy Doc object to lemmatise and remove stop words from
    :return: spacy Doc object that has been lemmatised and have had stop words removed
    """

    txt = [token.lemma_ for token in doc if not token.is_stop]
    # remove one or two word sentences as are not beneficial to Word2Vec
    if len(txt) > 2:
        return ' '.join(txt)


DATA_DIR = os.getenv("DATA_DIR")
FILE_NAME = "preprocessed_content_store_180920.csv.gz"
n_cores = multiprocessing.cpu_count() - 1

# load data
df = pd.read_csv(filepath_or_buffer=DATA_DIR + "/" + FILE_NAME,
                 compression='gzip',
                 encoding='utf-8',
                 sep='\t',
                 header=0,
                 names=list(CONTENT_STORE_HEADER.keys()),
                 dtype=CONTENT_STORE_HEADER,
                 parse_dates=CONTENT_STORE_DATE)

# remove rows with no text
df = df.dropna(subset=['text'], axis=0)

# remove non-alphabetic characters
df['text_clean'] = df['text'].str.lower()
df['text_clean'] = df['text_clean'].str.replace(pat=r"[^A-Za-z']+", repl=' ')

# disable ner for speed
nlp = spacy.load('en', disable=['ner', 'parser'])
# lemmatise and remove stopwords
t = time()
df['text_clean'] = [clean_text(doc) for doc in nlp.pipe(df['text_clean'], batch_size=5000)]
print(f'Time to clean up everything: {round((time() - t) / 60, 2)} minutes')

# remove NAs and duplicates
df = df.dropna(axis=0, subset=['text_clean']).drop_duplicates()

# export so can do document embeddings and SOMs in later script
df[['base_path', 'text', 'text_clean']].to_csv(path_or_buf='data/df.csv', index=False)
