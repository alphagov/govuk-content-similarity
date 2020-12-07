from src.utils.constants import CONTENT_STORE_HEADER, CONTENT_STORE_DATE
from src.utils.preprocess import clean_text

from time import time
import os
import multiprocessing

import pandas as pd
import spacy


DATA_DIR = os.getenv("data/raw")
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
df[['base_path', 'text', 'text_clean']].to_csv(path_or_buf='data/interim/df.csv', index=False)
