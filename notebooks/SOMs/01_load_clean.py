from time import time
import os
import re
import multiprocessing

import pandas as pd
import spacy


def clean_text(doc):
    """
    Lemmatises and remove stopwords.

    :param doc: spacy Doc object to lemmatise and remove stop words from
    :return: spacy Doc object that has been lemmatised and have had stop words removed
    """

    txt = [token.lemma_ for token in doc if not token.is_stop]
    # remove one or two word sentences as are not beneficial to @ord2Vec
    if len(txt) > 2:
        return ' '.join(txt)


DATA_DIR = os.getenv("DATA_DIR")
FILE_NAME = "preprocessed_content_store_180920.csv.gz"
n_cores = multiprocessing.cpu_count() - 1

dict_header = {'base_path': object,
               'content_id': object,
               'title': object,
               'description': object,
               'publishing_app': object,
               'document_type': object,
               'details': object,
               'text': object,
               'organisations': object,
               'taxons': object,
               'step_by_steps': object,
               'details_parts': object,
               'first_published_at': object,
               'public_updated_at': object,
               'updated_at': object,
               'finder': object,
               'facet_values': object,
               'facet_groups': object,
               'has_brexit_no_deal_notice': bool,
               'withdrawn': bool,
               'withdrawn_at': object,
               'withdrawn_explanation': object}
list_header_date = ['first_published_at',
                    'public_updated_at',
                    'updated_at',
                    'withdrawn_at']

# load data
df = pd.read_csv(filepath_or_buffer=DATA_DIR + "/" + FILE_NAME,
                 compression='gzip',
                 encoding='utf-8',
                 sep='\t',
                 header=0,
                 names=list(dict_header.keys()),
                 dtype=dict_header,
                 parse_dates=list_header_date)

del DATA_DIR, FILE_NAME, dict_header, list_header_date

# remove rows with no text
df = df.dropna(subset=['text'], axis=0)

# remove non-alphabetic characters
df['text_clean'] = [re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df['text']]

# focus on df['text'] column and a smaller subset for testing purposes
df_small = df[['base_path', 'text', 'text_clean', 'details']].iloc[:50000].copy()


# disable ner for speed
nlp = spacy.load('en', disable=['ner', 'parser'])
# lemmatise and remove stopwords
t = time()
df_small['text_clean'] = [clean_text(doc) for doc in nlp.pipe(df_small['text_clean'], batch_size=5000)]
print('Time to clean up everything: {} minutes'.format(round((time() - t) / 60, 2)))
del t

# remove NAs and duplicates
df_small = df_small.dropna(axis=0, subset=['text_clean']).drop_duplicates()

# export so can do document embeddings and SOMs in later script
df_small.to_csv(path_or_buf='data/df.csv', index=False)
