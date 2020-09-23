import os
import pandas as pd

DATA_DIR = os.getenv("DATA_DIR")
FILE_NAME = "preprocessed_content_store_180920.csv.gz"

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

# focus on df['text'] column and a smaller subset for testing purposes
df_small = df[['base_path', 'text', 'details']].iloc[:10000].copy()
