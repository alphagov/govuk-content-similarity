import numpy as np
import pandas as pd

# filter document types
DOC_TYPES_TO_REMOVE = [
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

COLS_TO_KEEP = ['document_type', 'base_path', 'content_id', 'first_published_at', 'text', 'title']


def create_content_store_mask(content_store_df,
                              doc_types_to_remove=DOC_TYPES_TO_REMOVE):
    doc_type_mask = content_store_df['document_type'].isin(doc_types_to_remove)
    # filter dates
    date_mask = content_store_df['first_published_at'].str[:4].fillna('2000').astype(int) > 2000
    # filter withdrawn documents that we want to exclude
    withdrawn_mask = content_store_df['withdrawn']
    # combine masks
    return ~withdrawn_mask & date_mask & ~doc_type_mask


def get_relevant_subset_of_content_store(content_store_df,
                                         doc_types_to_remove=DOC_TYPES_TO_REMOVE,
                                         cols_to_keep=COLS_TO_KEEP
                                         ):
    # This currently returns quite a large subset, which can make processing very slow
    # select fewer document types, and/or only more recently published documents to get a smaller DF
    content_mask = create_content_store_mask(content_store_df, doc_types_to_remove=doc_types_to_remove)
    return content_store_df.loc[content_mask, cols_to_keep].copy()


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


def get_lists_of_embeddings_and_text(subset_content_df, document_embedding_fn, embedding_dimension):
    # initialise an empty array for embeddings
    collected_doc_embeddings = np.zeros((subset_content_df.shape[0], embedding_dimension))
    collected_doc_text = []

    # fill array with embeddings for all docs
    # and store the original raw text extracted from the documents
    for i in range(subset_content_df.shape[0]):
        doc = subset_content_df.iloc[i]['text']
        try:
            extracted_sentences = extract_sentences(doc)
        except AttributeError:
            collected_doc_text.append('')
            continue
        if len(extracted_sentences) > 0:
            doc_embedding = document_embedding_fn(extracted_sentences)
            collected_doc_embeddings[i, :] = doc_embedding
            collected_doc_text.append(doc)
        else:
            collected_doc_text.append('')
        if i % 1000 == 0:
            progress = i / subset_content_df.shape[0]
            print('%s' % float('%.2g' % progress))
    return collected_doc_embeddings, collected_doc_text


def embeddings_and_text_to_dfs(subset_content_df, collected_doc_embeddings, collected_doc_text):
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

    return embeddings_df, text_df
