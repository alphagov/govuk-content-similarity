from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

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
                                         cols_to_keep=COLS_TO_KEEP):
    # This currently returns quite a large subset, which can make processing very slow
    # select fewer document types, and/or only more recently published documents to get a smaller DF
    content_mask = create_content_store_mask(content_store_df, doc_types_to_remove=doc_types_to_remove)
    return content_store_df.loc[content_mask, cols_to_keep].copy()


# sentences may be more performant
def extract_sentences(original_text):
    """
    Splits paragraph text into list of sentences. \n
    Note: If text contains xml tags, this does not remove them.

    :param original_text: Document text to split into list of sentences.
    :return: List of sentences from document text passed in.
    """
    return original_text.split('. ')


def get_lists_of_embeddings_and_text(subset_content_df: pd.DataFrame,
                                     document_embedding_fn: Callable[[list], np.ndarray],
                                     embedding_dimension: int) -> (np.array, list):
    """
    Compute embeddings from documents.

    :param subset_content_df: Dataframe of documents you want to compute embeddings from.
    :param document_embedding_fn: Function to use for computing document embeddings, takes a list of sentences.
    :param embedding_dimension: Number of dimensions/columns for embedding vectors.
    :return: An array of document embedding vectors and list of text these embeddings relate to.
    """
    # initialise an empty array for embeddings
    collected_doc_embeddings = np.zeros((subset_content_df.shape[0], embedding_dimension))
    collected_doc_text = []

    # fill array with embeddings for all docs
    # and store the original raw text extracted from the documents
    for i in tqdm(range(subset_content_df.shape[0])):
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


def get_cosine_from_similarity(similarity):
    """
    converts the similarity distance metric into a cosine of the angle between the vectors

    Annoy uses Euclidean distance of normalized vectors for its angular distance,
    which for two vectors u,v is equal to sqrt(2(1-cos(u,v)))

    Returns a value between 0 and 1, a "similarity score" where 0 is not similar at all
    and 1 is practically identical
    """
    cosine_angle = 1 - (similarity ** 2) / 2
    return cosine_angle


def get_similar_docs(base_path_idx_lookup_dict, content_text_df, annoy_index, base_path, no_of_results):
    """
    prints the 3 most similar pieces of content, with a similarity score between 0 and 1 (cosine)

    could parameterise how many results to return in future
    this needs tests
    """
    try:
        source_textdf_idx = base_path_idx_lookup_dict[base_path]
    except KeyError:
        return f'sorry, base_path {base_path} not found in our lookup'
    source_text_data = content_text_df.iloc[source_textdf_idx]
    if source_text_data['doc_text'] == np.nan:
        return f'sorry, there\'s no text in the content item {base_path}'

    results = np.array(annoy_index.get_nns_by_item(source_textdf_idx, no_of_results + 1, include_distances=True))
    print('query doc: ')
    #     display(HTML(f"""<a href="https://www.gov.uk{base_path}" target="_blank">{source_text_data['title']}</a>"""))
    print(source_text_data['title'])
    print(f"https://www.gov.uk{source_text_data['base_path']}")
    print('first_published_at: ' + source_text_data['first_published_at'][:10])

    print('\n similar content: \n')

    for i in range(1, no_of_results + 1):
        cosine_angle = get_cosine_from_similarity(results[1, i])
        text_data = content_text_df.iloc[int(results[0, i])]

        print(text_data['title'])
        print(f"https://www.gov.uk{text_data['base_path']}")
        print('first_published_at: ' + text_data['first_published_at'][:10])
        print(f"similarity score: {round(cosine_angle, 2)}")
        print("----")
