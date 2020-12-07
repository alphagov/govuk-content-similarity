import spacy


def clean_text(doc: spacy.tokens.Doc) -> spacy.tokens.Doc:
    """
    Lemmatises and remove stopwords.

    :param doc: spacy Doc object to lemmatise and remove stop words from
    :return: spacy Doc object that has been lemmatised and have had stop words removed
    """

    txt = [token.lemma_ for token in doc if not token.is_stop]
    # remove one or two word sentences as are not beneficial to Word2Vec
    if len(txt) > 2:
        return ' '.join(txt)
