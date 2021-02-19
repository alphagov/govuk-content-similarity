from src.utils.preprocess import clean_text
import spacy


# disable ner for speed
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])


def test_clean_text(text):
    text['text_process'] = [clean_text(doc) for doc in nlp.pipe(text['text_dirty'])]
    assert text['text_process'].equals(text['text_clean'])
