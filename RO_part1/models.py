import spacy
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

def load_models():
    try:
        nlp = spacy.load("en_core_web_lg")
    except OSError:
        from spacy.cli import download
        download("en_core_web_lg")
        nlp = spacy.load("en_core_web_lg")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    model.save("sentence_model")
    model = SentenceTransformer("sentence_model")
    kw_model = KeyBERT()

    return nlp, model, kw_model
