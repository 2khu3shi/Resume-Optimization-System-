import tempfile
import os
import docx2txt
from pdfminer.high_level import extract_text as extract_pdf_text

def extract_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        text = extract_pdf_text(tmp_path)
        os.remove(tmp_path)
        return text
    elif uploaded_file.type in [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword"
    ]:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        text = docx2txt.process(tmp_path)
        os.remove(tmp_path)
        return text
    else:
        return uploaded_file.read().decode('utf-8', errors='ignore')

def extract_semantic_chunks(text, nlp, chunk_size=300):
    doc = nlp(text)
    chunks = []
    temp = ""
    for sent in doc.sents:
        temp += sent.text.strip() + " "
        if len(temp) >= chunk_size:
            chunks.append(temp.strip())
            temp = ""
    if temp:
        chunks.append(temp.strip())
    return chunks

def extract_skills_with_pos(doc, nlp):
    from spacy.matcher import Matcher
    matcher = Matcher(nlp.vocab)
    pattern = [{"POS": "NOUN"}, {"POS": "NOUN", "OP": "?"}]
    matcher.add("SKILL_PHRASE", [pattern])
    matches = matcher(doc)
    phrases = set()
    for _, start, end in matches:
        span = doc[start:end]
        if len(span.text.strip()) > 3:
            phrases.add(span.text.lower())
    return list(phrases)
