import joblib
#import spacy
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import clean_resume

def load_components():
    tfidf = joblib.load("models/tfidf_vectorizer.joblib")
    le = joblib.load("models/label_encoder.joblib")
    category_vectors = joblib.load("models/category_vectors.joblib")
    return tfidf, le, category_vectors

def compute_similarity_score(resume_text, category, tfidf, le, category_vectors):
    cleaned_text = clean_resume(resume_text)
    user_vector = tfidf.transform([cleaned_text])
    cat_index = le.transform([category])[0]
    cat_vectors = category_vectors[cat_index]
    sims = cosine_similarity(user_vector, cat_vectors)
    return sims.mean() * 100

# valid_skills = set([
#     "python", "flask", "django", "react", "angular", "node.js", "express", "mongodb", "sql", "mysql",
#     "aws", "azure", "docker", "kubernetes", "tensorflow", "pytorch", "nlp", "machine learning", 
#     "deep learning", "html", "css", "javascript", "typescript", "pandas", "numpy", "git", "github", 
#     "linux", "bash", "java", "c++", "c#", "r", "matlab", "power bi", "tableau", "hadoop", "spark",
#     "mern", "mevn", "rest api", "graphql", "firebase", "postman", "selenium", "jest", "mocha", "ci/cd"
# ])

# nlp = spacy.load("en_core_web_sm")

# def extract_skills_with_pos(doc):
#     return [token.text.lower() for token in doc if token.pos_ in {"NOUN", "PROPN"} and token.text.lower() in valid_skills]

# def extract_semantic_chunks(resume_text):
#     doc = nlp(resume_text.lower())
#     chunks = []
#     for chunk in doc.noun_chunks:
#         text = chunk.text.strip().lower()
#         if text in valid_skills:
#             chunks.append(text)
#     return list(set(chunks))

# def analyze_resume(resume_text):
#     doc = nlp(resume_text)
#     sections = {"experience": [], "skills": [], "education": [], "projects": []}
#     lines = resume_text.split('\\n')
#     current_section = None
#     for line in lines:
#         line_lower = line.lower().strip()
#         if "experience" in line_lower:
#             current_section = "experience"
#         elif "skills" in line_lower:
#             current_section = "skills"
#         elif "education" in line_lower:
#             current_section = "education"
#         elif "projects" in line_lower:
#             current_section = "projects"
#         elif current_section and line.strip():
#             sections[current_section].append(line.strip())

#     skills = extract_skills_with_pos(doc)
#     return {
#         "sections": sections,
#         "skills": list(set(skills)),
#         "semantic_chunks": extract_semantic_chunks(resume_text)
#     }

# def suggest_keywords_spacy(resume_text):
#     doc = nlp(resume_text.lower())
#     present_skills = set()

#     for chunk in doc.noun_chunks:
#         phrase = chunk.text.strip()
#         if phrase in valid_skills:
#             present_skills.add(phrase)

#     for token in doc:
#         if token.text in valid_skills:
#             present_skills.add(token.text)

#     missing_skills = sorted(valid_skills - present_skills)
#     return missing_skills[:10]
