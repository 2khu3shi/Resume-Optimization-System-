from collections import defaultdict
import numpy as np
from sentence_transformers import util
from utils import extract_semantic_chunks, extract_skills_with_pos

def analyze_job_description(jd_text, nlp, kw_model):
    doc = nlp(jd_text)
    entities = defaultdict(list)

    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "TECH", "SKILL", "DOMAIN"]:
            entities[ent.label_].append(ent.text)

    tech_keywords = [
        "python", "flask", "react", "aws", "docker", "kubernetes", "numpy", "pandas",
        "django", "fastapi", "node.js", "express", "typescript", "javascript",
        "html", "css", "mongodb", "postgresql", "mysql", "redis", "git",
        "github", "gitlab", "ci/cd", "jenkins", "tensorflow", "pytorch",
        "scikit-learn", "matplotlib", "seaborn", "linux", "bash", "azure",
        "gcp", "restapi", "graphql", "firebase", "lambda", "s3", "terraform",
        "ansible", "selenium", "beautifulsoup", "openai", "huggingface"
    ]
    for kw in tech_keywords:
        if kw.lower() in jd_text.lower():
            entities["SKILL"].append(kw)

    requirements = []
    for sent in doc.sents:
        if "require" in sent.text.lower() or "must have" in sent.text.lower():
            requirements.append(sent.text.strip())

    key_phrases = [kw[0] for kw in kw_model.extract_keywords(jd_text, top_n=15)]

    return {
        "entities": dict(entities),
        "requirements": requirements,
        "semantic_chunks": extract_semantic_chunks(jd_text, nlp),
        "keywords": key_phrases
    }

def analyze_resume(resume_text, nlp):
    doc = nlp(resume_text)
    sections = {"experience": [], "skills": [], "education": [], "projects": []}
    lines = resume_text.split('\n')
    current_section = None
    for line in lines:
        line_lower = line.lower().strip()
        if "experience" in line_lower:
            current_section = "experience"
        elif "skills" in line_lower:
            current_section = "skills"
        elif "education" in line_lower:
            current_section = "education"
        elif "projects" in line_lower:
            current_section = "projects"
        elif current_section and line.strip():
            sections[current_section].append(line.strip())

    skills = extract_skills_with_pos(doc, nlp)
    return {
        "sections": sections,
        "skills": list(set(skills)),
        "semantic_chunks": extract_semantic_chunks(resume_text, nlp)
    }

def semantic_match(jd_chunks, resume_chunks, resume_sections, model):
    jd_embeddings = model.encode(jd_chunks, convert_to_tensor=True)
    resume_embeddings = model.encode(resume_chunks, convert_to_tensor=True)
    cosine_scores = util.cos_sim(jd_embeddings, resume_embeddings)

    weights = []
    for chunk in resume_chunks:
        joined = "\n".join(resume_sections["experience"] + resume_sections["skills"])
        weights.append(1.2 if chunk in joined else 1.0)

    top_matches = []
    for i in range(len(jd_chunks)):
        scores = cosine_scores[i] * np.array(weights)
        top_idx = np.argmax(scores)
        top_matches.append({
            "jd_chunk": jd_chunks[i],
            "resume_chunk": resume_chunks[top_idx],
            "score": scores[top_idx].item()
        })

    overall_score = sum(match["score"] for match in top_matches) / len(top_matches)
    return {
        "overall_score": overall_score * 100,
        "matches": sorted(top_matches, key=lambda x: x["score"], reverse=True)
    }

def generate_suggestions(jd_analysis, resume_analysis, matches):
    suggestions = []
    valid_skills = set([
        "python", "flask", "django", "react", "angular", "node.js", "express", "mongodb", "sql", "mysql",
        "aws", "azure", "docker", "kubernetes", "tensorflow", "pytorch", "nlp", "machine learning",
        "deep learning", "html", "css", "javascript", "typescript", "pandas", "numpy", "git", "github",
        "linux", "bash", "java", "c++", "c#", "r", "matlab", "power bi", "tableau", "hadoop", "spark",
        "mern", "mevn", "rest api", "graphql", "firebase", "postman", "selenium", "jest", "mocha", "ci/cd"
    ])
    jd_entities = {e.lower() for ent in jd_analysis["entities"].values() for e in ent if e.lower() in valid_skills}
    resume_entities = set([s.lower() for s in resume_analysis["skills"]])
    missing_entities = jd_entities - resume_entities

    if missing_entities:
        suggestions.append({
            "title": "Missing Skills/Technologies",
            "description": "These are mentioned in the job description but not found in your resume.",
            "items": list(missing_entities)
        })

    covered_reqs = set()
    for m in matches:
        for req in jd_analysis["requirements"]:
            if req in m["jd_chunk"]:
                covered_reqs.add(req)
    missing_reqs = set(jd_analysis["requirements"]) - covered_reqs

    if missing_reqs:
        suggestions.append({
            "title": "Missing Explicit Requirements",
            "description": "These explicit JD requirements weren't addressed clearly.",
            "items": list(missing_reqs)
        })
    return suggestions
