import streamlit as st
from inference import load_components, compute_similarity_score
from preprocessing import extract_text_from_pdf

st.title("Resume Optimiser")

tfidf, le, category_vectors = load_components()
category_options = le.classes_
selected_category = st.selectbox("Select the job category you are targeting:", category_options)
uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])

if uploaded_file and selected_category:
    raw_text = extract_text_from_pdf(uploaded_file)
    relevance_score = compute_similarity_score(raw_text, selected_category, tfidf, le, category_vectors)

    st.subheader("Resume Optimiser")
    st.write(f"{relevance_score:.2f}% relevant to category: *{selected_category}*")

    # suggestions = suggest_keywords_spacy(raw_text)    
    # if suggestions:
    #     st.subheader("Suggested Skills to Add")
    #     st.markdown(", ".join(suggestions))