import streamlit as st
from models import load_models
from utils import extract_text
from analysis import analyze_job_description, analyze_resume, semantic_match, generate_suggestions

st.set_page_config(page_title="Resume scorer and Optimiser(Suggestion based)", layout="wide")

@st.cache_resource
def get_models():
    return load_models()

nlp, model, kw_model = get_models()

def main():
    st.title("Resume scorer and Optimiser (Suggestion based)")
    st.write("Understand your resume’s alignment with a job description using deep NLP techniques.")

    col1, col2 = st.columns(2)
    with col1:
        jd_file = st.file_uploader("Upload Job Description", type=['pdf', 'docx', 'txt'])
    with col2:
        resume_file = st.file_uploader("Upload Resume", type=['pdf', 'docx', 'txt'])

    if st.button("Analyze", type="primary"):
        if not jd_file or not resume_file:
            st.error("Please upload both files.")
            return

        with st.spinner("Analyzing..."):
            jd_text = extract_text(jd_file)
            resume_text = extract_text(resume_file)

            jd_analysis = analyze_job_description(jd_text, nlp, kw_model)
            resume_analysis = analyze_resume(resume_text, nlp)

            match_results = semantic_match(
                jd_analysis["semantic_chunks"],
                resume_analysis["semantic_chunks"],
                resume_analysis["sections"],
                model
            )
            suggestions = generate_suggestions(
                jd_analysis, resume_analysis, match_results["matches"]
            )

            st.subheader(f"Match Score: {match_results['overall_score']:.1f}%")
            st.progress(match_results['overall_score'] / 100)

            st.markdown("## 🛠 Suggestions for Improvement")
            if not suggestions:
                st.success("Your resume is a strong semantic match!")
            else:
                for s in suggestions:
                    st.markdown(f"### {s['title']}")
                    st.write(s['description'])
                    for item in s['items']:
                        st.write(f"- {item}")

if __name__ == '__main__':
    main()