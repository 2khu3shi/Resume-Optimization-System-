# Resume Optimization System

This repository contains two related resume-analysis apps:

- `RO_part1`: a semantic resume scorer and suggestion engine using spaCy, Sentence Transformers, and KeyBERT.
- `RO_part2`: a category-based resume relevance scorer using TF-IDF and saved sklearn artifacts.

## Project Structure

```text
Resume_Optimiser-main/
├── RO_part1/
│   ├── analysis.py
│   ├── app.py
│   ├── models.py
│   ├── utils.py
│   └── sentence_model/
│       ├── config.json
│       ├── config_sentence_transformers.json
│       ├── model.safetensors
│       ├── modules.json
│       ├── README.md
│       ├── sentence_bert_config.json
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       ├── vocab.txt
│       └── 1_Pooling/
│           └── config.json
├── RO_part2/
│   ├── app.py
│   ├── inference.py
│   ├── preprocessing.py
│   ├── train_model.py
│   └── models/
│       ├── category_vectors.joblib
│       ├── label_encoder.joblib
│       └── tfidf_vectorizer.joblib
└── README.md
```

## Requirements

Use Python 3.10+ and install these packages:

```bash
pip install streamlit spacy sentence-transformers keybert scikit-learn pandas numpy joblib pdfminer.six docx2txt torch
python -m spacy download en_core_web_lg
```

## How To Run

### 1. Resume scorer and optimiser

```bash
cd RO_part1
streamlit run app.py
```

What it does:

- Upload a job description file and a resume file.
- Extract text from PDF, DOCX, or TXT files.
- Compare semantic similarity between the job description and resume.
- Show a match score and improvement suggestions.

### 2. Resume category relevance checker

```bash
cd RO_part2
streamlit run app.py
```

What it does:

- Select a target job category.
- Upload a resume in PDF format.
- Score how relevant the resume is for the selected category using the saved TF-IDF model.

## Training The Part 2 Model

If you want to regenerate the sklearn artifacts in `RO_part2/models/`, run:

```bash
cd RO_part2
python train_model.py
```

Important:

- `train_model.py` currently points to a local dataset path: `C:\Users\ACER\Desktop\resume_dataset_job_skills.csv`.
- Update that path before training on another machine.

## Notes

- The first app downloads or loads `en_core_web_lg` for spaCy and saves a Sentence Transformer model into `RO_part1/sentence_model/` if needed.
- The apps expect the bundled model files in `RO_part1/sentence_model/` and `RO_part2/models/` to remain in place.