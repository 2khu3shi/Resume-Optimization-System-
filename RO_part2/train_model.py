import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import clean_resume

# Load and clean dataset
df = pd.read_csv("C:\\Users\\ACER\\Desktop\\resume_dataset_job_skills.csv", encoding="utf-8")
df['cleaned_resume'] = df['Resume'].apply(clean_resume)

# Encode labels
le = LabelEncoder()
df['Category_encoded'] = le.fit_transform(df['Category'])

# Vectorize text
tfidf = TfidfVectorizer(max_features=1500, stop_words='english')
X = tfidf.fit_transform(df['cleaned_resume'])
y = df['Category_encoded']

# Save models
os.makedirs("models", exist_ok=True)
joblib.dump(tfidf, "models/tfidf_vectorizer.joblib")
joblib.dump(le, "models/label_encoder.joblib")

category_vectors = {cat: X[y == cat] for cat in np.unique(y)}
joblib.dump(category_vectors, "models/category_vectors.joblib")

print("Training complete and models saved.")