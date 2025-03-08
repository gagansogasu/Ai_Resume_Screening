import pdfplumber
import nltk
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# Download NLTK resources (run once)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
model = SentenceTransformer('all-MiniLM-L6-v2')  # Pre-trained model for embeddings

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        text = f"Error extracting text: {str(e)}"
    return text.strip()

# Function to clean and preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resume_texts):
    job_desc_clean = preprocess_text(job_description)
    resume_texts_clean = [preprocess_text(text) for text in resume_texts]

    # Convert text into numerical vectors using Sentence Transformers
    job_embedding = model.encode([job_desc_clean])
    resume_embeddings = model.encode(resume_texts_clean)

    # Compute similarity scores
    scores = cosine_similarity(job_embedding, resume_embeddings)[0]

    # Create list of (index, score) tuples and sort by score descending
    ranked_indices = [(i, scores[i]) for i in range(len(scores))]
    ranked_indices.sort(key=lambda x: x[1], reverse=True)

    return ranked_indices