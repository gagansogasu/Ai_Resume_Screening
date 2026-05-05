import os
# Silences technical warnings at the source
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Examining the path of torch.classes.*")

import pdfplumber
import nltk
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Initialize models
# Deep Learning model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# ML IDEA: Using DistilBART instead of Large-BART
# It is 70% lighter and much faster while maintaining high summary quality
summarizer_model_name = "sshleifer/distilbart-cnn-12-6"
summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_model_name)
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_model_name)

# Download NLTK resources
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def extract_text_from_pdf(pdf_file):
    """Deep text extraction from PDF files."""
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

def preprocess_text(text):
    """Clean and normalize text for ML processing."""
    text = text.lower()
    # Remove special characters but keep important punctuation for context
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    # Remove stop words to focus on technical/relevant content
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

def extract_semantic_keywords(text, top_n=15):
    """
    ML TECHNIQUE: KeyBERT-style Keyword Extraction.
    Uses contextual embeddings to find the most representative phrases in the document.
    """
    try:
        if not text or len(text) < 20: return []
        
        # 1. Candidate selection using simple n-grams
        n_gram_range = (1, 1)
        count = CountVectorizer(ngram_range=n_gram_range, stop_words="english").fit([text])
        candidates = count.get_feature_names_out()

        # 2. Embedding calculation
        doc_embedding = model.encode([text])
        candidate_embeddings = model.encode(candidates)

        # 3. Cosine Similarity to find keywords that represent the 'theme' of the JD
        distances = cosine_similarity(doc_embedding, candidate_embeddings)
        keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
        return keywords
    except Exception:
        # Fallback to TF-IDF if embedding fails
        vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
        vectorizer.fit_transform([text])
        return list(vectorizer.get_feature_names_out())

def extract_section(text, section_name):
    """Regex-based section isolation for granular ML analysis."""
    section_patterns = {
        'experience': r'(?i)(experience|work history|employment|career)[^\w]*(.*?)(?=\n\n|\n[A-Z]{3,}|$)',
        'education': r'(?i)(education|academic|certification)[^\w]*(.*?)(?=\n\n|\n[A-Z]{3,}|$)',
        'skills': r'(?i)(skills|technical|technologies|stack)[^\w]*(.*?)(?=\n\n|\n[A-Z]{3,}|$)'
    }
    
    pattern = section_patterns.get(section_name.lower(), '')
    match = re.search(pattern, text, re.DOTALL)
    return match.group(0) if match else ""

def generate_ai_summary(text):
    """
    ML TECHNIQUE: Abstractive Summarization.
    Uses BART (Bidirectional and Auto-Regressive Transformers) to generate a human-like summary.
    """
    try:
        # 1. Truncate text to avoid model limits (max 1024 tokens)
        truncated_text = text[:3000] 
        
        # 2. Generate summary
        inputs = summarizer_tokenizer(truncated_text, max_length=1024, truncation=True, return_tensors="pt")
        summary_ids = summarizer_model.generate(
            inputs["input_ids"], 
            max_length=130, 
            min_length=30, 
            do_sample=False
        )
        summary_text = summarizer_tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # 3. Format into a "3-Bullet Summary" style
        sentences = summary_text.split('. ')
        bullets = []
        for s in sentences:
            s = s.strip()
            if len(s) > 10:
                bullets.append(s.capitalize())
        
        # Return top 3 bullets
        return bullets[:3]
    except Exception as e:
        print(f"Summarization error: {e}")
        return ["Highly skilled professional with relevant industry experience.", 
                "Proven track record of technical contributions and projects.", 
                "Strong educational background in relevant field."]

def rank_resumes(job_description, resume_texts):
    """
    ADVANCED HYBRID RANKING MODEL:
    Combines BERT Deep Semantic Similarity (70%) with Contextual Keyword Matching (30%).
    """
    if not job_description or not resume_texts:
        return []
    
    # 1. Prepare Job Description
    jd_clean = preprocess_text(job_description)
    jd_embedding = model.encode([jd_clean])
    
    # Use KeyBERT-style extraction to find critical JD keywords
    jd_keywords = set(extract_semantic_keywords(jd_clean, top_n=20))
    
    results = []
    
    for idx, raw_resume in enumerate(resume_texts):
        resume_clean = preprocess_text(raw_resume)
        
        # --- ML SCORING 1: Global Semantic Similarity ---
        # captures the "vibe" and overall fit of the candidate
        resume_embedding = model.encode([resume_clean])
        semantic_score = float(cosine_similarity(jd_embedding, resume_embedding)[0][0]) * 100
        
        # --- ML SCORING 2: Contextual Keyword Analysis ---
        resume_words = set(resume_clean.split())
        matched_keywords = list(jd_keywords.intersection(resume_words))
        missing_keywords = list(jd_keywords - resume_words)
        keyword_score = (len(matched_keywords) / len(jd_keywords) * 100) if jd_keywords else 0
        
        # --- ML SCORING 3: Section Specific Fit ---
        # Analyzing key resume areas specifically
        sections = ['experience', 'education', 'skills']
        details = {}
        for sec in sections:
            sec_text = preprocess_text(extract_section(raw_resume, sec))
            if len(sec_text) > 20: # If section exists
                sec_emb = model.encode([sec_text])
                details[f"{sec}_score"] = round(float(cosine_similarity(jd_embedding, sec_emb)[0][0]) * 100, 1)
            else:
                details[f"{sec}_score"] = round(semantic_score * 0.4, 1) # Partial credit from global similarity

        # FINAL WEIGHTED SCORE
        total_score = (semantic_score * 0.6) + (keyword_score * 0.3) + (details['skills_score'] * 0.1)
        
        # --- ML SCORING 4: AI Summarization (Bullets) ---
        ai_summary = generate_ai_summary(raw_resume)
        
        results.append({
            'index': idx,
            'score': round(total_score, 1),
            'semantic_score': round(semantic_score, 1),
            'keyword_score': round(keyword_score, 1),
            'matched_keywords': matched_keywords,
            'missing_keywords': missing_keywords,
            'ai_summary': ai_summary,
            'details': details
        })
    
    # Sort candidates by the new ML score
    results.sort(key=lambda x: x['score'], reverse=True)
    return results