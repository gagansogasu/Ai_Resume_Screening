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
def extract_keywords(text, top_n=20):
    """Extract top N keywords from text using TF-IDF weights."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    
    # Initialize and fit TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform([text])
    
    # Get feature names and their scores
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
    
    # Return top N keywords
    return feature_array[tfidf_sorting][:top_n]

def calculate_keyword_match_score(job_keywords, resume_text, weight=0.4):
    """Calculate keyword matching score between job keywords and resume text."""
    if not job_keywords.size or not resume_text:
        return 0.0
    
    # Count how many job keywords appear in the resume
    matches = sum(1 for keyword in job_keywords if keyword.lower() in resume_text.lower())
    return (matches / len(job_keywords)) * 100 * weight

def calculate_section_scores(job_desc, resume_text, sections):
    """Calculate scores for different sections of the resume."""
    section_scores = {}
    
    for section in sections:
        # Simple section matching - in a real app, you'd want more sophisticated parsing
        section_text = extract_section(resume_text, section)
        if section_text:
            # Calculate similarity for this section
            section_embedding = model.encode([section_text])
            job_embedding = model.encode([job_desc])
            similarity = cosine_similarity(job_embedding, section_embedding)[0][0]
            section_scores[section] = similarity * 100  # Convert to percentage
        else:
            section_scores[section] = 0.0
    
    return section_scores

def extract_section(text, section_name):
    """Extract a specific section from resume text."""
    # This is a simple implementation - you might want to use more sophisticated parsing
    section_patterns = {
        'experience': r'(?i)(experience|work history|employment)[^\w]*(.*?)(?=\n\n|$)',
        'education': r'(?i)(education|academic background)[^\w]*(.*?)(?=\n\n|$)',
        'skills': r'(?i)(skills|technical skills|key skills)[^\w]*(.*?)(?=\n\n|$)'
    }
    
    pattern = section_patterns.get(section_name.lower(), '')
    if not pattern:
        return ""
    
    import re
    match = re.search(pattern, text, re.DOTALL)
    return match.group(0) if match else ""

def rank_resumes(job_description, resume_texts):
    """
    Enhanced resume ranking algorithm that combines multiple techniques:
    1. Semantic similarity using sentence transformers
    2. Keyword matching for important terms
    3. Section-based scoring for experience, education, and skills
    """
    if not job_description or not resume_texts:
        return []
    
    # Preprocess texts
    job_desc_clean = preprocess_text(job_description)
    resume_texts_clean = [preprocess_text(text) for text in resume_texts]
    
    # Extract job keywords for matching
    job_keywords = extract_keywords(job_desc_clean)
    
    # Define important sections to analyze
    sections = ['experience', 'education', 'skills']
    
    # Calculate scores for each resume
    ranked_resumes = []
    
    for idx, (resume_text, clean_text) in enumerate(zip(resume_texts, resume_texts_clean)):
        # 1. Semantic similarity (40% weight)
        resume_embedding = model.encode([clean_text])
        job_embedding = model.encode([job_desc_clean])
        semantic_score = cosine_similarity(job_embedding, resume_embedding)[0][0] * 100 * 0.4
        
        # 2. Keyword matching (30% weight)
        keyword_score = calculate_keyword_match_score(job_keywords, resume_text, weight=0.3)
        
        # 3. Section-based scoring (30% weight)
        section_scores = calculate_section_scores(job_desc_clean, resume_text, sections)
        section_avg = sum(section_scores.values()) / len(section_scores) * 0.3 if section_scores else 0
        
        # Calculate total score (normalized to 0-100)
        total_score = semantic_score + keyword_score + section_avg
        
        # Store index and score
        ranked_resumes.append((idx, total_score))
    
    # Sort by score in descending order
    ranked_resumes.sort(key=lambda x: x[1], reverse=True)
    
    # Convert to list of (score, index) tuples for compatibility
    return [(score, idx) for idx, score in ranked_resumes]