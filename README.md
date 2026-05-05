# AI Resume Screening & Ranking System 📄🤖

A professional, state-of-the-art resume screening application that leverages **Deep Learning (BERT)** and **Contextual NLP** to analyze, rank, and visualize candidate resumes against job descriptions.

## 🌟 Key Features

*   **Deep Learning Analysis**: Uses **BERT (Bidirectional Encoder Representations from Transformers)** for semantic understanding of candidate profiles beyond simple keyword matching.
*   **Contextual KeyBERT Extraction**: Automatically identifies critical skills and requirements from job descriptions using contextual embeddings.
*   **Multi-PDF Processing**: Extract and analyze multiple candidate resumes simultaneously.
*   **Interactive Visualizations**:
    *   **Candidate Ranking Chart**: Horizontal bar charts showing overall match scores.
    *   **Semantic Radar Maps**: Visual breakdown of experience, education, and skill alignment.
    *   **Keyword Analysis**: Side-by-side comparison of matched vs. missing critical keywords.
*   **Advanced Hybrid Scoring**: A weighted algorithmic approach (60% Semantic, 30% Keyword, 10% Section-specific fit).

## 🛠️ Technology Stack

*   **Frontend**: [Streamlit](https://streamlit.io/) (Highly customized with Vanilla CSS)
*   **ML Engine**: [Sentence-Transformers](https://www.sbert.net/) (all-MiniLM-L6-v2)
*   **NLP Tools**: NLTK, Scikit-learn
*   **PDF Extraction**: PyPDF2 / pdfplumber
*   **Charts**: Plotly Express, Plotly Graph Objects

## 📁 Project Structure

```text
resume_screening_system/
├── app.py               # Main Streamlit application & UI logic
├── resume_processing.py  # ML/NLP engine (BERT embeddings, Ranking logic)
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## 🚀 Getting Started

### 1. Prerequisites
*   Python 3.11+
*   pip

### 2. Installation

1.  **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    venv\Scripts\activate     # Windows
    source venv/bin/activate  # Linux/Mac
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Usage

1.  **Start the application**:
    ```bash
    streamlit run app.py
    ```
2.  **Open your browser** and navigate to `http://localhost:8501`.
3.  **Navigate to "Upload & Process"**: Paste your JD and upload multiple PDF resumes.
4.  **Analyze Results**: Switch to the "Results" tab to see AI-driven rankings.

## 🧠 ML Methodology

The system uses a **Hybrid Contextual Scoring** model:
1.  **Global Semantic Fit (60%)**: Measures the "thematic" alignment using cosine similarity of BERT embeddings.
2.  **Contextual Keyword Matching (30%)**: Uses KeyBERT-style extraction to find high-value technical terms.
3.  **Section Targeting (10%)**: Specifically analyzes isolated "Skills" sections for high-density requirement matching.

## 🎨 Design Philosophy
The UI is built with a focus on "Rich Aesthetics" as per modern dashboard standards:
*   **Color Palette**: Deep Charcoal (#0e1117) with vibrant Red accents (#ff4b4b).
*   **Experience**: Zero-animation philosophy in high-data areas for performance, with subtle glassmorphism in containers.

## 📝 License
This project is licensed under the MIT License.