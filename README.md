<<<<<<< HEAD
# 🚀 AI Resume Screening & Ranking System

An AI-powered web application that automates resume screening and ranking based on job descriptions using Natural Language Processing (NLP) and Machine Learning. It helps recruiters quickly shortlist the best candidates.



## 📌 Table of Contents
- [🔥 Features](#-features)
- [💻 Tech Stack](#-tech-stack)
- [⚡ Installation & Setup](#-installation--setup)
- [🚀 Usage](#-usage)
- [🧠 How It Works](#-how-it-works)
- [📈 Future Enhancements](#-future-enhancements)
- [💡 Contributing](#-contributing)

---

## 🔥 Features
✅ Upload multiple resumes (PDF format)  
✅ AI-powered semantic matching  
✅ Job description-based ranking  
✅ Keyword extraction from resumes  
✅ Interactive visualizations for candidate comparison  
✅ Downloadable CSV report  
✅ User-friendly web interface using Streamlit  

---

## 💻 Tech Stack
| **Technology** | **Purpose** |
|--------------|------------|
| **Python** | Backend & Machine Learning |
| **Streamlit** | Web Framework |
| **NLTK, Spacy** | NLP Processing |
| **Scikit-learn** | Model Training |
| **Pandas & Matplotlib** | Data Analysis & Visualization |

---


## ⚡ Installation & Setup  

### 🏢 Step 1: Clone the Repository  
```bash
git clone https://github.com/gagansogasu/Ai_Resume_Screening.git
cd Ai_Resume_Screening
```

### 🛠️ Step 2: Create a Virtual Environment  
For **Windows**:  
```bash
python -m venv venv
venv\Scripts\activate
```
For **Mac/Linux**:  
```bash
python3 -m venv venv
source venv/bin/activate
```

### 🛋️ Step 3: Install Dependencies  
```bash
pip install -r requirements.txt
```

### 🚀 Step 4: Run the Application  
```bash
streamlit run app.py
```

---

## 🚀 Usage  
1. **Upload resumes** (in **PDF** format).  
2. **Enter a job description** to match candidates.  
3. **Click on "Analyze"** to start screening.  
4. **View ranked candidates** based on **best fit**.  
5. **Download the CSV report** for further analysis.  

---

## 🧠 How It Works  
1. **Resume Parsing**: Extracts text from resumes using `PyPDF2`.  
2. **Data Preprocessing**: Tokenization, stopword removal, and lemmatization with `NLTK` and `Spacy`.  
3. **Feature Engineering**: Converts text into numerical vectors using `TF-IDF`.  
4. **Candidate Ranking**: Uses `Scikit-learn` models to rank resumes.  
5. **Visualization**: Uses `Matplotlib` and `Seaborn` to present analytics.  

---

## 📈 Future Enhancements  
🔹 Support for DOCX and image-based resumes (OCR integration)  
🔹 Use BERT/GPT for advanced resume matching  
🔹 Integration with LinkedIn and job portals  
🔹 Multi-language support  
🔹 More AI-powered insights & recommendations  

---

## 💡 Contributing  
We welcome contributions! Follow these steps:  
1. Fork the repository.  
2. Create a feature branch:  
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:  
   ```bash
   git commit -m "Added new feature"
   ```
4. Push the branch:  
   ```bash
   git push origin feature-name
   ```
5. Open a **Pull Request (PR)**.  

---

=======
# Ai-resume-screening

"An AI-powered resume screening and ranking system using NLP &amp; Machine Learning."

# 🚀 AI Resume Screening & Ranking System

An AI-powered web app that automates resume screening and ranking based on job descriptions using NLP & Machine Learning.

## 🔥 Features

✅ Upload multiple resumes (PDF)  
✅ AI-powered **semantic matching**  
✅ Job description-based **ranking**  
✅ **Interactive visualizations** for candidate comparison  
✅ **Downloadable CSV report**

## 💻 Tech Stack

| Technology              | Purpose                       |
| ----------------------- | ----------------------------- |
| **Python**              | Backend & ML                  |
| **Streamlit**           | Web Framework                 |
| **NLTK, Spacy**         | NLP Processing                |
| **Scikit-learn**        | Model Training                |
| **Pandas & Matplotlib** | Data Analysis & Visualization |

## ⚡ Installation & Setup

Clone this repository:

```sh
git clone https://github.com/your-username/ai-resume-screening.git
cd ai-resume-screening

```
>>>>>>> 40b7253 (Added requirements.txt)
