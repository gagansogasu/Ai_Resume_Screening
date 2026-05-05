import os
# Silences technical warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Examining the path of torch.classes.*")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import base64
import numpy as np
from resume_processing import extract_text_from_pdf, preprocess_text, rank_resumes

# Configure the page - must be the first Streamlit command
st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for enhanced modern styling
st.markdown("""
<style>
/* Global styles */
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #0e1117;
    color: #ffffff;
}

.main-header {
    font-size: 2.8rem;
    font-weight: 800;
    color: #ffffff;
    text-align: center;
    margin-bottom: 2rem;
    letter-spacing: -0.01em;
}

.sub-header {
    font-size: 1.8rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 1rem;
}

/* Reference Image specific styles */
.content-section {
    margin-top: 2rem;
    padding: 0 1rem;
}

.column-title {
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 1.2rem;
    color: #ffffff;
}

.list-item {
    margin-bottom: 0.8rem;
    font-size: 1rem;
    color: #e0e6ed;
    line-height: 1.5;
}

.feature-item {
    display: flex;
    align-items: center;
    margin-bottom: 0.8rem;
    color: #e0e6ed;
}

.check-icon {
    color: #00c853;
    margin-right: 10px;
    font-weight: bold;
}

/* Cards - White rounded as per image */
.metric-box {
    background: #ffffff;
    border-radius: 10px;
    height: 50px;
    margin-bottom: 2rem;
}

/* Sidebar styling to match image */
section[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid #30363d;
}

section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

.sidebar-about-box {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 8px;
    margin-top: 20px;
    font-size: 0.9rem;
    line-height: 1.6;
    color: #ffffff !important;
}

/* Red button style */
.stButton > button {
    background-color: #ff4b4b !important;
    color: white !important;
    border: none !important;
    padding: 10px 24px !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: opacity 0.2s;
}

.stButton > button:hover {
    opacity: 0.9;
}

/* Progress bar - Red */
.stProgress > div > div > div > div {
    background-color: #ff4b4b !important;
}

/* Input fields */
.stTextArea > div > div > textarea {
    background-color: #161b22;
    color: white;
    border: 1px solid #30363d;
    border-radius: 8px;
}

.stTextArea > div > div > textarea:focus {
    border-color: #ff4b4b;
    box-shadow: 0 0 0 1px #ff4b4b;
}

/* File uploader */
.stFileUploader > div > div {
    background-color: #161b22;
    border: 1px dashed #30363d;
    border-radius: 8px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #30363d;
}

.stTabs [data-baseweb="tab"] {
    color: #8b949e !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 2px solid #ff4b4b !important;
}

/* Tables */
.dataframe {
    background-color: #161b22;
    color: white;
    border: 1px solid #30363d;
}

/* Remove all animations exactly like image */
* { animation: none !important; }

</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if "ranked_resumes" not in st.session_state:
    st.session_state["ranked_resumes"] = None
if "resume_texts" not in st.session_state:
    st.session_state["resume_texts"] = None
if "resume_files" not in st.session_state:
    st.session_state["resume_files"] = None
if "job_description" not in st.session_state:
    st.session_state["job_description"] = ""
if "page" not in st.session_state:
    st.session_state["page"] = "Home"
if "processing_complete" not in st.session_state:
    st.session_state["processing_complete"] = False

# Navigation configurations and callbacks
nav_labels = ["🏠 Home", "📂 Upload & Process", "📊 Results"]
nav_mapping = {
    "🏠 Home": "Home",
    "📂 Upload & Process": "Upload & Process",
    "📊 Results": "Results"
}
reverse_mapping = {v: k for k, v in nav_mapping.items()}

def navigate_to(page_name):
    st.session_state["page"] = page_name
    st.session_state["sidebar_navigation_widget"] = reverse_mapping.get(page_name, "🏠 Home")

# Sidebar Navigation with enhanced modern design
with st.sidebar:
    st.markdown("""
    <div style='padding: 10px 0; border-bottom: 1px solid #30363d; margin-bottom: 20px;'>
        <h3 style='font-size: 1.1rem; color: #ffffff; margin-bottom: 15px;'>📌 Navigation</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Find the index of the current page
    # Source of truth is st.session_state["page"]
    current_page = st.session_state.get("page", "Home")
    
    # Calculate index for radio
    try:
        current_index = nav_labels.index(reverse_mapping.get(current_page, "🏠 Home"))
    except ValueError:
        current_index = 0
    
    # Navigation radio with a unique key
    selected = st.radio(
        "Navigation",
        options=nav_labels,
        label_visibility="collapsed",
        index=current_index,
        key="sidebar_navigation_widget"
    )
    
    # Update page state and trigger rerun if changed by user
    new_page_val = nav_mapping[selected]
    if new_page_val != st.session_state["page"]:
        st.session_state["page"] = new_page_val
        st.rerun()
    
    page = st.session_state["page"]
    
    # Enhanced divider
    st.markdown("<hr style='margin: 20px 0; border: none; height: 1px; background: #30363d;'>", unsafe_allow_html=True)
    
    # Enhanced about section
    st.markdown("<h3 style='font-size: 1.1rem; color: #ffffff; margin-top: 30px; margin-bottom: 15px;'>About</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class='sidebar-about-box'>
        This application uses NLP and AI to match resumes with job descriptions. 
        It analyzes the semantic similarity between the content of resumes and job requirements.
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced footer
    st.markdown("""
    <div style='text-align: center; padding: 20px; margin-top: 30px; border-top: 1px solid #30363d;'>
        <div style='font-size: 12px; color: #8b949e; margin-bottom: 5px;'>Version 2.0.0</div>
        <div style='font-size: 11px; color: #666;'> 2025 AI Resume Screener</div>
        <div style='margin-top: 10px;'>
            <span style='background: #ff4b4b; color: white; padding: 4px 12px; border-radius: 4px; font-size: 11px; font-weight: 700;'>v2.0</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Helper function for feature cards
def feature_card(icon, title, description):
    return f"""
    <div class='card' style='background: #161b22; padding: 24px; margin-bottom: 20px; border: 1px solid #30363d; border-radius: 12px;'>
        <div style='font-size: 1.5rem; color: #ffffff; margin-bottom: 12px;'>{icon}</div>
        <h3 style='margin: 0 0 10px 0; color: #ffffff; font-weight: 700;'>{title}</h3>
        <p style='color: #8b949e; font-size: 0.95rem; line-height: 1.6;'>{description}</p>
    </div>
    """

# Home Page
if page == "Home":
    # Title
    st.markdown("<h1 class='main-header'>AI Resume Screening & Ranking System</h1>", unsafe_allow_html=True)

    # Main content columns
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("<h2 class='column-title'>How It Works</h2>", unsafe_allow_html=True)
        how_it_works = [
            "1. Upload Resume PDFs: Submit multiple candidate resumes in PDF format",
            "2. Enter Job Description: Provide detailed job requirements and qualifications",
            "3. AI Analysis: Our system extracts, processes, and analyzes the content",
            "4. Get Rankings: View candidate rankings with match scores and visualizations"
        ]
        for item in how_it_works:
            st.markdown(f"<div class='list-item'>{item}</div>", unsafe_allow_html=True)
            
        st.markdown("""
        <div style='margin-top: 20px; color: #c0c0c0; font-size: 0.95rem; line-height: 1.6;'>
            The system uses natural language processing and semantic similarity algorithms to match candidate 
            skills and experiences with job requirements.
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("<h2 class='column-title'>Features</h2>", unsafe_allow_html=True)
        features = [
            "Multi-resume processing",
            "Semantic matching",
            "Interactive visualizations",
            "Text extraction & analysis",
            "Candidate comparison"
        ]
        for feat in features:
            st.markdown(f"""
            <div class='feature-item'>
                <span class='check-icon'>✓</span>
                <span>{feat}</span>
            </div>
            """, unsafe_allow_html=True)

    # Getting Started section
    st.markdown("<h2 class='sub-header' style='margin-top: 2rem;'>Getting Started</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #e0e6ed; margin-bottom: 20px;'>Click on \"Upload & Process\" in the sidebar to begin uploading resumes and job descriptions.</p>", unsafe_allow_html=True)
    
    st.button("Go to Upload Section", use_container_width=False, key="cta_button_red", on_click=navigate_to, args=("Upload & Process",))


elif page == "Upload & Process":
    # Header
    st.markdown("<h1 style='color: #ffffff; font-size: 2.5rem; font-weight: 700; margin-bottom: 2rem;'>Upload Resumes & Job Description</h1>", unsafe_allow_html=True)
    
    # Job Description Section
    st.markdown("<label style='color: #ffffff; font-size: 1rem; margin-bottom: 0.5rem; display: block;'>Job Description</label>", unsafe_allow_html=True)
    job_desc = st.text_area(
        "Job Description",
        placeholder="Paste the complete job description here...",
        height=250,
        value=st.session_state.get("job_description", ""),
        label_visibility="collapsed"
    )
    
    st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)
    
    # Upload Section
    st.markdown("<h2 style='color: #ffffff; font-size: 1.8rem; font-weight: 700; margin-bottom: 0.5rem;'>Upload Resumes</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8b949e; font-size: 0.9rem; margin-bottom: 1rem;'>Upload Resumes (PDF format only)</p>", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Upload Resumes",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    # Process Button
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    process_clicked = st.button(
        "Process Resumes", 
        type="primary", 
        disabled=(not uploaded_files or not job_desc.strip()),
        use_container_width=False
    )
    
    # Reset completeness if files are changed or button is clicked
    if process_clicked:
        st.session_state["processing_complete"] = False

    # Process resumes when button is clicked
    if process_clicked and uploaded_files and job_desc.strip():
        # Create a progress container
        progress_container = st.container()
        
        with progress_container:
            st.markdown("<div style='background: #161b22; padding: 1.5rem; border-radius: 8px; border: 1px solid #30363d; margin-top: 2rem;'>", unsafe_allow_html=True)
            st.markdown("<h3 style='color: #ffffff; margin-bottom: 1rem;'>⚙️ Processing Resumes</h3>", unsafe_allow_html=True)
            
            # Initialize progress bar
            progress_bar = st.progress(0, text="Starting...")
            
            # Create a placeholder for file processing status
            file_status = st.empty()
            
            # Initialize lists to store results
            resume_texts = []
            file_names = []
            
            # Process each file
            for i, file in enumerate(uploaded_files):
                try:
                    # Update status
                    progress_text = f"Processing {i+1} of {len(uploaded_files)}: {file.name}"
                    progress_bar.progress((i) / len(uploaded_files), text=progress_text)
                    
                    # Extract text from PDF
                    with st.spinner(f"Extracting text from {file.name}..."):
                        text = extract_text_from_pdf(file)
                    
                    if not text.strip():
                        file_status.warning(f"⚠️ No text found in {file.name}. It may be a scanned PDF.")
                    else:
                        file_status.success(f"✅ Processed: {file.name}")
                    
                    resume_texts.append(text)
                    file_names.append(file.name)
                    
                    # Update progress
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress, text=f"Processed {i+1} of {len(uploaded_files)} resumes")
                    
                except Exception as e:
                    file_status.error(f"❌ Error processing {file.name}: {str(e)}")
                    continue  # Skip the failed file
            
            if resume_texts:
                # Rank the resumes
                with st.spinner("Analyzing and ranking resumes..."):
                    ranked_resumes = rank_resumes(job_desc, resume_texts)
                
                # Save to session state
                st.session_state["ranked_resumes"] = ranked_resumes
                st.session_state["resume_texts"] = resume_texts
                st.session_state["resume_files"] = file_names
                st.session_state["job_description"] = job_desc
                st.session_state["processing_complete"] = True
                
                # Show completion message
                progress_bar.progress(100, "Analysis complete!")
                st.balloons()
            else:
                st.error("❌ No valid resumes were processed. Please check the uploaded files and try again.")
            
            st.markdown("</div>", unsafe_allow_html=True)

    # Show success message and navigate button if processing is complete
    if st.session_state.get("processing_complete"):
        st.markdown(f"""
        <div style='background-color: #1e293b; padding: 1.5rem; border-radius: 8px; margin: 2rem 0; border: 1px solid #30363d;'>
            <div style='display: flex; align-items: center;'>
                <div style='background: #4CAF50; color: white; width: 40px; height: 40px; border-radius: 8px; display: flex; align-items: center; justify-content: center; margin-right: 1.5rem; font-size: 1.2rem;'>✓</div>
                <div>
                    <h3 style='margin: 0 0 0.5rem 0; color: #ffffff; font-weight: 700;'>Analysis Complete</h3>
                    <p style='margin: 0; color: #e0e6ed;'>Successfully processed {len(st.session_state.get("resume_files", []))} resumes. View the results on the Results page.</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.button("📊 View Results", type="primary", use_container_width=True, key="view_results_btn_fixed", on_click=navigate_to, args=("Results",))

 
 
 # Results Page
elif page == "Results":
    # Page header
    st.markdown("""
    <div style='margin-bottom: 2rem;'>
        <div style='display: flex; justify-content: space-between; align-items: center; margin: 1rem 0;'>
            <div>
                <h1 style='color: #ffffff; margin: 0;'>Resume Ranking Results</h1>
                <p style='color: #8b949e; margin: 0.5rem 0 0;'>AI-powered analysis of candidate resumes</p>
            </div>
            <div style='background: #161b22; padding: 0.5rem 1rem; border-radius: 4px; border: 1px solid #30363d;'>
                <span style='color: #ffffff; font-weight: 700;'>
                    {0} {1} analyzed
                </span>
            </div>
        </div>
    </div>
    """.format(
        len(st.session_state.get("resume_files", [])), 
        'resume' if len(st.session_state.get("resume_files", [])) == 1 else 'resumes'
    ), unsafe_allow_html=True)

    if st.session_state.get("ranked_resumes") is None:
        st.warning("Please upload resumes and process them to view results.")
        st.button("← Go to Upload Section", type="primary", on_click=navigate_to, args=("Upload & Process",))
    else:
        ranked_resumes = st.session_state["ranked_resumes"]
        resume_texts = st.session_state["resume_texts"]
        file_names = st.session_state["resume_files"]
        job_desc = st.session_state["job_description"]

        # ---------------------------------------------------------
        # Prepare DataFrames for Charts
        # ---------------------------------------------------------
        # ranked_resumes is a list of dicts now
        df_data = []
        for r in ranked_resumes:
            idx = r['index']
            if 0 <= idx < len(file_names):
                name = file_names[idx]
                df_data.append({
                    'Name': name,
                    'Total Score': r['score'],
                    'Semantic Match': r['semantic_score'],
                    'Keyword Match': r['keyword_score'],
                    'Experience': r['details']['experience_score'],
                    'Education': r['details']['education_score'],
                    'Skills Section': r['details']['skills_score'],
                    'Matched Keywords': len(r['matched_keywords']),
                    'Missing Keywords': len(r['missing_keywords']),
                    'Original Index': idx
                })
        
        if not df_data:
            st.error("No valid data found.")
            st.stop()
            
        df_results = pd.DataFrame(df_data)
        
        # ---------------------------------------------------------
        # Tabs
        # ---------------------------------------------------------
        tab1, tab2, tab3 = st.tabs(["📊 Overview & Rankings", "🔍 Deep Dive Analysis", "📄 Resume Content"])

        with tab1:
            # --- Top Stats Row ---
            col1, col2, col3 = st.columns(3)
            avg_score = df_results['Total Score'].mean()
            top_score = df_results['Total Score'].max()
            top_candidate = df_results.loc[df_results['Total Score'].idxmax()]['Name']
            
            with col1:
                st.metric("Average Match Score", f"{avg_score:.1f}%")
            with col2:
                st.metric("Top Score", f"{top_score:.1f}%")
            with col3:
                st.metric("Top Candidate", top_candidate)

            st.markdown("---")

            # --- Main Bar Chart (Plotly) ---
            st.subheader("🏆 Candidate Rankings")
            fig_bar = px.bar(
                df_results, 
                x='Total Score', 
                y='Name', 
                orientation='h',
                text='Total Score',
                color='Total Score',
                color_continuous_scale=['#440154', '#ff4b4b'],
                labels={'Total Score': 'Match Score (%)'}
            )
            fig_bar.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, height=400, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_bar, use_container_width=True)

            # --- Detailed Data Table ---
            st.subheader("📋 ML-Powered Detailed Scores")
            st.dataframe(
                df_results[['Name', 'Total Score', 'Semantic Match', 'Keyword Match', 'Matched Keywords']],
                column_config={
                    "Total Score": st.column_config.ProgressColumn(
                        "Final ML Score", format="%.1f%%", min_value=0, max_value=100
                    ),
                    "Semantic Match": st.column_config.NumberColumn("BERT Similarity (%)", format="%.1f%%"),
                    "Keyword Match": st.column_config.NumberColumn("Contextual Match (%)", format="%.1f%%"),
                },
                hide_index=True,
                use_container_width=True
            )

        with tab2:
            st.subheader("🔍 Individual ML Analysis")
            
            # AI Summary Card
            selected_name = st.selectbox("Select Candidate to Analyze", df_results['Name'].tolist())
            candidate_data = df_results[df_results['Name'] == selected_name].iloc[0]
            r_data = next(r for r in ranked_resumes if r['index'] == candidate_data['Original Index'])

            st.markdown(f"""
            <div style='background: #1e293b; padding: 20px; border-radius: 12px; border: 1px solid #30363d; margin-bottom: 25px;'>
                <h4 style='color: #ff4b4b; margin-top: 0; margin-bottom: 15px;'>📝 AI Recruiter Summary (3-Bullet)</h4>
                <ul style='color: #e0e6ed; padding-left: 20px; line-height: 1.6;'>
                    {" ".join([f"<li>{bullet}</li>" for bullet in r_data.get('ai_summary', [])])}
                </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style='background: #161b22; padding: 10px 15px; border-radius: 8px; border-left: 4px solid #ff4b4b; margin-bottom: 20px;'>
                <span style='color: #8b949e; font-size: 0.9rem;'>
                    💡 <b>ML Insight:</b> This analysis uses <b>BERT</b> for semantic understanding and <b>BART</b> for abstractive summarization.
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # --- Radar Chart for Component Scores ---
            col_radar, col_keywords = st.columns([1, 1])
            
            with col_radar:
                st.markdown("#### 🎯 Semantic Breakdown")
                # Prepare data for Radar
                categories = ['Semantic', 'Keywords', 'Experience', 'Education', 'Skills Section']
                values = [
                    r_data['semantic_score'],
                    r_data['keyword_score'],
                    r_data['details']['experience_score'],
                    r_data['details']['education_score'],
                    r_data['details']['skills_score']
                ]
                
                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=selected_name,
                    line_color='#ff4b4b'
                ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100], gridcolor="#30363d", tickfont=dict(color="#8b949e")),
                        angularaxis=dict(gridcolor="#30363d", tickfont=dict(color="#8b949e")),
                        bgcolor="rgba(0,0,0,0)"
                    ),
                    showlegend=False,
                    height=350,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=40, r=40, t=20, b=20)
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                
                # Semantic Fit Badge
                fit_vibe = "High" if candidate_data['Semantic Match'] > 75 else "Medium" if candidate_data['Semantic Match'] > 50 else "Low"
                fit_color = "#4CAF50" if fit_vibe == "High" else "#FF9800" if fit_vibe == "Medium" else "#f44336"
                
                st.markdown(f"""
                <div style='text-align: center; margin-top: -20px;'>
                    <span style='background: {fit_color}; color: white; padding: 5px 15px; border-radius: 20px; font-size: 0.8rem; font-weight: 700;'>
                        {fit_vibe} Semantic Alignment
                    </span>
                </div>
                """, unsafe_allow_html=True)

            with col_keywords:
                st.markdown("#### 🔑 Contextual Keywords")
                
                # Matched Keywords (Green pills)
                st.markdown("**✅ Matched Keywords**")
                if r_data['matched_keywords']:
                    html_matched = " ".join([
                        f"<span style='background-color:#1e293b; color:#ffffff; border: 1px solid #30363d; padding:4px 8px; border-radius:4px; font-size:0.85em; margin-right:4px; display:inline-block; margin-bottom:4px; font-weight:600;'>{k}</span>" 
                        for k in r_data['matched_keywords']
                    ])
                    st.markdown(html_matched, unsafe_allow_html=True)
                else:
                    st.info("No direct keyword matches found.")
                
                st.markdown("---")
                
                # Missing Keywords (Red pills)
                st.markdown("**❌ Missing / Recommended Keywords**")
                if r_data['missing_keywords']:
                    html_missing = " ".join([
                        f"<span style='background-color:#0d1117; color:#8b949e; border: 1px solid #30363d; padding:4px 8px; border-radius:4px; font-size:0.85em; margin-right:4px; display:inline-block; margin-bottom:4px;'>{k}</span>" 
                        for k in r_data['missing_keywords'][:15] # Limit to 15 to save space
                    ])
                    st.markdown(html_missing, unsafe_allow_html=True)
                    if len(r_data['missing_keywords']) > 15:
                        st.caption(f"...and {len(r_data['missing_keywords']) - 15} more.")
                else:
                    st.success("Great! No major missing keywords identified from the job description.")

        with tab3:
            st.subheader("📄 Resume Content Viewer")
            
            # Select Candidate (in case they switched tabs)
            # Synchronize with tab2 if possible, otherwise independent
            view_name = st.selectbox("Select Resume to View", df_results['Name'].tolist(), key="view_select")
            
            view_idx = df_results[df_results['Name'] == view_name].iloc[0]['Original Index']
            
            col_text, col_jd = st.columns(2)
            
            with col_text:
                st.markdown("**Resume Text**")
                st.text_area("Resume Content", resume_texts[view_idx], height=500)
            
            with col_jd:
                st.markdown("**Job Description**")
                st.text_area("JD Content", job_desc, height=500)