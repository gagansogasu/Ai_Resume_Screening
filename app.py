import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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

# Apply custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .highlight {
        background-color: #e3f2fd;
        padding: 5px;
        border-radius: 5px;
    }
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

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h2 style='color: #1E88E5; margin-bottom: 5px;'>AI Resume Screener</h2>
        <p style='color: #666; font-size: 14px;'>Powered by NLP & Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation with icons
    nav_options = {
        "🏠 Home": "Home",
        "📤 Upload & Process": "Upload & Process",
        "📊 Results": "Results"
    }
    
    # Create navigation with custom styling
    selected = st.radio(
        "",
        options=list(nav_options.keys()),
        label_visibility="collapsed"
    )
    
    # Set the page based on selection
    page = nav_options[selected]
    
    # Divider with custom style
    st.markdown("<hr style='margin: 20px 0; border: 0.5px solid #e0e0e0;'>", unsafe_allow_html=True)
    
    # About section with improved styling
    with st.expander("ℹ️  About", expanded=True):
        st.markdown("""
        <div style='font-size: 14px; color: #444; line-height: 1.6;'>
            <p>This application uses advanced NLP and AI to match resumes with job descriptions, helping recruiters identify the most qualified candidates efficiently.</p>
            <p style='margin-top: 10px;'><b>Key Features:</b></p>
            <ul style='margin-top: 0; padding-left: 20px;'>
                <li>Automated resume parsing</li>
                <li>Semantic analysis</li>
                <li>Smart ranking</li>
                <li>Detailed insights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Add version and credits
    st.markdown("""
    <div style='position: fixed; bottom: 20px; left: 20px; right: 20px; font-size: 12px; color: #888; text-align: center;'>
        <p>v1.0.0</p>
        <p> 2025 AI Resume Screener</p>
    </div>
    """, unsafe_allow_html=True)

# Helper function for feature cards
def feature_card(icon, title, description):
    return f"""
    <div class='card' style='padding: 20px; margin-bottom: 20px;'>
        <div style='font-size: 2rem; color: #1E88E5; margin-bottom: 10px;'>{icon}</div>
        <h3 style='margin: 10px 0; color: #2c3e50;'>{title}</h3>
        <p style='color: #666; font-size: 0.95rem; line-height: 1.5;'>{description}</p>
    </div>
    """

# Home Page
if page == "Home":
    # Hero Section
    st.markdown("""
    <div style='text-align: center; padding: 3rem 1rem; background: linear-gradient(135deg, #f5f9ff 0%, #e3f2fd 100%); border-radius: 12px; margin-bottom: 2rem;'>
        <h1 style='font-size: 2.8rem; color: #0D47A1; margin-bottom: 1rem;'>AI-Powered Resume Screening</h1>
        <p style='font-size: 1.2rem; color: #455a64; max-width: 700px; margin: 0 auto 2rem;'>
            Streamline your hiring process with intelligent resume analysis and candidate matching
        </p>
        <a href='#get-started' style='background: linear-gradient(90deg, #1E88E5, #0D47A1); color: white; padding: 0.8rem 2rem; border-radius: 8px; text-decoration: none; font-weight: 500; display: inline-block; margin-top: 1rem;'>
            Get Started Now
        </a>
    </div>
    """, unsafe_allow_html=True)

    # How It Works Section
    st.markdown("<h2 style='text-align: center; margin: 2rem 0; color: #2c3e50;'>How It Works</h2>", unsafe_allow_html=True)
    
    steps = [
        ("", "Upload Resumes", "Easily upload multiple resumes in PDF format"),
        ("", "Add Job Description", "Paste the job description and requirements"),
        ("", "AI Analysis", "Our system processes and analyzes the content"),
        ("", "Get Results", "View ranked candidates with detailed insights")
    ]
    
    cols = st.columns(4)
    for i, (icon, title, desc) in enumerate(steps):
        with cols[i]:
            st.markdown(f"""
            <div style='text-align: center; padding: 1rem;'>
                <div style='font-size: 2rem; margin-bottom: 0.5rem;'>{icon}</div>
                <h3 style='margin: 0.5rem 0; color: #2c3e50; font-size: 1.1rem;'>{title}</h3>
                <p style='color: #666; font-size: 0.9rem;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Features Section
    st.markdown("<div id='features' style='margin: 4rem 0 2rem;'><h2 style='text-align: center; color: #2c3e50;'>Key Features</h2></div>", unsafe_allow_html=True)
    
    # Feature cards in a 2x2 grid
    features = [
        ("", "Lightning Fast Processing", "Process hundreds of resumes in seconds with our optimized algorithms"),
        ("", "Smart Matching", "Advanced NLP to match candidate skills with job requirements"),
        ("", "Data-Driven Insights", "Comprehensive analytics and visualizations for better decision making"),
        ("", "Secure & Private", "Your data stays secure with enterprise-grade encryption")
    ]
    
    # Create two rows of two columns each
    for i in range(0, len(features), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(features):
                with cols[j]:
                    st.markdown(feature_card(*features[i + j]), unsafe_allow_html=True)
    
    # Call to Action
    st.markdown("""
    <div id='get-started' style='background: #f8f9fa; padding: 3rem; border-radius: 12px; text-align: center; margin-top: 3rem;'>
        <h2 style='color: #2c3e50; margin-bottom: 1rem;'>Ready to Transform Your Hiring Process?</h2>
        <p style='color: #666; margin-bottom: 2rem; max-width: 600px; margin-left: auto; margin-right: auto;'>
            Join thousands of recruiters who trust our AI-powered resume screening solution to find the best candidates faster.
        </p>
    """, unsafe_allow_html=True)
    
    if st.button("Start Screening Now", type="primary", use_container_width=True, key="cta_button"):
        st.session_state["page"] = "Upload & Process"
        st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Upload & Process":
    # Page header with breadcrumb
    st.markdown("""
    <div style='margin-bottom: 2rem;'>
        <a href='#' style='color: #1E88E5; text-decoration: none;' onclick='window.history.back(); return false;'>← Back to Home</a>
        <h1 style='color: #2c3e50; margin: 1rem 0;'>Upload Resumes & Job Description</h1>
        <p style='color: #666;'>Upload candidate resumes and paste the job description to start the screening process.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for job description and upload sections
    col1, col2 = st.columns([3, 2], gap="large")
    
    with col1:
        # Job description section with improved styling
        with st.container():
            st.markdown("<div class='card' style='padding: 1.5rem;'>", unsafe_allow_html=True)
            st.markdown("<h3 style='color: #2c3e50; margin-bottom: 1rem;'>📋 Job Description</h3>", unsafe_allow_html=True)
            
            job_desc = st.text_area(
                "",
                placeholder="Paste the complete job description here including required skills, qualifications, and experience...",
                height=300,
                value=st.session_state.get("job_description", ""),
                label_visibility="collapsed"
            )
            
            st.markdown("<div style='font-size: 0.85rem; color: #666; margin-top: 0.5rem;'>Tip: Be as detailed as possible for better matching results.</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # File uploader with improved styling
        with st.container():
            st.markdown("<div class='card' style='padding: 1.5rem; height: 100%;'>", unsafe_allow_html=True)
            st.markdown("<h3 style='color: #2c3e50; margin-bottom: 1rem;'>📤 Upload Resumes</h3>", unsafe_allow_html=True)
            
            # Custom file uploader with drag and drop
            uploaded_files = st.file_uploader(
                "",
                type=["pdf"],
                accept_multiple_files=True,
                help="You can upload multiple PDF files at once",
                label_visibility="collapsed"
            )
            
            # Show upload status
            if uploaded_files:
                st.markdown("<div style='margin-top: 1.5rem;'>", unsafe_allow_html=True)
                st.markdown(f"<div style='color: #4CAF50; font-weight: 500; margin-bottom: 1rem;'>✅ {len(uploaded_files)} file(s) selected</div>", unsafe_allow_html=True)
                
                # Show file previews
                st.markdown("<div style='max-height: 200px; overflow-y: auto; border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px;'>", unsafe_allow_html=True)
                for file in uploaded_files:
                    file_size = f"{file.size / 1024:.1f} KB" if file.size < 1024 * 1024 else f"{file.size / (1024 * 1024):.1f} MB"
                    st.markdown(f"""
                    <div style='display: flex; align-items: center; padding: 8px 0; border-bottom: 1px solid #f0f0f0;'>
                        <div style='margin-right: 12px;'>📄</div>
                        <div style='flex: 1;'>
                            <div style='font-weight: 500; font-size: 0.9rem;'>{file.name}</div>
                            <div style='font-size: 0.8rem; color: #666;'>{file_size}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)  # End of file preview
                st.markdown("</div>", unsafe_allow_html=True)  # End of upload status
            else:
                st.markdown("""
                <div style='text-align: center; padding: 2rem; border: 2px dashed #e0e0e0; border-radius: 8px; margin: 1rem 0; background-color: #fafafa;'>
                    <div style='font-size: 2rem; margin-bottom: 1rem;'>📄</div>
                    <h4 style='margin: 0.5rem 0; color: #2c3e50;'>Drag & drop PDFs here</h4>
                    <p style='color: #666; font-size: 0.9rem; margin: 0;'>or click to browse files</p>
                    <p style='color: #999; font-size: 0.8rem; margin: 0.5rem 0 0;'>Supports multiple PDF files</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)  # End of card

    # Process button with improved styling
    st.markdown("<div style='margin: 2rem 0;'>", unsafe_allow_html=True)
    
    # Create two columns for the button and status
    btn_col, status_col = st.columns([1, 3])
    
    with btn_col:
        process_clicked = st.button(
            "🚀 Process Resumes", 
            type="primary", 
            disabled=(not uploaded_files or not job_desc),
            use_container_width=True,
            help="Analyze and rank the uploaded resumes"
        )
    
    # Add status indicator
    with status_col:
        if not uploaded_files and not job_desc:
            st.markdown("<div style='padding: 1rem; color: #666;'>Please upload resumes and enter a job description</div>", unsafe_allow_html=True)
        elif not uploaded_files:
            st.markdown("<div style='padding: 1rem; color: #F44336;'>⚠️ Please upload at least one resume file</div>", unsafe_allow_html=True)
        elif not job_desc.strip():
            st.markdown("<div style='padding: 1rem; color: #F44336;'>⚠️ Job description cannot be empty</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Process resumes when button is clicked
    if process_clicked and uploaded_files and job_desc.strip():
        # Create a progress container
        progress_container = st.container()
        
        with progress_container:
            st.markdown("<div class='card' style='padding: 1.5rem;'>", unsafe_allow_html=True)
            st.markdown("<h3 style='color: #2c3e50; margin-bottom: 1rem;'>⚙️ Processing Resumes</h3>", unsafe_allow_html=True)
            
            # Initialize progress bar
            progress_bar = st.progress(0, text="Starting...")
            status_text = st.empty()
            
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
                
                # Show completion message
                progress_bar.progress(100, "Analysis complete!")
                st.balloons()
                
                # Success message with animation
                st.markdown("""
                <div style='background-color: #E8F5E9; padding: 1.5rem; border-radius: 8px; margin: 1rem 0; border-left: 5px solid #4CAF50;'>
                    <div style='display: flex; align-items: center;'>
                        <span style='font-size: 1.5rem; margin-right: 1rem;'>🎉</span>
                        <div>
                            <h3 style='margin: 0 0 0.5rem 0; color: #2E7D32;'>Analysis Complete!</h3>
                            <p style='margin: 0; color: #1B5E20;'>Successfully processed {0} resumes. View the results on the Results page.</p>
                        </div>
                    </div>
                </div>
                """.format(len(resume_texts)), unsafe_allow_html=True)
                
                # Add a nice button to view results
                if st.button("📊 View Results", type="secondary", use_container_width=True, key="view_results_btn"):
                    # Ensure page state is properly set and trigger navigation
                    st.session_state.page = "Results"
                    st.experimental_rerun()
                    
            else:
                st.error("❌ No valid resumes were processed. Please check the uploaded files and try again.")
                
            st.markdown("</div>", unsafe_allow_html=True)  # Close card


# Results Page
elif page == "Results":
    # Page header with breadcrumb and stats
    st.markdown("""
    <div style='margin-bottom: 2rem;'>
        <a href='#' style='color: #1E88E5; text-decoration: none;' onclick='window.history.back(); return false;'>← Back to Upload</a>
        <div style='display: flex; justify-content: space-between; align-items: center; margin: 1rem 0;'>
            <div>
                <h1 style='color: #2c3e50; margin: 0;'>Resume Ranking Results</h1>
                <p style='color: #666; margin: 0.5rem 0 0;'>AI-powered analysis of candidate resumes</p>
            </div>
            <div style='background: #E3F2FD; padding: 0.5rem 1rem; border-radius: 20px;'>
                <span style='color: #0D47A1; font-weight: 500;'>
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
        st.warning("""
        <div style='background-color: #FFF3E0; border-left: 5px solid #FFA000; padding: 1rem; border-radius: 4px;'>
            <div style='display: flex; align-items: center;'>
                <span style='font-size: 1.5rem; margin-right: 1rem;'>🔍</span>
                <div>
                    <h3 style='margin: 0 0 0.5rem 0; color: #E65100;'>No Results Found</h3>
                    <p style='margin: 0; color: #E65100;'>Please upload resumes and process them to view results.</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("← Go to Upload Section", type="primary", key="go_to_upload_btn"):
            st.session_state.page = "Upload & Process"
            st.experimental_rerun()
    else:
        ranked_resumes = st.session_state["ranked_resumes"]
        resume_texts = st.session_state["resume_texts"]
        file_names = st.session_state["resume_files"]
        
        # Validate that all indices in ranked_resumes are within bounds
        valid_ranked_resumes = []
        for score, idx in ranked_resumes:
            if 0 <= idx < len(file_names):
                valid_ranked_resumes.append((score, idx))
            else:
                st.warning(f"Warning: Skipping invalid resume index {idx} (out of bounds)")
        
        if not valid_ranked_resumes:
            st.error("No valid resume data to display. Please check your input files and try again.")
            st.stop()
            
        ranked_resumes = valid_ranked_resumes
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "👥 Candidate Details", "📄 Resume Content"])
        job_desc = st.session_state["job_description"]
        
        with tab1:
            st.markdown("""
            <div style='margin-bottom: 2rem;'>
                <h2 style='color: #2c3e50; margin-bottom: 1rem;'>Candidate Rankings</h2>
                <p style='color: #666;'>Candidates ranked by their match score with the job description.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create a row for each candidate with a progress bar
            for idx, (score, text_idx) in enumerate(ranked_resumes):
                # Convert index to integer and ensure it's within bounds
                try:
                    text_idx_int = int(round(float(text_idx)))  # Convert numpy.float32 to Python float then to int
                    if not (0 <= text_idx_int < len(file_names)):
                        st.warning(f"Skipping out-of-bounds resume index {text_idx}")
                        continue
                except (ValueError, TypeError) as e:
                    st.warning(f"Skipping invalid resume index {text_idx}: {str(e)}")
                    continue
                    
                # Calculate color based on score (green to red gradient)
                color_r = int(max(0, 255 * (1 - score / 100)))
                color_g = int(max(0, 255 * (score / 100)))
                color_hex = f"#{color_r:02x}{color_g:02x}60"
                
                with st.container():
                    st.markdown(f"""
                    <div class='card' style='padding: 1.25rem; margin-bottom: 1rem; border-left: 4px solid {color_hex};'>
                        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>
                            <div style='display: flex; align-items: center;'>
                                <div style='background: #E3F2FD; width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 1rem;'>
                                    <span style='font-weight: bold; color: #0D47A1;'>{idx + 1}</span>
                                </div>
                                <div>
                                    <h3 style='margin: 0; color: #2c3e50;'>{file_names[text_idx_int] if 0 <= text_idx_int < len(file_names) else f'Unknown ({text_idx})'}</h3>
                                    <div style='font-size: 0.9rem; color: #666;'>Match Score: <span style='font-weight: 600; color: {color_hex};'>{score:.1f}%</span></div>
                                </div>
                            </div>
                            <div style='display: flex; gap: 0.5rem;'>
                                <button class='stButton' style='background: #E3F2FD; color: #0D47A1; border: none; padding: 0.5rem 1rem; border-radius: 6px; cursor: pointer; font-weight: 500;'>View Details</button>
                                <button class='stButton' style='background: #E8F5E9; color: #2E7D32; border: none; padding: 0.5rem 1rem; border-radius: 6px; cursor: pointer; font-weight: 500;'>Download</button>
                            </div>
                        </div>
                        <div style='width: 100%; background: #f0f0f0; border-radius: 10px; height: 8px; margin-top: 0.75rem; overflow: hidden;'>
                            <div style='width: {score}%; background: {color_hex}; height: 100%; border-radius: 10px;'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Add summary statistics
            st.markdown("""
            <div style='margin: 2rem 0;'>
                <h3 style='color: #2c3e50; margin-bottom: 1rem;'>📈 Match Score Distribution</h3>
                <div style='background: white; border-radius: 10px; padding: 1.5rem; box-shadow: 0 2px 10px rgba(0,0,0,0.05);'>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 1rem;'>
                        <div>
                            <div style='font-size: 0.9rem; color: #666;'>Average Match</div>
                            <div style='font-size: 1.8rem; font-weight: 600; color: #1E88E5;'>{0:.1f}%</div>
                        </div>
                        <div>
                            <div style='font-size: 0.9rem; color: #666;'>Top Candidate</div>
                            <div style='font-size: 1.8rem; font-weight: 600; color: #43A047;'>{1:.1f}%</div>
                        </div>
                        <div>
                            <div style='font-size: 0.9rem; color: #666;'>Candidates</div>
                            <div style='font-size: 1.8rem; font-weight: 600; color: #7E57C2;'>{2}</div>
                        </div>
                    </div>
                    <div style='height: 10px; background: #f0f0f0; border-radius: 5px; overflow: hidden; margin-top: 1.5rem;'>
                        <div style='width: 100%; height: 100%; background: linear-gradient(90deg, #43A047, #1E88E5, #7E57C2); border-radius: 5px;'></div>
                    </div>
                </div>
            </div>
            """.format(
                sum(score for score, _ in ranked_resumes) / len(ranked_resumes) if ranked_resumes else 0,
                ranked_resumes[0][0] if ranked_resumes else 0,
                len(ranked_resumes)
            ), unsafe_allow_html=True)
            
            # Add a section for actions
            st.markdown("""
            <div style='margin: 2rem 0; text-align: center;'>
                <h3 style='color: #2c3e50; margin-bottom: 1rem;'>Next Steps</h3>
                <div style='display: flex; justify-content: center; gap: 1rem; margin-top: 1.5rem;'>
                    <button class='stButton' style='background: #1E88E5; color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 8px; cursor: pointer; font-weight: 500; display: flex; align-items: center;'>
                        <span style='margin-right: 0.5rem;'>📤</span> Export Results
                    </button>
                    <button class='stButton' style='background: white; color: #1E88E5; border: 1px solid #1E88E5; padding: 0.75rem 1.5rem; border-radius: 8px; cursor: pointer; font-weight: 500; display: flex; align-items: center;'>
                        <span style='margin-right: 0.5rem;'>🔄</span> Process New Batch
                    </button>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add a disclaimer
            st.markdown("""
            <div style='margin-top: 3rem; padding: 1rem; background: #FFF8E1; border-radius: 8px; border-left: 4px solid #FFA000;'>
                <div style='display: flex; align-items: flex-start;'>
                    <span style='font-size: 1.2rem; margin-right: 0.75rem;'>ℹ️</span>
                    <div>
                        <p style='margin: 0 0 0.5rem 0; color: #5D4037; font-weight: 500;'>About These Results</p>
                        <p style='margin: 0; color: #5D4037; font-size: 0.9rem;'>These rankings are based on semantic analysis of the resumes against the job description. We recommend reviewing the top candidates in detail before making any hiring decisions.</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Prepare enhanced data for visualization with safe access to file_names
            candidate_data = []
            for score, idx in ranked_resumes:
                try:
                    idx_int = int(round(float(idx)))  # Convert to int safely
                    candidate_name = file_names[idx_int] if 0 <= idx_int < len(file_names) else f"Unknown ({idx})"
                    candidate_data.append({
                        'Candidate': candidate_name,
                        'Match Score': float(score),  # Ensure score is a float
                        'Experience': "5+ years",  # Placeholder - would come from actual data
                        'Skills': "Python, Machine Learning, Data Analysis",  # Placeholder
                        'Education': "Master's Degree",  # Placeholder
                        'Original_Index': idx_int
                    })
                except (ValueError, TypeError) as e:
                    st.warning(f"Skipping invalid resume index {idx}: {str(e)}")
                    continue
            
            if not candidate_data:
                st.error("No valid resume data to display. Please check your input files and try again.")
                st.stop()
                
            df_ranks = pd.DataFrame(candidate_data)

            # Sort by match score
            df_ranks = df_ranks.sort_values('Match Score', ascending=False).reset_index(drop=True)
            df_ranks.index = df_ranks.index + 1  # Start index from 1

            # Single column layout for the bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#4CAF50' if x == df_ranks['Match Score'].max() else '#2196F3' for x in df_ranks['Match Score']]
            bars = ax.barh(df_ranks['Candidate'], df_ranks['Match Score'], color=colors, height=0.6)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
                       f'{width:.1f}%',
                       ha='left', va='center',
                       fontweight='bold')
            
            ax.set_xlim(0, 110)
            ax.set_xlabel('Match Score (%)', fontweight='bold')
            ax.set_ylabel('')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

            # Display results as an interactive table
            st.markdown("### 📊 Candidate Rankings")
            
            # Define a function to style the DataFrame
            def highlight_rows(row):
                score = row['Match Score']
                if score >= 80:
                    return ['background-color: #E8F5E9'] * len(row)
                elif score >= 60:
                    return ['background-color: #FFF8E1'] * len(row)
                else:
                    return ['background-color: #FFEBEE'] * len(row)
            
            # Apply styling to the DataFrame
            styled_df = df_ranks[['Candidate', 'Match Score', 'Experience', 'Skills']].style.apply(highlight_rows, axis=1)
            
            # Display the styled DataFrame
            st.dataframe(
                styled_df,
                column_config={
                    "Match Score": st.column_config.ProgressColumn(
                        "Match Score",
                        help="Match score with job description",
                        format="%f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "Candidate": "Candidate",
                    "Experience": "Experience",
                    "Skills": st.column_config.Column("Key Skills", width="large")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Add a section for candidate actions
            st.markdown("---")
            st.markdown("### 🎯 Candidate Actions")
            
            # Create columns for action buttons
            action_cols = st.columns(3)
            
            with action_cols[0]:
                if st.button("📤 Export All Results", use_container_width=True):
                    # Convert DataFrame to CSV
                    csv = df_ranks.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💾 Download as CSV",
                        data=csv,
                        file_name="candidate_rankings.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with action_cols[1]:
                if st.button("📧 Contact Top Candidates", use_container_width=True, 
                           help="Send emails to top candidates"):
                    st.session_state['show_email_modal'] = True
            
            with action_cols[2]:
                if st.button("🔄 Process New Batch", use_container_width=True,
                           type="secondary"):
                    st.session_state['page'] = "Upload & Process"
                    st.rerun()
            
            # Add a section for detailed candidate comparison
            st.markdown("---")
            st.markdown("### 🔍 Compare Candidates")
            
            # Allow selecting multiple candidates for comparison
            candidate_options = [f"{name} ({score:.1f}%)" for name, score in 
                              zip(df_ranks['Candidate'], df_ranks['Match Score'])]
            
            selected_candidates = st.multiselect(
                "Select up to 3 candidates to compare:",
                options=candidate_options,
                default=candidate_options[:min(3, len(candidate_options))],
                max_selections=3,
                placeholder="Select candidates..."
            )
            
            if selected_candidates:
                # Create a comparison table
                st.markdown("#### Comparison Summary")
                
                # Extract selected candidate data
                selected_data = []
                for candidate in selected_candidates:
                    name = candidate.split(' (')[0]
                    candidate_info = df_ranks[df_ranks['Candidate'] == name].iloc[0]
                    selected_data.append(candidate_info)
                
                # Display comparison
                comparison_cols = st.columns(len(selected_data))
                
                for idx, (col, candidate) in enumerate(zip(comparison_cols, selected_data)):
                    with col:
                        st.markdown(f"""
                        <div style='background: #f5f9ff; border-radius: 10px; padding: 1.5rem; text-align: center; height: 100%;'>
                            <div style='font-size: 2rem; margin-bottom: 1rem;'>
                                {['🥇', '🥈', '🥉'][idx] if idx < 3 else '👤'}
                            </div>
                            <h3 style='margin: 0 0 0.5rem 0; color: #1E88E5;'>{candidate['Candidate']}</h3>
                            <div style='font-size: 1.8rem; font-weight: bold; color: #0D47A1; margin-bottom: 1rem;'>
                                {candidate['Match Score']:.1f}%
                            </div>
                            <div style='text-align: left; margin-top: 1rem;'>
                                <p style='margin: 0.5rem 0;'><strong>Experience:</strong> {candidate['Experience']}</p>
                                <p style='margin: 0.5rem 0;'><strong>Education:</strong> {candidate['Education']}</p>
                                <p style='margin: 0.5rem 0;'><strong>Key Skills:</strong> {candidate['Skills']}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            # Download results as CSV
            csv = df_ranks.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="resume_rankings.csv">Download Rankings as CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

        with tab2:
            st.markdown("### Detailed Candidate Analysis")

            # Create a selectbox to choose a candidate with safe indexing
            candidate_options = []
            valid_indices = []
            for i, (score, idx) in enumerate(ranked_resumes):
                try:
                    idx_int = int(round(float(idx)))
                    if 0 <= idx_int < len(file_names):
                        candidate_options.append(f"#{i + 1}: {file_names[idx_int]} ({float(score):.2f}%)")
                        valid_indices.append((float(score), idx_int))
                    else:
                        st.warning(f"Skipping out-of-bounds resume index {idx}")
                except (ValueError, TypeError) as e:
                    st.warning(f"Skipping invalid resume index {idx}: {str(e)}")
            
            if not candidate_options:
                st.error("No valid candidates to display. Please check your input files and try again.")
                st.stop()
                
            selected_candidate = st.selectbox("Select Candidate", candidate_options)

            # Get the selected candidate index
            selected_idx = candidate_options.index(selected_candidate)
            if 0 <= selected_idx < len(valid_indices):
                match_score, resume_idx = valid_indices[selected_idx]
            else:
                st.error("Invalid selection. Please try again.")
                st.stop()

            # Display candidate information
            st.markdown(f"**Selected Candidate**: {file_names[resume_idx]}")
            st.markdown(f"**Match Score**: {match_score:.2f}%")

            # Create a radar chart for visualizing match in different areas
            # This is a mockup - in a real application, you'd want to break down
            # the match score into different categories

            # Mock data for radar chart (in a real app, you'd calculate these)
            categories = ['Technical Skills', 'Experience', 'Education', 'Soft Skills', 'Overall Match']
            values = [
                min(100, match_score + np.random.uniform(-10, 10)),
                min(100, match_score + np.random.uniform(-15, 15)),
                min(100, match_score + np.random.uniform(-5, 5)),
                min(100, match_score + np.random.uniform(-10, 10)),
                match_score
            ]
            values = [max(0, min(100, v)) for v in values]  # Ensure between 0-100

            # Create radar chart
            categories = np.array(categories)
            N = len(categories)

            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            values = np.array(values)
            values = np.concatenate((values, [values[0]]))  # Close the loop
            angles = np.concatenate((angles, [angles[0]]))  # Close the loop
            categories = np.concatenate((categories, [categories[0]]))  # Close the loop

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            ax.plot(angles, values, 'o-', linewidth=2, color='#1E88E5')
            ax.fill(angles, values, alpha=0.25, color='#1E88E5')
            ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
            ax.set_ylim(0, 100)
            ax.grid(True)
            ax.set_title('Candidate Match Analysis', size=15)

            st.pyplot(fig)

            # Key highlights section
            st.markdown("### Key Highlights")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("#### Strengths")
                # In a real application, you would use NLP to extract actual strengths
                st.markdown("- Strong technical background")
                st.markdown("- Relevant experience in the field")
                st.markdown("- Education aligns with requirements")
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("#### Areas for Improvement")
                # In a real application, you would use NLP to extract actual gaps
                st.markdown("- Limited experience with specific tools")
                st.markdown("- Could benefit from more leadership roles")
                st.markdown("- Few mentioned soft skills")
                st.markdown("</div>", unsafe_allow_html=True)

        with tab3:
            st.markdown("### Resume Content")

            # Create a selectbox to choose a candidate with safe indexing
            content_candidate_options = []
            content_valid_indices = []
            for i, (score, idx) in enumerate(ranked_resumes):
                try:
                    idx_int = int(round(float(idx)))
                    if 0 <= idx_int < len(file_names):
                        content_candidate_options.append(f"#{i + 1}: {file_names[idx_int]} ({float(score):.2f}%)")
                        content_valid_indices.append((float(score), idx_int))
                    else:
                        st.warning(f"Skipping out-of-bounds resume index {idx} in content selection")
                except (ValueError, TypeError) as e:
                    st.warning(f"Skipping invalid resume index {idx} in content selection: {str(e)}")
            
            if not content_candidate_options:
                st.error("No valid resume content to display. Please check your input files and try again.")
                st.stop()
            
            content_selected_candidate = st.selectbox(
                "Select Candidate Resume",
                content_candidate_options,
                key="content_select"
            )

            # Get the selected candidate index
            if content_selected_candidate in content_candidate_options:
                content_selected_idx = content_candidate_options.index(content_selected_candidate)
                if 0 <= content_selected_idx < len(content_valid_indices):
                    _, content_resume_idx = content_valid_indices[content_selected_idx]
                else:
                    st.error("Invalid selection. Please try again.")
                    st.stop()
            else:
                st.error("Invalid selection. Please select a valid candidate.")
                st.stop()

            # Display resume content
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"**Resume Text for {file_names[content_resume_idx]}:**")
            st.markdown("<div class='highlight'>", unsafe_allow_html=True)
            st.text_area(
                "Extracted Resume Content",
                resume_texts[content_resume_idx],
                height=400,
                disabled=True
            )
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Job description comparison
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("**Job Description:**")
            st.markdown("<div class='highlight'>", unsafe_allow_html=True)
            st.text_area(
                "Job Description Content",
                job_desc,
                height=200,
                disabled=True
            )
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)