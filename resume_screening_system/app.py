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
    page = st.radio("📌 Navigation", ["🏠 Home", "📤 Upload & Process", "📊 Results"])

    st.markdown("---")
    st.markdown("### About")
    st.info(
        "This application uses NLP and AI to match resumes with job descriptions. "
        "It analyzes the semantic similarity between the content of resumes and job requirements."
    )

# Home Page
if page == "🏠 Home":
    st.markdown("<h1 class='main-header'>AI Resume Screening & Ranking System</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### How It Works")
        st.markdown("""
        1. **Upload Resume PDFs**: Submit multiple candidate resumes in PDF format
        2. **Enter Job Description**: Provide detailed job requirements and qualifications
        3. **AI Analysis**: Our system extracts, processes, and analyzes the content
        4. **Get Rankings**: View candidate rankings with match scores and visualizations

        The system uses natural language processing and semantic similarity algorithms to match candidate skills and experiences with job requirements.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Features")
        st.markdown("""
        - ✅ Multi-resume processing
        - ✅ Semantic matching
        - ✅ Interactive visualizations
        - ✅ Text extraction & analysis
        - ✅ Candidate comparison
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Getting Started")
    st.markdown("""
    Click on "**Upload & Process**" in the sidebar to begin uploading resumes and job descriptions.
    """)

    if st.button("Go to Upload Section", type="primary"):
        st.session_state["page"] = "📤 Upload & Process"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "📤 Upload & Process":
    st.markdown("<h1 class='sub-header'>Upload Resumes & Job Description</h1>", unsafe_allow_html=True)

    # Job description input
    job_desc = st.text_area(
        "Job Description",
        placeholder="Paste the complete job description here...",
        height=200,
        value=st.session_state.get("job_description", "")
    )

    # File uploader
    st.markdown("### Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload Resumes (PDF format only)",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload multiple PDF files at once"
    )

    # Debugging: Show uploaded file details
    if uploaded_files:
        st.write(f"✅ {len(uploaded_files)} file(s) uploaded:")
        for file in uploaded_files:
            st.write(f"- {file.name} ({file.size} bytes)")

    # Process button
    if st.button("Process Resumes", type="primary", disabled=(not uploaded_files or not job_desc)):
        if not uploaded_files:
            st.error("⚠️ Please upload at least one resume file.")
        elif not job_desc.strip():
            st.error("⚠️ Job description cannot be empty.")
        else:
            # Show a progress bar for processing
            progress_bar = st.progress(0)
            status_text = st.empty()

            resume_texts = []
            file_names = []

            for i, file in enumerate(uploaded_files):
                try:
                    status_text.text(f"Extracting text from {file.name}...")

                    # Read file contents
                    text = extract_text_from_pdf(file)

                    if not text.strip():
                        st.warning(f"⚠️ No text found in {file.name}. It may be a scanned PDF.")

                    resume_texts.append(text)
                    file_names.append(file.name)

                    progress_bar.progress((i + 1) / len(uploaded_files))

                except Exception as e:
                    st.error(f"❌ Error processing {file.name}: {e}")
                    continue  # Skip the failed file

            if resume_texts:
                # Rank the resumes
                status_text.text("Ranking resumes...")
                ranked_resumes = rank_resumes(job_desc, resume_texts)

                # Save to session state
                st.session_state["ranked_resumes"] = ranked_resumes
                st.session_state["resume_texts"] = resume_texts
                st.session_state["resume_files"] = file_names
                st.session_state["job_description"] = job_desc

                # Complete progress bar
                progress_bar.progress(100)
                status_text.text("Processing complete!")

                st.success("✅ Resume analysis complete! Go to the Results page to view rankings.")

                if st.button("View Results"):
                    st.session_state["page"] = "📊 Results"
                    st.rerun()
            else:
                st.error("❌ No valid resumes processed. Please check the uploaded files.")


# Results Page
elif page == "📊 Results":
    st.markdown("<h1 class='sub-header'>Resume Ranking Results</h1>", unsafe_allow_html=True)

    if st.session_state["ranked_resumes"] is None:
        st.warning("⚠️ No results to display. Please upload resumes and a job description first.")
        if st.button("Go to Upload Section"):
            st.session_state["page"] = "📤 Upload & Process"
            st.rerun()
    else:
        ranked_resumes = st.session_state["ranked_resumes"]
        resume_texts = st.session_state["resume_texts"]
        file_names = st.session_state["resume_files"]
        job_desc = st.session_state["job_description"]

        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📈 Ranking Overview", "📝 Detailed Analysis", "📄 Resume Content"])

        with tab1:
            st.markdown("### Candidate Ranking")

            # Prepare data for visualization
            df_ranks = pd.DataFrame({
                'Candidate': [file_names[idx] for idx, _ in ranked_resumes],
                'Match Score': [score * 100 for _, score in ranked_resumes]
            })

            # Sort by match score
            df_ranks = df_ranks.sort_values('Match Score', ascending=False).reset_index(drop=True)
            df_ranks.index = df_ranks.index + 1  # Start index from 1

            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(df_ranks['Candidate'], df_ranks['Match Score'], color='#1E88E5')

            # Add percentages to the end of each bar
            for i, bar in enumerate(bars):
                ax.text(
                    bar.get_width() + 1,
                    bar.get_y() + bar.get_height() / 2,
                    f"{df_ranks['Match Score'].iloc[i]:.2f}%",
                    va='center'
                )

            ax.set_xlabel('Match Score (%)')
            ax.set_title('Candidate Ranking by Job Description Match')
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            ax.set_xlim(0, max(df_ranks['Match Score']) * 1.1)  # Add some space for labels

            # Display the chart
            st.pyplot(fig)

            # Display results as a table
            st.markdown("### Ranking Table")
            st.dataframe(
                df_ranks.style.background_gradient(cmap='Blues', subset=['Match Score']),
                use_container_width=True
            )

            # Download results as CSV
            csv = df_ranks.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="resume_rankings.csv">Download Rankings as CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

        with tab2:
            st.markdown("### Detailed Candidate Analysis")

            # Create a selectbox to choose a candidate
            candidate_options = [f"#{i + 1}: {file_names[idx]} ({score:.2f}%)"
                                 for i, (idx, score) in enumerate(ranked_resumes)]
            selected_candidate = st.selectbox("Select Candidate", candidate_options)

            # Get the selected candidate index
            selected_idx = candidate_options.index(selected_candidate)
            resume_idx = ranked_resumes[selected_idx][0]
            match_score = ranked_resumes[selected_idx][1] * 100

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

            # Create a selectbox to choose a candidate
            content_candidate_options = [f"#{i + 1}: {file_names[idx]} ({score:.2f}%)"
                                         for i, (idx, score) in enumerate(ranked_resumes)]
            content_selected_candidate = st.selectbox(
                "Select Candidate Resume",
                content_candidate_options,
                key="content_select"
            )

            # Get the selected candidate index
            content_selected_idx = content_candidate_options.index(content_selected_candidate)
            content_resume_idx = ranked_resumes[content_selected_idx][0]

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

# Footer
st.markdown("---")
st.markdown("📄 AI Resume Screening & Ranking System | Built with Streamlit and NLP")