import os
import sys
from resume_processing import extract_text_from_pdf, rank_resumes

def get_multiline_input(prompt):
    print(prompt)
    print("(Paste your text and then press Ctrl+Z on Windows or Ctrl+D on Linux and Enter to finish)")
    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass
    return "\n".join(lines)

def main():
    print("\n" + "="*80)
    print("                     AI RESUME SCREENER - TERMINAL VERSION")
    print("="*80)
    
    # Check for resumes folder
    resume_folder = "resumes"
    if not os.path.exists(resume_folder):
        os.makedirs(resume_folder)
        print(f"\n[!] Created '{resume_folder}' directory.")
        print("Please place your PDF resumes in this folder and run the script again.")
        return

    # Load pdf files
    pdf_files = [f for f in os.listdir(resume_folder) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"\n[!] No PDF resumes found in '{resume_folder}' folder.")
        print(f"Please place your PDF resumes in the '{os.path.abspath(resume_folder)}' folder and run the script again.")
        return

    # Get Job Description
    jd = get_multiline_input("\n[+] Step 1: Paste the Job Description below:")
    
    if not jd.strip():
        print("\n[!] Error: Job Description cannot be empty.")
        return

    print(f"\n[+] Step 2: Processing {len(pdf_files)} resumes from '{resume_folder}' folder...")
    
    resume_texts = []
    valid_file_names = []
    
    for filename in pdf_files:
        filepath = os.path.join(resume_folder, filename)
        try:
            text = extract_text_from_pdf(filepath)
            if text.strip():
                resume_texts.append(text)
                valid_file_names.append(filename)
            else:
                print(f"    - ⚠️ Skipping {filename}: No text content could be extracted.")
        except Exception as e:
            print(f"    - ❌ Error processing {filename}: {e}")

    if not resume_texts:
        print("\n[!] Error: No valid resume content could be extracted from the PDF files.")
        return

    print("[+] Step 3: Analyzing and Ranking candidates using ML Model...")
    # This might take a few seconds as it loads models and processes embeddings
    results = rank_resumes(jd, resume_texts)

    # Display Results
    print("\n" + "="*80)
    print(f"{'RANK':<5} | {'CANDIDATE FILENAME':<35} | {'MATCH SCORE':<12} | {'SEMANTIC'}")
    print("-" * 80)
    
    for i, res in enumerate(results):
        rank = i + 1
        name = valid_file_names[res['index']]
        score = f"{res['score']:.1f}%"
        semantic = f"{res['semantic_score']:.1f}%"
        
        # Color coding logic (simulated with symbols for pure terminal compatibility)
        marker = "⭐" if i == 0 else "  "
        
        print(f"{rank:<5} | {name[:35]:<35} | {score:<12} | {semantic}% {marker}")
        
    print("="*80)
    
    if results:
        top_res = results[0]
        top_name = valid_file_names[top_res['index']]
        print(f"\n🏆 TOP CANDIDATE: {top_name}")
        print(f"📊 Overall Match: {top_res['score']}%")
        print(f"🧠 Semantic Alignment: {top_res['semantic_score']}%")
        
        print("\n📝 AI Summary:")
        for bullet in top_res.get('ai_summary', []):
            print(f"  • {bullet}")
            
        print("\n✅ Matched Keywords:")
        matched = ", ".join(top_res.get('matched_keywords', []))
        print(f"  {matched if matched else 'None'}")
    
    print("\n" + "="*80)
    print("Analysis Complete.")

if __name__ == "__main__":
    # Suppress technical logs for a cleaner terminal experience
    import warnings
    warnings.filterwarnings("ignore")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nAn unexpected error occurred: {e}")
        sys.exit(1)
