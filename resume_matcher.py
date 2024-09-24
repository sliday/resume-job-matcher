import os
import sys
import json
import PyPDF2
import anthropic
from glob import glob
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import logging
from termcolor import colored
import time
import json5
import requests
from bs4 import BeautifulSoup  # Added import

# Initialize logging with more detailed format
logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the Anthropic client
client = anthropic.Anthropic(api_key=os.environ.get("CLAUDE_API_KEY"))

def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF {file_path}: {str(e)}")
        return ""

def match_resume_to_job(resume_text, job_desc, file_path):
    prompt = f"""Your role is RESUME HR EXPERT. Your goal is to compare the following resume to the job description and provide a match score from 0 to 100.

Method overview:
1. Read the resume and job description carefully.
2. Compare the skills, experience, and qualifications listed in the resume to the requirements of the job description.
3. Assign a score based on how well the resume matches the job description.

Advanced method:
# Enhanced Scoring Matching System Configuration

# Step 1: Parse Job Description to Extract Key Requirements
# The system will extract the following components from job_description.txt
job_description:
  required_experience_years: 5  # Extracted number of years of experience
  required_education_level: 'Masters'  # Extracted minimum education level
  required_skills:  # SAMPLE, see job_description.txt
    - 'Python'
    - 'Machine Learning'
    - 'Data Analysis'
    - 'Deep Learning'
  optional_skills:
    - 'Cloud Computing'
    - 'Big Data'
    - 'Computer Vision'
  certifications_preferred:
    - 'AWS Certified Solutions Architect'
    - 'Certified Data Scientist'
  soft_skills:
    - 'Communication'
    - 'Team Leadership'
    - 'Problem Solving'
    - 'Adaptability'
  keywords_to_match:
    - 'Neural Networks'
    - 'NLP'
    - 'Predictive Modeling'
  emphasis:
    technical_skills_weight: 50  # Percentage weight for technical skills
    soft_skills_weight: 20       # Percentage weight for soft skills
    experience_weight: 20        # Percentage weight for experience
    education_weight: 10         # Percentage weight for education

# Step 2: Define Scoring Criteria Based on Extracted Requirements
criteria:
  - name: 'Language Proficiency'
    weight: 10
    scoring_logic:
      description: |
        Assign points based on the candidate's language proficiency, with additional points for multilingual abilities or proficiency in languages relevant to the job.
      levels:
        'Native': 100
        'Fluent': 90
        'Professional Working Proficiency': 80
        'Intermediate': 70
        'Basic': 50
      multilingual_bonus: 10  # Additional points for each additional relevant language

  - name: 'Education Level'
    weight: "job_description.emphasis.education_weight"
    scoring_logic:
      description: |
        Assign points based on the candidate's highest degree or equivalent experience. If the candidate's education level meets or exceeds the required level, full points are awarded. If it is one level below, 75% of points are awarded. If two levels below, 50% of points are awarded. Relevant certifications or significant relevant experience can compensate for lacking formal education.
      levels:
        'PhD': 100
        'Masters': 90
        'Bachelors': 80
        'Associate': 70
        'High School': 60
        'No Formal Education': 50
      alternative_paths_bonus: 20  # Additional points for relevant certifications or significant experience
      required_level: "job_description.required_education_level"

  - name: 'Years of Experience'
    weight: "job_description.emphasis.experience_weight"
    scoring_logic:
      description: |
        Calculate points proportionally based on the required experience. Full points for meeting or exceeding required years. Additional points for highly relevant experience. Partial points for experience slightly below the required years.
      required_years: "job_description.required_experience_years"
      max_points: 100
      experience_points_formula: |
        If candidate_years >= required_years:
          points = 100 + (candidate_years - required_years) * 2  # Bonus points for extra years, up to a cap
        Else:
          points = (candidate_years / required_years) * 100
      max_bonus_points: 20  # Cap for bonus points

  - name: 'Technical Skills'
    weight: "job_description.emphasis.technical_skills_weight"
    scoring_logic:
      description: |
        Assign points for each required and optional skill, weighted by the candidate's proficiency level in each skill. Required skills have higher weight than optional skills. Proficiency levels are considered.
      proficiency_levels:
        'Expert': 100
        'Advanced': 80
        'Intermediate': 60
        'Beginner': 40
      required_skills_weight: 1.0  # Multiplier for required skills
      optional_skills_weight: 0.5  # Multiplier for optional skills
      required_skills: "job_description.required_skills"
      optional_skills: "job_description.optional_skills"
      keywords_points: 2          # Additional points per keyword matched
      keywords: "job_description.keywords_to_match"

  - name: 'Certifications'
    weight: 10  # Static weight unless specified in job description
    scoring_logic:
      description: |
        Assign points for each preferred certification the candidate possesses. Equivalent certifications or significant practical experience can also earn points.
      certifications_points: 5  # Points per certification
      certifications_preferred: "job_description.certifications_preferred"
      equivalent_certifications_bonus: 5  # Points for equivalent certifications
      practical_experience_bonus: 5       # Points for practical experience demonstrating certification-level expertise

  - name: 'Soft Skills'
    weight: "job_description.emphasis.soft_skills_weight"
    scoring_logic:
      description: |
        Assign points for each soft skill mentioned in the resume, considering the candidate's demonstrated proficiency or examples of these skills.
      soft_skills_points: 5     # Base points per soft skill
      proficiency_bonus: 5      # Additional points if proficiency is demonstrated through examples or achievements
      soft_skills: "job_description.soft_skills"

  - name: 'Relevance of Experience'
    weight: 10  # Additional weight for relevant job titles or industries
    scoring_logic:
      description: |
        Assign points if the candidate has worked in similar or related roles or industries. Consider transferable skills and experiences.
      relevant_titles_points: 10  # Points if matching job titles are found
      related_titles_points: 5    # Points for related job titles
      relevant_titles: "job_description.keywords_to_match"
      consider_transferable_skills: true

  - name: 'Non-Traditional Experience'
    weight: 5
    scoring_logic:
      description: |
        Assign points for relevant experience gained through non-traditional means, such as open-source contributions, personal projects, volunteer work, or self-directed learning.
      max_points: 100

# Step 3: Additional Settings
max_total_score: 100  # The scoring logic should normalize scores to this maximum
normalization_method: |
  The total score from all criteria will be scaled to a maximum of 100. Each criterion's score will be calculated based on its weight, and the sum will be normalized accordingly.
career_breaks_policy:
  description: |
    Candidates will not be penalized for career breaks. The scoring system will focus on the relevance and quality of experience, not just continuity.
notes: |
  - The scoring system dynamically adjusts criteria weights based on the job description.
  - Emphasizes the most critical aspects of the job requirements.
  - Encourages a more tailored and fair evaluation of candidates.
  - Considers alternative education paths, non-traditional experience, and transferable skills.
  - Does not penalize candidates for career breaks or non-linear career paths.

# Step 4: Calculate the final score: 0 - 100
The scoring logic should normalize the total score to a maximum of 100.

# Step 5: Check your results and make sure you are happy with them.

Resume:
===
{resume_text}
===

job_description.txt:
===
{job_desc}
===
Provide numeric score as the response and 1-paragraph long professional email response I can send to the candidate. No need to explain the score. You can only speak JSON.

Politely reject anyone below 90. Use personal details in reponse, it has to be personalized.

Output format:
{{
  "score": 85,
  "email_response": "Thank you for applying to Frontend Developer at Sky. Your skills impress us. We invite you to next stage. Expect contact for pair programming interview.",
  "subject_response": "Subject line of the response email",
  "match_reasons": "highlight1 | highlight2 | highlight3"
  "website": "personal_website_link_or_empty_string"
}}

Provide 3-4 key reasons for match. Use telegraphic English. Max 10 words per reason.
For the website, only include a personal website link if found in the resume. If no personal website is mentioned, leave it as an empty string.

Strictly JSON. No explanations. No \`\`\`json\`\`\` wrappers.
"""

    try:
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=350,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text.strip()
        
        logging.debug(f"Raw response for {file_path}: {response_text}")
        
        # Use json5 to parse the response
        response = json5.loads(response_text)
        
        score = response.get('score', 0)
        if not isinstance(score, (int, float)):
            raise ValueError(f"Invalid score type: {type(score)}. Expected int or float.")
        
        email_response = response.get('email_response', '')
        subject_response = response.get('subject_response', '')  # Get the subject line
        match_reasons = response.get('match_reasons', '')
        website = response.get('website', '')
        
        # Only keep the website if it's not empty and not 'usesky.ai'
        if website and 'usesky.ai' not in website:
            website = website
        else:
            website = ''
        
        os.makedirs('out', exist_ok=True)
        file_name = f"out/{os.path.splitext(os.path.basename(file_path))[0]}_response.txt"
        with open(file_name, 'w') as f:
            f.write(f"Subject: {subject_response}\n\n{email_response}")
        
        return {'score': float(score), 'match_reasons': match_reasons, 'website': website}
    except json5.JSONDecodeError as e:
        logging.error(f"Error parsing JSON for {file_path}: {str(e)}")
        logging.error(f"Raw response: {response_text}")
        return {'score': 0.0, 'match_reasons': f"Error: Invalid JSON response - {str(e)}", 'website': ''}
    except ValueError as e:
        logging.error(f"Invalid score for {file_path}: {str(e)}")
        return {'score': 0.0, 'match_reasons': f"Error: Invalid score - {str(e)}", 'website': ''}
    except Exception as e:
        logging.error(f"Unexpected error processing {file_path}: {str(e)}")
        return {'score': 0.0, 'match_reasons': f"Error: Unexpected - {str(e)}", 'website': ''}

def get_score_details(score):
    if not isinstance(score, (int, float)):
        raise ValueError(f"Invalid score type: {type(score)}. Expected int or float.")
    
    score = float(score)  # Ensure score is a float
    
    score_ranges = [
        {"min_score": 98, "label": 'Cosmic Perfection', "color": 'magenta', "emoji": '🌟🚀'},
        {"min_score": 95, "max_score": 97, "label": 'Unicorn Candidate', "color": 'blue', "emoji": '🦄✨'},
        {"min_score": 93, "max_score": 94, "label": 'Superstar Plus', "color": 'cyan', "emoji": '🌠💫'},
        {"min_score": 90, "max_score": 92, "label": 'Coding Wizard', "color": 'green', "emoji": '🧙‍♂️💻'},
        {"min_score": 87, "max_score": 89, "label": 'Rockstar Coder', "color": 'cyan', "emoji": '🎸'},
        {"min_score": 85, "max_score": 86, "label": 'Coding Virtuoso', "color": 'cyan', "emoji": '🏆'},
        {"min_score": 83, "max_score": 84, "label": 'Tech Maestro', "color": 'green', "emoji": '🧙‍♂️'},
        {"min_score": 80, "max_score": 82, "label": 'Awesome Fit', "color": 'green', "emoji": '🚀'},
        {"min_score": 78, "max_score": 79, "label": 'Stellar Match', "color": 'green', "emoji": '🌠'},
        {"min_score": 75, "max_score": 77, "label": 'Great Prospect', "color": 'green', "emoji": '🌈'},
        {"min_score": 73, "max_score": 74, "label": 'Very Promising', "color": 'light_green', "emoji": '🍀'},
        {"min_score": 70, "max_score": 72, "label": 'Solid Contender', "color": 'light_green', "emoji": '🌴'},
        {"min_score": 68, "max_score": 69, "label": 'Strong Potential', "color": 'yellow', "emoji": '🌱'},
        {"min_score": 65, "max_score": 67, "label": 'Good Fit', "color": 'yellow', "emoji": '👍'},
        {"min_score": 63, "max_score": 64, "label": 'Promising Talent', "color": 'yellow', "emoji": '🌻'},
        {"min_score": 60, "max_score": 62, "label": 'Worthy Consideration', "color": 'yellow', "emoji": '🤔'},
        {"min_score": 58, "max_score": 59, "label": 'Potential Diamond', "color": 'yellow', "emoji": '💎'},
        {"min_score": 55, "max_score": 57, "label": 'Decent Prospect', "color": 'yellow', "emoji": '🍋'},
        {"min_score": 53, "max_score": 54, "label": 'Worth a Look', "color": 'yellow', "emoji": '🔍'},
        {"min_score": 50, "max_score": 52, "label": 'Average Joe/Jane', "color": 'yellow', "emoji": '🌼'},
        {"min_score": 48, "max_score": 49, "label": 'Middling Match', "color": 'yellow', "emoji": '🍯'},
        {"min_score": 45, "max_score": 47, "label": 'Fair Possibility', "color": 'yellow', "emoji": '🌾'},
        {"min_score": 43, "max_score": 44, "label": 'Needs Polish', "color": 'yellow', "emoji": '💪'},
        {"min_score": 40, "max_score": 42, "label": 'Diamond in the Rough', "color": 'yellow', "emoji": '🐣'},
        {"min_score": 38, "max_score": 39, "label": 'Underdog Contender', "color": 'light_yellow', "emoji": '🐕'},
        {"min_score": 35, "max_score": 37, "label": 'Needs Work', "color": 'light_yellow', "emoji": '🛠��'},
        {"min_score": 33, "max_score": 34, "label": 'Significant Gap', "color": 'light_yellow', "emoji": '🌉'},
        {"min_score": 30, "max_score": 32, "label": 'Mismatch Alert', "color": 'light_yellow', "emoji": '🚨'},
        {"min_score": 25, "max_score": 29, "label": 'Back to Drawing Board', "color": 'red', "emoji": '🎨'},
        {"min_score": 20, "max_score": 24, "label": 'Way Off Track', "color": 'red', "emoji": '🚂'},
        {"min_score": 15, "max_score": 19, "label": 'Resume Misfire', "color": 'red', "emoji": '🎯'},
        {"min_score": 10, "max_score": 14, "label": 'Cosmic Mismatch', "color": 'red', "emoji": '☄️'},
        {"min_score": 5, "max_score": 9, "label": 'Did You Mean to Apply?', "color": 'red', "emoji": '🤷‍♂️'},
        {"min_score": 0, "max_score": 4, "label": 'Oops! Wrong Universe', "color": 'red', "emoji": '🌀🤖'},
    ]

    for range_info in score_ranges:
        if score >= range_info["min_score"] and (
            "max_score" not in range_info or score <= range_info["max_score"]
        ):
            return range_info["emoji"], range_info["color"], range_info["label"]

    return "💀", "red", "Unable to score"  # Fallback for any unexpected scores

def check_website(url):
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200, url
    except requests.RequestException:
        return False, url

def worker(args):
    file, job_desc = args
    try:
        resume_text = extract_text_from_pdf(file)
        if not resume_text:
            return (os.path.basename(file), 0, "🔴", "red", "Error: Failed to extract text from PDF", "", "")
        result = match_resume_to_job(resume_text, job_desc, file)
        score = result['score']
        match_reasons = result['match_reasons']
        website = result.get('website', '')
        
        # Check if the website is accessible
        if website:
            is_accessible, updated_url = check_website(website)
            if not is_accessible:
                score = max(0, score - 25)  # Reduce score, but not below 0
                website = f"{updated_url} (inactive)"
            else:
                website = updated_url
                # Fetch website content
                try:
                    response = requests.get(website, timeout=5)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.content, 'html.parser')
                    website_text = soup.get_text(separator=' ', strip=True)
                    
                    # Combine resume_text and website_text
                    combined_text = f"{resume_text}\n\nWebsite Content:\n{website_text}"
                    
                    # Re-run match_resume_to_job with combined_text
                    result = match_resume_to_job(combined_text, job_desc, file)
                    score = result['score']
                    match_reasons = result['match_reasons']
                except Exception as e:
                    logging.error(f"Error fetching website content for {file}: {str(e)}")
        
        emoji, color, label = get_score_details(score)
        return (os.path.basename(file), score, emoji, color, label, match_reasons, website)
    except Exception as e:
        logging.error(f"Error processing {file}: {str(e)}")
        return (os.path.basename(file), 0, "🔴", "red", f"Error: {str(e)}", "", "")

def process_resumes(job_desc, pdf_files):
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(worker, [(file, job_desc) for file in pdf_files]), total=len(pdf_files), desc="Processing resumes"))
    return results

def generate_ascii_art(text):
    ascii_art = f"""
    ╔{'═' * (len(text) + 2)}╗
    ║ {text} ║
    ╚{'═' * (len(text) + 2)}╝
    """
    return ascii_art

def resume_matching_dance():
    dance_moves = [
        "😀", "😃", "😄", "😁", "😆", "😊", "🙂", "😉", "😎", "🥳"
    ]
    for _ in range(10):
        for move in dance_moves:
            print(move, end='', flush=True)
            time.sleep(0.1)
            print('\b', end='', flush=True)

def main():
    if len(sys.argv) == 1:
        job_desc_file = "job_description.txt"
        if not os.path.exists(job_desc_file):
            job_desc_file = input("Enter the path to the job description file: ")
    else:
        job_desc_file = sys.argv[1]

    pdf_folder = sys.argv[2] if len(sys.argv) > 2 else "src"

    # Ensure the job description file exists
    while not os.path.exists(job_desc_file):
        print(f"File not found: {job_desc_file}")
        job_desc_file = input("Enter the path to the job description file: ")

    # Read job description
    with open(job_desc_file, 'r') as file:
        job_desc = file.read().strip()

    # Get all PDF files in the specified folder
    pdf_files = glob(os.path.join(pdf_folder, "*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {pdf_folder}")
        sys.exit(1)

    logging.info(f"Found {len(pdf_files)} PDF files in {pdf_folder}")
    logging.info("Starting resume processing...")

    print("\nLet's do the resume matching dance while we process the results!")
    resume_matching_dance()

    print("\n🎭 Resume Matching Results 🎭")
    results = process_resumes(job_desc, pdf_files)
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    top_score = 0
    bottom_score = 100
    total_score = 0
    processed_count = 0
    error_count = 0

    max_filename_length = max(len(filename) for filename, _, _, _, _, _, _ in sorted_results)

    for i, (filename, score, emoji, color, label, match_reasons, website) in enumerate(sorted_results):
        if emoji == "🔴":
            result_line = f"{emoji} \033[1m{filename:<{max_filename_length}}\033[0m: {label}"
            error_count += 1
        else:
            score_str = f"{score:.0f}%" if isinstance(score, (int, float)) and score.is_integer() else f"{score:.2f}%"
            website_str = f" - {website}" if website else ""
            result_line = f"{emoji} \033[1m{filename:<{max_filename_length}}{website_str}\033[0m : {score_str} - {label}"
            top_score = max(top_score, score)
            bottom_score = min(bottom_score, score)
            total_score += score
            processed_count += 1
        
        print(colored(result_line, color))
        
        if score > 80 and match_reasons:
            print(colored(f"→ {match_reasons}", 'cyan'))

    if processed_count > 0:
        avg_score = total_score / processed_count
        print("\n\033[1mResume Matching Summary\033[0m")
        print(colored(f"🏆 Top Score: {top_score:.0f}%" if top_score.is_integer() else f"🏆 Top Score: {top_score:.2f}%", 'yellow'))
        print(colored(f"🏅 Average Score: {avg_score:.2f}%", 'cyan'))
        print(colored(f"🏐 Bottom Score: {bottom_score:.0f}%" if bottom_score.is_integer() else f"🏐 Bottom Score: {bottom_score:.2f}%", 'magenta'))
        print(colored(f"📄 Processed Resumes: {processed_count}", 'green'))
    
    if error_count > 0:
        print(colored(f"⚠️ Errors encountered: {error_count}", 'red'))
    
    logging.info("Resume processing completed.")
    print(colored("\nResume Matching Complete! 🎉", 'yellow'))

if __name__ == "__main__":
    main()