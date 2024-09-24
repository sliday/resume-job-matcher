import os
import sys
import json
import PyPDF2
import anthropic
import openai
from glob import glob
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import logging
from termcolor import colored
import time
import json5
import requests
from bs4 import BeautifulSoup
from PIL import Image
import io
import statistics
import base64

# Initialize logging with more detailed format
logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the Anthropic client globally
default_anthropic_client = anthropic.Anthropic(api_key=os.environ.get("CLAUDE_API_KEY"))

# Initialize the OpenAI client globally
default_openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Global variable to store the chosen API
chosen_api = "anthropic"

import os
from termcolor import colored

def choose_api():
    global chosen_api
    prompt = "Use OpenAI API instead of Anthropic? [y/N]: "
    choice = input(colored(prompt, "cyan")).strip().lower()
    
    if choice in ["y", "yes"]:
        chosen_api = "openai"
    else:
        chosen_api = "anthropic"
    
    print(colored(f"\nSelected API: {chosen_api.capitalize()}", "green", attrs=["bold"]))

def talk_to_ai(prompt, max_tokens=1000, image_data=None, client=None):
    global chosen_api
    
    if chosen_api == "anthropic":
        return talk_to_anthropic(prompt, max_tokens, image_data, client)
    else:
        return talk_to_openai(prompt, max_tokens, image_data, client)

def talk_to_anthropic(prompt, max_tokens=1000, image_data=None, client=None):
    if client is None:
        client = default_anthropic_client
    
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    
    if image_data:
        for img in image_data:
            base64_image = base64.b64encode(img).decode('utf-8')
            messages[0]["content"].append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64_image
                }
            })
    
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=max_tokens,
            messages=messages
        )
        return response.content[0].text.strip()
    except Exception as e:
        logging.error(f"Error in Anthropic AI communication: {str(e)}")
        return None

def talk_to_openai(prompt, max_tokens=1000, image_data=None, client=None):
    if client is None:
        client = default_openai_client
    
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    
    if image_data:
        model = "gpt-4-vision-preview"
        for img in image_data:
            base64_image = base64.b64encode(img).decode('utf-8')
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
    else:
        model = "gpt-4-turbo-preview"
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error in OpenAI communication: {str(e)}")
        return None

def rank_job_description(job_desc, client=None):
    prompt = f"""
As a hiring consultant, analyze the following job description and provide a ranking based on modern best practices. Also, suggest 3-5 tips for improvement.

Job Description:
{job_desc}

Criteria for Effective Job Descriptions:
1. Clarity and Specificity (20%)
2. Inclusivity and Bias-Free Language (20%)
3. Company Culture and Values Description (15%)
4. Realistic and Prioritized Qualifications (15%)
5. Opportunities for Growth and Development (10%)
6. Compensation and Benefits Transparency (10%)
7. Search Engine Optimization (SEO) (5%)
8. Legal Compliance (5%)

Advanced scoring criteria:
  - name: 'Clarity and Specificity'
    weight: 20
    description: |
      The job description should clearly outline the responsibilities, required qualifications, and expectations without ambiguity.
    factors:
      - Use of clear and concise language
      - Detailed list of job responsibilities
      - Specific qualifications and experience required
      - Avoidance of vague terms like "sometimes," "maybe," or "as needed"

  - name: 'Inclusivity and Bias-Free Language'
    weight: 20
    description: |
      The job description should use inclusive language that encourages applications from a diverse range of candidates.
    factors:
      - Gender-neutral pronouns and job titles
      - Avoidance of ageist, ableist, or culturally biased language
      - Inclusion of diversity and inclusion statements

  - name: 'Company Culture and Values Description'
    weight: 15
    description: |
      The job description should provide insight into the company's culture, mission, and values to help candidates assess cultural fit.
    factors:
      - Clear statement of company mission and values
      - Description of team dynamics and work environment
      - Emphasis on aspects like innovation, collaboration, or employee development

  - name: 'Realistic and Prioritized Qualifications'
    weight: 15
    description: |
      The qualifications section should distinguish between essential and preferred qualifications to avoid deterring qualified candidates.
    factors:
      - Separate lists for mandatory and preferred qualifications
      - Realistic experience and education requirements
      - Justification for any stringent requirements

  - name: 'Opportunities for Growth and Development'
    weight: 10
    description: |
      The job description should mention any opportunities for career advancement, professional development, or training.
    factors:
      - Information on potential career paths within the company
      - Availability of training programs or educational assistance
      - Mentorship or leadership development opportunities

  - name: 'Compensation and Benefits Transparency'
    weight: 10
    description: |
      Providing information on compensation ranges and benefits can attract candidates aligned with what the company offers.
    factors:
      - Inclusion of salary range or compensation package details
      - Highlighting key benefits (e.g., health insurance, retirement plans)
      - Mention of unique perks (e.g., remote work options, flexible hours)

  - name: 'Search Engine Optimization (SEO)'
    weight: 5
    description: |
      The job description should be optimized with relevant keywords to improve visibility in job searches.
    factors:
      - Use of industry-standard job titles
      - Inclusion of relevant keywords and phrases

Provide a score for each criterion and an overall score out of 100. Then, suggest 3-5 tips for improvement.

Output your response in the following JSON format:
{{
  "scores": {{
    "clarity_specificity": 0,
    "inclusivity": 0,
    "company_culture": 0,
    "realistic_qualifications": 0,
    "growth_opportunities": 0,
    "compensation_transparency": 0,
    "seo": 0,
    "legal_compliance": 0
  }},
  "overall_score": 0,
  "improvement_tips": [
    "Tip 1",
    "Tip 2",
    "Tip 3"
  ]
}}

Strictly JSON. No explanations. No ```json``` wrappers.
"""

    try:
        response_text = talk_to_ai(prompt, max_tokens=200, client=client)
        if response_text:
            response = json5.loads(response_text)
            return response
        else:
            return None
    except Exception as e:
        logging.error(f"Error ranking job description: {str(e)}")
        return None

def extract_text_and_image_from_pdf(file_path):
    import pytesseract
    from pdf2image import convert_from_path
    from PyPDF2 import PdfReader

    try:
        text = ""
        resume_images = []

        # Extract text from the PDF using PyPDF2
        reader = PdfReader(file_path)
        first_page_text = ""
        if reader.pages:
            first_page = reader.pages[0]
            first_page_text = first_page.extract_text()
            if first_page_text:
                text += first_page_text

        # Extract image from the first page
        images = convert_from_path(file_path, first_page=1, last_page=1)
        if images:
            img = images[0]
            # Convert to grayscale and compress image
            img_gray = img.convert('L')
            img_buffer = io.BytesIO()
            img_gray.save(img_buffer, format='JPEG', quality=51)
            img_buffer.seek(0)

            # Add image data to resume_images list
            resume_images.append(img_buffer.getvalue())

            # If text extraction is insufficient, perform OCR
            if not first_page_text or len(first_page_text.strip()) < 500:
                ocr_text = pytesseract.image_to_string(Image.open(img_buffer))
                text += ocr_text

        else:
            logging.error(f"No images found in PDF {file_path}")

        return text, resume_images

    except Exception as e:
        logging.error(f"Error extracting text and image from PDF {file_path}: {str(e)}")
        return "", []

def assess_resume_quality(resume_images, client=None):
    prompt = """
You are a Resume Clarity and Visual Appeal Scoring expert.

Let's define criteria to assess the clarity and visual appeal of a candidate's resume.

criteria:
  - name: 'Formatting and Layout'
    weight: 10
    description: |
      Assess the overall formatting and layout of the resume. Points are awarded for consistent formatting, proper alignment, and effective use of white space.
    factors:
      - Consistent font styles and sizes
      - Proper alignment of text and sections
      - Effective use of white space to enhance readability
      - Appropriate margins and spacing

  - name: 'Section Organization and Headings'
    weight: 15
    description: |
      Evaluate the organization of content into clear sections with appropriate headings. Points are awarded for logical flow and ease of navigation.
    factors:
      - Clear and descriptive section headings
      - Logical sequence of sections (e.g., summary, experience, education)
      - Use of subheadings where appropriate
      - Ease of locating key information

  - name: 'Clarity and Conciseness of Content'
    weight: 25
    description: |
      Assess the clarity and conciseness of the information presented. Points are awarded for clear language, avoidance of jargon, and concise descriptions.
    factors:
      - Use of clear and straightforward language
      - Concise bullet points
      - Avoidance of unnecessary jargon or buzzwords
      - Focus on relevant information

  - name: 'Visual Elements and Design'
    weight: 20
    description: |
      Evaluate the visual appeal of the resume, including the use of visual elements such as icons, color accents, or charts, if appropriate for the industry.
    factors:
      - Appropriate use of color accents
      - Inclusion of relevant visual elements (e.g., icons, charts)
      - Consistency in design elements
      - Professional appearance suitable for the industry

  - name: 'Grammar and Spelling'
    weight: 20
    description: |
      Assess the resume for grammatical correctness and spelling accuracy. Points are deducted for errors.
    factors:
      - Correct grammar usage
      - Accurate spelling throughout
      - Proper punctuation
      - Professional tone and language

  - name: 'Length and Completeness'
    weight: 10
    description: |
      Evaluate whether the resume is of appropriate length and includes all necessary sections without unnecessary filler.
    factors:
      - Resume length appropriate for experience level (typically 1-2 pages)
      - Inclusion of all relevant sections
      - Absence of irrelevant or redundant information

# Additional Settings

max_total_score: 100  # Scores from all criteria sum up to this maximum

notes: |
  - The clarity and visual appeal scoring is separate from the job matching score.
  - These criteria aim to assess how effectively the candidate presents their information.
  - A well-formatted resume can enhance readability and make a strong first impression.

Based on the image(s) of the resume provided, please assess its quality according to the criteria above. For each criterion, provide a score (out of its maximum weight) and a brief explanation. Then, calculate the total weighted score (out of 100).

Output your response in the following JSON format:
{
  "total_score": Float
}

No explanations. No ```json``` wrappers.
    """

    try:
        response_text = talk_to_ai(prompt, max_tokens=100, image_data=resume_images, client=client)
        if response_text:
            response = json5.loads(response_text)
            return response['total_score']
        return 0
    except Exception as e:
        logging.error(f"Error assessing resume quality: {str(e)}")
        return 0

def match_resume_to_job(resume_text, job_desc, file_path, resume_images, client=None):
    prompt = f"""Your role is RESUME HR EXPERT. Your goal is to compare the following resume to the job description and provide a match score from 0 to 100.

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
    technical_skills_weight: 40  # Percentage weight for technical skills
    soft_skills_weight: 20       # Percentage weight for soft skills
    experience_weight: 20        # Percentage weight for experience
    education_weight: 10         # Percentage weight for education
    language_proficiency_weight: 5  # Weight adjustable based on job requirements
    certifications_weight: 5        # Weight adjustable based on job requirements

# Step 2: Define Scoring Criteria Based on Extracted Requirements
criteria:
  - name: 'Language Proficiency'
    weight: "job_description.emphasis.language_proficiency_weight"
    scoring_logic:
      description: |
        Assign points based on the candidate's proficiency in languages relevant to the job. Full points are awarded for meeting the required proficiency. Multilingual abilities are valued when relevant.
      required_languages: ['English']  # Adjust based on job description
      proficiency_levels:
        'Native': 100
        'Fluent': 90
        'Professional Working Proficiency': 80
        'Intermediate': 70
        'Basic': 60
      multilingual_bonus_per_language: 5  # Additional points per relevant language

  - name: 'Education Level'
    weight: "job_description.emphasis.education_weight"
    scoring_logic:
      description: |
        Assign points based on the candidate's highest level of education or equivalent experience. Emphasize relevant knowledge and skills over formal education. Alternative education paths and significant relevant experience are equally valued.
      levels:
        'PhD': 100
        'Masters': 90
        'Bachelors': 80
        'Associate': 70
        'High School or Equivalent': 60
        'No Formal Education': 50
      alternative_paths_bonus: 20  # Points for relevant certifications, bootcamps, or self-directed learning
      required_level: "job_description.required_education_level"

  - name: 'Years of Experience'
    weight: "job_description.emphasis.experience_weight"
    scoring_logic:
      description: |
        Assign points based on the relevance and quality of experience. Full points are awarded for meeting required years with relevant experience. Additional points for significant achievements.
      required_years: "job_description.required_experience_years"
      max_points: 100
      experience_points_formula: |
        If candidate_relevant_years >= required_years:
          points = 100
        Else:
          points = (candidate_relevant_years / required_years) * 100
      additional_relevance_bonus: 10  # Bonus for highly relevant experience

  - name: 'Technical Skills'
    weight: "job_description.emphasis.technical_skills_weight"
    scoring_logic:
      description: |
        Assign points for each required and optional skill, considering proficiency level. Emphasize required skills but value transferable skills and learning ability.
      proficiency_levels:
        'Expert': 100
        'Advanced': 85
        'Intermediate': 70
        'Beginner': 50
        'Familiar': 30
      required_skills_weight: 1.0
      optional_skills_weight: 0.7
      transferable_skills_bonus: 10  # Bonus for transferable skills
      required_skills: "job_description.required_skills"
      optional_skills: "job_description.optional_skills"
      keywords_points: 2  # Additional points per matched keyword
      keywords: "job_description.keywords_to_match"

  - name: 'Certifications'
    weight: "job_description.emphasis.certifications_weight"
    scoring_logic:
      description: |
        Assign points for each relevant certification. Practical experience and self-learning demonstrating equivalent expertise are equally valued.
      certifications_points: 5  # Points per relevant certification
      certifications_preferred: "job_description.certifications_preferred"
      equivalent_experience_bonus: 5  # Points for practical equivalent experience
      self_learning_projects_bonus: 5  # Points for self-directed projects

  - name: 'Soft Skills'
    weight: "job_description.emphasis.soft_skills_weight"
    scoring_logic:
      description: |
        Assign points for each soft skill demonstrated through examples or achievements. Emphasize importance in team dynamics and culture.
      soft_skills_points: 5  # Base points per soft skill
      proficiency_bonus: 5   # Additional points if demonstrated
      soft_skills: "job_description.soft_skills"

  - name: 'Relevance of Experience'
    weight: 10  # Weight for relevant roles or industries
    scoring_logic:
      description: |
        Assign points for experience in similar or related roles or industries. Value transferable skills and adaptability.
      relevant_roles_points: 10  # Points for matching roles
      related_roles_points: 5    # Points for related roles
      relevant_industries: "job_description.keywords_to_match"
      consider_transferable_skills: true

  - name: 'Non-Traditional Experience'
    weight: 5
    scoring_logic:
      description: |
        Assign points for relevant non-traditional experience such as open-source contributions, personal projects, volunteer work, or self-directed learning.
      max_points: 100
      scoring_method: |
        Evaluate relevance and impact of experiences. Assign points proportionally.

# Step 3: Additional Settings
max_total_score: 100  # Normalize scores to this maximum
normalization_method: |
  Total scores are scaled to a maximum of 100. Each criterion's score is calculated based on its weight relative to total emphasis weights. Penalties are subtracted after scoring and normalization.

diversity_and_inclusion_policy:
  description: |
    The scoring system promotes fairness and minimizes bias. It values diverse backgrounds and focuses on abilities, potential, and contributions over traditional metrics.
career_breaks_policy:
  description: |
    Candidates are not penalized for career breaks or non-linear paths. The focus is on relevance and quality of experience, skills, and potential.

# Penalizing Rules
red_flags:
  - name: 'Inconsistencies in Employment History'
    description: |
      Penalize for unexplained gaps or overlapping dates in employment history that are not accounted for.
    penalty_points: 5  # Points to deduct from total score
    evaluation_logic: |
      If gaps longer than 6 months are present without explanation, apply penalty.

  - name: 'Misrepresentation of Qualifications'
    description: |
      Penalize if there is evidence that the candidate has exaggerated or falsified qualifications, certifications, or experience.
    penalty_points: 20
    evaluation_logic: |
      If discrepancies are found between stated qualifications and verifiable information, apply penalty.

  - name: 'Frequent Job Changes'
    description: |
      Penalize for multiple job changes within short periods without clear reasons, indicating potential lack of commitment.
    penalty_points: 5
    evaluation_logic: |
      If more than 3 job changes in the past 2 years without justification, apply penalty.

  - name: 'Unprofessional Resume Presentation'
    description: |
      Penalize for numerous typos, grammatical errors, or poor formatting that reflect a lack of attention to detail.
    penalty_points: 5
    evaluation_logic: |
      If significant errors are present affecting readability, apply penalty.

  - name: 'Negative References to Past Employers'
    description: |
      Penalize if the candidate speaks negatively about past employers or colleagues, indicating potential interpersonal issues.
    penalty_points: 5
    evaluation_logic: |
      If unprofessional comments are detected, apply penalty.

  - name: 'Unprofessional Contact Information'
    description: |
      Penalize for unprofessional email addresses or inappropriate content in contact information.
    penalty_points: 2
    evaluation_logic: |
      If contact information contains inappropriate language or nicknames, apply penalty.

  - name: 'Lack of Required Certifications or Licenses'
    description: |
      Penalize if the candidate lacks mandatory certifications or licenses required for the job.
    penalty_points: 10
    evaluation_logic: |
      If essential certifications are missing and not compensated by equivalent experience, apply penalty.

  - name: 'Plagiarism'
    description: |
      Penalize if there is evidence that the resume content has been plagiarized from templates or other sources without personalization.
    penalty_points: 10
    evaluation_logic: |
      If identical content is found elsewhere without customization, apply penalty.

  - name: 'Failure to Meet Non-Negotiable Requirements'
    description: |
      Penalize if the candidate does not meet essential legal or regulatory requirements (e.g., work authorization).
    penalty_points: 20
    evaluation_logic: |
      If non-negotiable requirements are not met, apply penalty.

notes: |
  - Penalties are subtracted from the total normalized score.
  - The total penalties should not reduce the score below zero.
  - The scoring system ensures fairness by focusing on job-relevant red flags.
  - All penalizations are applied uniformly to all candidates.

# Step 4: Calculate the final score: 0 - 100
The scoring logic should normalize the total score to a maximum of 100, then subtract any penalty points. The final score cannot be less than zero. Try to be precise.

# Step 5: Check your results and make sure you are happy with them.

MATERIALS:

Resume:
===
{resume_text}
===

Job Description (job_description.txt):
===
{job_desc}
===
Provide numeric score as the response and 1-paragraph long professional email response I can send to the candidate. No need to explain the score. You can only speak JSON.

Politely reject anyone below 90. Use personal details in reponse, it has to be personalized.

Output format:
{{
  "score": Float,
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
        response_text = talk_to_ai(prompt, max_tokens=350, image_data=resume_images, client=client)
        if response_text:
            response = json5.loads(response_text)
            score = response.get('score', 0)
            if not isinstance(score, (int, float)):
                raise ValueError(f"Invalid score type: {type(score)}. Expected int or float.")
            
            email_response = response.get('email_response', '')
            subject_response = response.get('subject_response', '')
            match_reasons = response.get('match_reasons', '')
            website = response.get('website', '')
            
            os.makedirs('out', exist_ok=True)
            file_name = f"out/{os.path.splitext(os.path.basename(file_path))[0]}_response.txt"
            with open(file_name, 'w') as f:
                f.write(f"Subject: {subject_response}\n\n{email_response}")
            
            # Assess resume quality
            resume_quality_score = assess_resume_quality(resume_images)
            
            # IMPORTANT: Calculate final score
            # Normalize and combine AI-generated score with resume quality score
            normalized_combined_score = (score * 0.75 + resume_quality_score * 0.25)
            # Ensure the final score is between 0 and 100
            final_score = min(max(normalized_combined_score, 0), 100)
            
            return {'score': final_score, 'match_reasons': match_reasons, 'website': website}
        else:
            return {'score': 0.0, 'match_reasons': "Error: No response from AI", 'website': ''}
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
        {"min_score": 98,  "max_score": 101, "label": 'Cosmic Perfection',      "color": 'magenta',     "emoji": '🌟'},
        {"min_score": 95,  "max_score": 98,  "label": 'Unicorn Candidate',     "color": 'blue',        "emoji": '🦄'},
        {"min_score": 93,  "max_score": 95,  "label": 'Superstar Plus',        "color": 'cyan',        "emoji": '🌠'},
        {"min_score": 90,  "max_score": 93,  "label": 'Coding Wizard',         "color": 'green',       "emoji": '🧙'},
        {"min_score": 87,  "max_score": 90,  "label": 'Rockstar Coder',        "color": 'cyan',        "emoji": '🎸'},
        {"min_score": 85,  "max_score": 87,  "label": 'Coding Virtuoso',       "color": 'cyan',        "emoji": '🏆'},
        {"min_score": 83,  "max_score": 85,  "label": 'Tech Maestro',          "color": 'green',       "emoji": '🎭'},
        {"min_score": 80,  "max_score": 83,  "label": 'Awesome Fit',           "color": 'green',       "emoji": '🚀'},
        {"min_score": 78,  "max_score": 80,  "label": 'Stellar Match',         "color": 'green',       "emoji": '💫'},
        {"min_score": 75,  "max_score": 78,  "label": 'Great Prospect',        "color": 'green',       "emoji": '🌈'},
        {"min_score": 73,  "max_score": 75,  "label": 'Very Promising',        "color": 'light_green', "emoji": '🍀'},
        {"min_score": 70,  "max_score": 73,  "label": 'Solid Contender',       "color": 'light_green', "emoji": '🌴'},
        {"min_score": 68,  "max_score": 70,  "label": 'Strong Potential',      "color": 'yellow',      "emoji": '🌱'},
        {"min_score": 65,  "max_score": 68,  "label": 'Good Fit',              "color": 'yellow',      "emoji": '👍'},
        {"min_score": 63,  "max_score": 65,  "label": 'Promising Talent',      "color": 'yellow',      "emoji": '🌻'},
        {"min_score": 60,  "max_score": 63,  "label": 'Worthy Consideration',  "color": 'yellow',      "emoji": '🤔'},
        {"min_score": 58,  "max_score": 60,  "label": 'Potential Diamond',     "color": 'yellow',      "emoji": '💎'},
        {"min_score": 55,  "max_score": 58,  "label": 'Decent Prospect',       "color": 'yellow',      "emoji": '🍋'},
        {"min_score": 53,  "max_score": 55,  "label": 'Worth a Look',          "color": 'yellow',      "emoji": '🔍'},
        {"min_score": 50,  "max_score": 53,  "label": 'Average Candidate',     "color": 'yellow',      "emoji": '🌼'},
        {"min_score": 48,  "max_score": 50,  "label": 'Middling Match',        "color": 'yellow',      "emoji": '🍯'},
        {"min_score": 45,  "max_score": 48,  "label": 'Fair Possibility',      "color": 'yellow',      "emoji": '🌾'},
        {"min_score": 43,  "max_score": 45,  "label": 'Needs Polish',          "color": 'yellow',      "emoji": '💪'},
        {"min_score": 40,  "max_score": 43,  "label": 'Diamond in the Rough',  "color": 'yellow',      "emoji": '🐣'},
        {"min_score": 38,  "max_score": 40,  "label": 'Underdog Contender',    "color": 'light_yellow',"emoji": '🐕'},
        {"min_score": 35,  "max_score": 38,  "label": 'Needs Work',            "color": 'light_yellow',"emoji": '🛠'},
        {"min_score": 33,  "max_score": 35,  "label": 'Significant Gap',       "color": 'light_yellow',"emoji": '🌉'},
        {"min_score": 30,  "max_score": 33,  "label": 'Mismatch Alert',        "color": 'light_yellow',"emoji": '🚨'},
        {"min_score": 25,  "max_score": 30,  "label": 'Back to Drawing Board', "color": 'red',         "emoji": '🎨'},
        {"min_score": 20,  "max_score": 25,  "label": 'Way Off Track',         "color": 'red',         "emoji": '🚂'},
        {"min_score": 15,  "max_score": 20,  "label": 'Resume Misfire',        "color": 'red',         "emoji": '🎯'},
        {"min_score": 10,  "max_score": 15,  "label": 'Cosmic Mismatch',       "color": 'red',         "emoji": '☄'},
        {"min_score": 5,   "max_score": 10,  "label": 'Did You Mean to Apply?', "color": 'red',        "emoji": '🤷'},
        {"min_score": 0,   "max_score": 5,   "label": 'Oops! Wrong Universe',  "color": 'red',         "emoji": '🌀'},
    ]
    
    for range_info in score_ranges:
        min_score = range_info["min_score"]
        max_score = range_info["max_score"]
        if min_score <= score < max_score:
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
        resume_text, resume_images = extract_text_and_image_from_pdf(file)
        if not resume_text:
            return (os.path.basename(file), 0, "🔴", "red", "Error: Failed to extract text from PDF", "", "")
        result = match_resume_to_job(resume_text, job_desc, file, resume_images)
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
                    result = match_resume_to_job(combined_text, job_desc, file, resume_images)
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
        results = list(tqdm(pool.imap(worker, [(file, job_desc) for file in pdf_files]), total=len(pdf_files), desc=f"Processing {len(pdf_files)} resumes"))
    return results

def analyze_overall_matches(job_desc, results):
    # Prepare data for analysis
    match_data = []
    for filename, score, _, _, _, match_reasons, _ in results:
        match_data.append({
            "filename": filename,
            "score": score,
            "match_reasons": match_reasons
        })
    
    # Create a prompt for Claude AI
    prompt = f"""
As a hiring consultant, analyze the following resume match data and suggest adjustments to the job description to attract better candidates.

Job Description:
{job_desc}

Resume Match Data:
{json.dumps(match_data, indent=2)}

Provide a detailed analysis highlighting common strengths and weaknesses among the candidates. Suggest specific changes to the job description to improve candidate matches.

Output format, no more than 5 suggestions, 1-sentence long each:
- Observation or suggestion
...

Only output the suggestions, no intro, no explanations, no comments.
"""
    
    try:
        suggestions = talk_to_ai(prompt, max_tokens=500)
        if suggestions:
            print("\n\033[1mHow can I improve the job description?\033[0m")
            print(suggestions)
        else:
            print("\n\033[1mError: Unable to generate analysis and suggestions\033[0m")
    except Exception as e:
        logging.error(f"Error during overall match analysis: {str(e)}")

def improve_job_description(job_desc, ranking, client=None):
    prompt = f"""
As a hiring consultant, improve the following job description based on the ranking and improvement tips provided. Maintain the overall structure and key information while addressing the areas for improvement.

Original Job Description:
{job_desc}

Ranking:
{json.dumps(ranking, indent=2)}

Please provide an improved version of the job description that addresses the improvement tips and enhances the areas with lower scores. Output the improved job description as plain text, ready to be saved to a file.
"""

    try:
        improved_desc = talk_to_ai(prompt, max_tokens=1000, client=client)
        return improved_desc.strip() if improved_desc else None
    except Exception as e:
        logging.error(f"Error improving job description: {str(e)}")
        return None

def main():
    choose_api()
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

    # Prompt user for job description analysis
    analyze_job_desc = input("Would you like to analyze and improve the job description? (y/N): ").lower().strip() == 'y'

    if analyze_job_desc:
        # Rank job description
        job_desc_ranking = rank_job_description(job_desc)

        if job_desc_ranking:
            print("\n\033[1mJob Description Ranking\033[0m")
            for criterion, score in job_desc_ranking['scores'].items():
                print(colored(f"{criterion.replace('_', ' ').title()}: {score}%", 'cyan'))
            print(colored(f"\nOverall: {job_desc_ranking['overall_score']}%", 'yellow'))

            print("\n\033[1mImprovement Tips\033[0m")
            for tip in job_desc_ranking['improvement_tips']:
                print(colored(f"• {tip}", 'green'))

            # Improve job description
            improved_job_desc = improve_job_description(job_desc, job_desc_ranking)
            
            if improved_job_desc:
                with open('job_description_enhanced.txt', 'w') as f:
                    f.write(improved_job_desc)
                print(colored("\nEnhanced job description saved", 'green'))
            else:
                print(colored("\nFailed to enhance job description", 'red'))

    # Get all PDF files in the specified folder
    pdf_files = glob(os.path.join(pdf_folder, "*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {pdf_folder}")
        sys.exit(1)

    logging.info(f"Found {len(pdf_files)} PDF files in {pdf_folder}")
    logging.info("Starting resume processing...")

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
            score_str = f"{score:.0f}%" if isinstance(score, int) else f"{score:.2f}%"
            website_str = f" - {website}" if website else ""
            result_line = f"{emoji} \033[1m{filename:<{max_filename_length}}{website_str}\033[0m: {score_str} - {label}"
            top_score = max(top_score, score)
            bottom_score = min(bottom_score, score)
            total_score += score
            processed_count += 1
        
        print(colored(result_line, color))
        
        if score > 80 and match_reasons:
            print(colored(f"→ {match_reasons}", 'cyan'))

    if processed_count > 0:
        avg_score = total_score / processed_count

        # Collect all scores
        scores = [score for _, score, _, _, _, _, _ in sorted_results]

        # Calculate median score
        median_score = statistics.median(scores)

        # Calculate standard deviation
        if len(scores) > 1:
            std_dev_score = statistics.stdev(scores)
        else:
            std_dev_score = 0.0  # Standard deviation is zero if only one score

        # Count resumes above certain thresholds
        resumes_above_90 = sum(1 for s in scores if s >= 90)
        resumes_above_80 = sum(1 for s in scores if s >= 80)

        # Distribution of labels
        label_counts = {}
        for _, _, _, _, label, _, _ in sorted_results:
            label_counts[label] = label_counts.get(label, 0) + 1

        # Existing summary
        print("\n\033[1mSummary\033[0m")
        print(colored(f"Top Score: {top_score:.2f}%", 'yellow'))
        print(colored(f"Average: {avg_score:.2f}%", 'cyan'))
        print(colored(f"Median: {median_score:.2f}%", 'green'))
        print(colored(f"Standard Deviation: {std_dev_score:.2f}", 'magenta'))
        if resumes_above_90 > 0:
            print(colored(f"Resumes ≥ 90%: {resumes_above_90}", 'blue'))
        if resumes_above_80 > 0:
            print(colored(f"Resumes ≥ 80%: {resumes_above_80}", 'cyan'))
        print(colored(f"Lowest Score: {bottom_score:.2f}%", 'magenta'))
        print(colored(f"Processed: {processed_count}", 'green'))

        # Display label distribution with emojis
        print("\n\033[1mCandidates Distribution\033[0m")
        sorted_distribution = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        for label, count in sorted_distribution:
            emoji = next((e for _, _, e, _, l, _, _ in sorted_results if l == label), "")
            print(colored(f"{emoji} {label}: {count}", 'yellow'))

        # Analyze overall matches
        analyze_overall_matches(job_desc, sorted_results)

    if error_count > 0:
        print(colored(f"Errors: {error_count}", 'red'))
    
    logging.info("Resume processing completed.")
    print(colored("\nMatching Complete", 'yellow'))

if __name__ == "__main__":
    main()