import sys, json, json5, PyPDF2, anthropic, openai
from openai import OpenAI, OpenAIError
from glob import glob
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import logging
from termcolor import colored
import time, requests, statistics, base64, os, markdown, pdfkit, io
from bs4 import BeautifulSoup
from PIL import Image
from pathlib import Path
from weasyprint import HTML
import pdfkit
from pathlib import Path
import argparse

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
    
    try:
        if chosen_api == "anthropic":
            response = talk_to_anthropic(prompt, max_tokens, image_data, client)
        else:
            response = talk_to_openai(prompt, max_tokens, image_data, client)
        
        return response.strip() if response else ""
    except Exception as e:
        logging.error(f"Error in talk_to_ai: {str(e)}")
        return ""

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
        return ""

def talk_to_openai(prompt, max_tokens=1000, image_data=None, client=None):
    if client is None:
        client = default_openai_client
    
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    
    if image_data:
        model = "gpt-4o"
        for img in image_data:
            base64_image = base64.b64encode(img).decode('utf-8')
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
    else:
        model = "gpt-4o"
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error in OpenAI communication: {str(e)}")
        return ""

from pydantic import BaseModel, Field
from typing import List, Union
from enum import Enum

class ResponseType(str, Enum):
    score = "score"
    reasons = "reasons"
    url = "url"
    email = "email"

class Score(BaseModel):
    value: int = Field(..., ge=0, le=100)

class Reasons(BaseModel):
    items: List[str] = Field(..., max_items=5)

class URL(BaseModel):
    value: str

class Email(BaseModel):
    subject: str
    body: str

class AIResponse(BaseModel):
    response_type: ResponseType
    content: Union[Score, Reasons, URL, Email]


def talk_fast(messages, model="gpt-4o-mini", max_tokens=1000, client=None, image_data=None):
    import tiktoken  # Ensure this package is installed: pip install tiktoken

    if client is None:
        client = default_openai_client
    
    content = []
    if isinstance(messages, str):
        content.append({"type": "text", "text": messages})
    elif isinstance(messages, list):
        content.extend(messages)
    else:
        raise ValueError("Messages should be a string or a list of message objects")

    if image_data:
        if isinstance(image_data, list):
            for img in image_data:
                base64_image = base64.b64encode(img).decode('utf-8')
                content.append({
                    "type": "image",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
        else:
            base64_image = base64.b64encode(image_data).decode('utf-8')
            content.append({
                "type": "image",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })

    # Estimate token count
    encoding = tiktoken.encoding_for_model(model)
    content_text = ''
    for item in content:
        if item['type'] == 'text':
            content_text += item['text']
    input_tokens = len(encoding.encode(content_text))

    # Define the model's context window
    model_context_windows = {
        "gpt-4": 8192,
        "gpt-4o-mini": 4096  # Adjust according to the actual context window
    }
    context_window = model_context_windows.get(model, 4096)

    # Set default max_tokens if not provided
    if max_tokens is None:
        max_tokens = 1000  # Default value

    # Ensure total tokens do not exceed context window
    if input_tokens + max_tokens > context_window:
        max_tokens = context_window - input_tokens - 1  # Reserve 1 token for safety

        # Ensure max_tokens is positive
        if max_tokens <= 0:
            logging.error("Input text is too long for the model to process.")
            return None  # Or handle as needed

    try:
        logging.debug(f"Sending request to OpenAI API with model: {model}")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            tools=[{
                "type": "function",
                "function": {
                    "name": "AIResponse",
                    "parameters": AIResponse.model_json_schema()
                }
            }]
        )
        logging.debug(f"Received response from OpenAI API: {response}")
        
        if response.choices and response.choices[0].message:
            if response.choices[0].message.content:
                result = response.choices[0].message.content.strip()
            elif response.choices[0].message.tool_calls:
                # Handle tool calls (function calls)
                tool_call = response.choices[0].message.tool_calls[0]
                result = tool_call.function.arguments
            else:
                raise ValueError("Unexpected response format")
            
            logging.debug(f"Extracted content from OpenAI response: {result}")
            try:
                parsed_result = json5.loads(result)
                return parsed_result
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing JSON response: {str(e)}")
                return None
        else:
            logging.error("Empty or invalid response from OpenAI API")
            logging.debug(f"Full response object: {response}")
            return None
    except Exception as e:
        error_message = str(e)
        if hasattr(e, 'response'):
            error_message += f"\nResponse content: {e.response.text}"
        logging.error(f"Error in talk_fast: {error_message}")
        return None

def rank_job_description(job_desc, client=None):
    criteria = [
        {
            'name': 'Clarity and Specificity',
            'key': 'clarity_specificity',
            'weight': 20,
            'description': 'The job description should clearly outline the responsibilities, required qualifications, and expectations without ambiguity.',
            'factors': [
                'Use of clear and concise language',
                'Detailed list of job responsibilities',
                'Specific qualifications and experience required',
                'Avoidance of vague terms like "sometimes," "maybe," or "as needed"'
            ]
        },
        {
            'name': 'Inclusivity and Bias-Free Language',
            'key': 'inclusivity',
            'weight': 20,
            'description': 'The job description should use inclusive language that encourages applications from a diverse range of candidates.',
            'factors': [
                'Gender-neutral pronouns and job titles',
                'Avoidance of ageist, ableist, or culturally biased language',
                'Inclusion of diversity and inclusion statements'
            ]
        },
        {
            'name': 'Company Culture and Values Description',
            'key': 'company_culture',
            'weight': 15,
            'description': 'The job description should provide insight into the company\'s culture, mission, and values to help candidates assess cultural fit.',
            'factors': [
                'Clear statement of company mission and values',
                'Description of team dynamics and work environment',
                'Emphasis on aspects like innovation, collaboration, or employee development'
            ]
        },
        {
            'name': 'Realistic and Prioritized Qualifications',
            'key': 'realistic_qualifications',
            'weight': 15,
            'description': 'The qualifications section should distinguish between essential and preferred qualifications to avoid deterring qualified candidates.',
            'factors': [
                'Separate lists for mandatory and preferred qualifications',
                'Realistic experience and education requirements',
                'Justification for any stringent requirements'
            ]
        },
        {
            'name': 'Opportunities for Growth and Development',
            'key': 'growth_opportunities',
            'weight': 10,
            'description': 'The job description should mention any opportunities for career advancement, professional development, or training.',
            'factors': [
                'Information on potential career paths within the company',
                'Availability of training programs or educational assistance',
                'Mentorship or leadership development opportunities'
            ]
        },
        {
            'name': 'Compensation and Benefits Transparency',
            'key': 'compensation_transparency',
            'weight': 10,
            'description': 'Providing information on compensation ranges and benefits can attract candidates aligned with what the company offers.',
            'factors': [
                'Inclusion of salary range or compensation package details',
                'Highlighting key benefits (e.g., health insurance, retirement plans)',
                'Mention of unique perks (e.g., remote work options, flexible hours)'
            ]
        },
        {
            'name': 'Search Engine Optimization (SEO)',
            'key': 'seo',
            'weight': 5,
            'description': 'The job description should be optimized with relevant keywords to improve visibility in job searches.',
            'factors': [
                'Use of industry-standard job titles',
                'Inclusion of relevant keywords and phrases'
            ]
        },
        {
            'name': 'Legal Compliance',
            'key': 'legal_compliance',
            'weight': 5,
            'description': 'Ensure the job description complies with employment laws and regulations.',
            'factors': [
                'Compliance with labor laws',
                'Non-discriminatory language',
                'Properly stated equal opportunity statements'
            ]
        }
    ]

    scores = {}
    total_weight = sum(criterion['weight'] for criterion in criteria)
    total_score = 0

    for criterion in criteria:
        prompt = f"""
        Evaluate the job description based on the criterion: "{criterion['name']}".

        Criterion Description:
        {criterion['description']}

        Factors to consider:
        {', '.join(criterion['factors'])}

        Job Description:
        {job_desc}

        Provide your evaluation as an integer score from 0 to 100, where 0 is the lowest and 100 is the highest.
        Only return the integer score, nothing else.
        """

        response = talk_fast(prompt, client=client)
        try:
            if isinstance(response, dict) and 'content' in response and 'value' in response['content']:
                score = response['content']['value']
            else:
                raise ValueError("Unexpected response format")
            
            if 0 <= score <= 100:
                criterion['score'] = score
            else:
                raise ValueError("Score out of range")
        except ValueError as ve:
            logging.error(f"Error parsing score for criterion {criterion['name']}: {ve}")
            criterion['score'] = 0
        except Exception as e:
            logging.error(f"Unexpected error for criterion {criterion['name']}: {e}")
            criterion['score'] = 0

        scores[criterion['key']] = criterion['score']
        weighted_score = (criterion['score'] * criterion['weight']) / 100
        total_score += weighted_score

    overall_score = int((total_score / total_weight) * 100)  # Normalize to 0-100 scale

    # Collect improvement tips
    tips_prompt = f"""
    Based on your evaluation of the job description, provide 3-5 tips for improvement.

    Job Description:
    {job_desc}

    Focus on areas that can be enhanced according to modern best practices.

    Output your response as a JSON array of strings, e.g.:

    [
        "Tip 1",
        "Tip 2",
        "Tip 3"
    ]
    """
    tips_text = talk_fast(tips_prompt, max_tokens=150, client=client)
    try:
        improvement_tips = json5.loads(tips_text)
        if not isinstance(improvement_tips, list):
            raise ValueError("Improvement tips should be a list.")
        # Ensure tips are strings
        improvement_tips = [str(tip) for tip in improvement_tips]
    except Exception as e:
        logging.error(f"Error parsing improvement tips: {str(e)}")
        improvement_tips = []

    result = {
        "scores": scores,
        "overall_score": overall_score,
        "improvement_tips": improvement_tips[:5]  # Limit to 5 tips
    }

    return result

def extract_text_and_image_from_pdf(file_path):
    import pytesseract
    from pdf2image import convert_from_path
    from PyPDF2 import PdfReader

    try:
        text = ""
        resume_images = []

        # Extract text from all pages of the PDF using PyPDF2
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        # Extract images from all pages
        images = convert_from_path(file_path)
        for i, img in enumerate(images):
            # Convert to grayscale and compress image
            img_gray = img.convert('L')
            img_buffer = io.BytesIO()
            img_gray.save(img_buffer, format='JPEG', quality=51)
            img_buffer.seek(0)

            # Add image data to resume_images list
            resume_images.append(img_buffer.getvalue())

            # If text extraction is insufficient, perform OCR
            if not text or len(text.strip()) < 500:
                ocr_text = pytesseract.image_to_string(Image.open(img_buffer))
                text += ocr_text + "\n"

        if not images:
            logging.error(f"No images found in PDF {file_path}")

        return text.strip(), resume_images

    except Exception as e:
        logging.error(f"Error extracting text and image from PDF {file_path}: {str(e)}")
        return "", []

def assess_resume_quality(resume_images, client=None):
    # Ensure resume_images is a list of base64 encoded strings
    if not isinstance(resume_images, list) or not resume_images:
        logging.error("Invalid resume_images format")
        return 0

    # Use only the first image (front page)
    front_page_image = resume_images[0]

    criteria = [
        {
            'name': 'Formatting and Layout',
            'key': 'formatting_layout',
            'weight': 10,
            'description': 'Assess the overall formatting and layout of the resume.',
            'factors': [
                'Consistent font styles and sizes',
                'Proper alignment of text and sections',
                'Effective use of white space to enhance readability',
                'Appropriate margins and spacing'
            ]
        },
        {
            'name': 'Section Organization and Headings',
            'key': 'section_organization',
            'weight': 15,
            'description': 'Evaluate the organization of content into clear sections with appropriate headings.',
            'factors': [
                'Clear and descriptive section headings',
                'Logical sequence of sections (e.g., summary, experience, education)',
                'Use of subheadings where appropriate',
                'Ease of locating key information'
            ]
        },
        {
            'name': 'Clarity and Conciseness of Content',
            'key': 'content_clarity',
            'weight': 25,
            'description': 'Assess the clarity and conciseness of the information presented.',
            'factors': [
                'Use of clear and straightforward language',
                'Concise bullet points',
                'Avoidance of unnecessary jargon or buzzwords',
                'Focus on relevant information'
            ]
        },
        {
            'name': 'Visual Elements and Design',
            'key': 'visual_design',
            'weight': 20,
            'description': 'Evaluate the visual appeal of the resume, including the use of visual elements.',
            'factors': [
                'Appropriate use of color accents',
                'Inclusion of relevant visual elements (e.g., icons, charts)',
                'Consistency in design elements',
                'Professional appearance suitable for the industry'
            ]
        },
        {
            'name': 'Grammar and Spelling',
            'key': 'grammar_spelling',
            'weight': 20,
            'description': 'Assess the resume for grammatical correctness and spelling accuracy.',
            'factors': [
                'Correct grammar usage',
                'Accurate spelling throughout',
                'Proper punctuation',
                'Professional tone and language'
            ]
        },
        {
            'name': 'Length and Completeness',
            'key': 'length_completeness',
            'weight': 10,
            'description': 'Evaluate whether the resume is of appropriate length and includes all necessary sections.',
            'factors': [
                'Resume length appropriate for experience level (typically 1-2 pages)',
                'Inclusion of all relevant sections',
                'Absence of irrelevant or redundant information'
            ]
        }
    ]

    scores = {}
    total_weight = sum(criterion['weight'] for criterion in criteria)
    total_score = 0

    for criterion in criteria:
        prompt = f"""
        Evaluate the resume image based on the criterion: "{criterion['name']}".

        Criterion Description:
        {criterion['description']}

        Factors to consider:
        {', '.join(criterion['factors'])}

        Provide your evaluation as an integer score from 0 to 100, where 0 is the lowest and 100 is the highest.
        Only return the integer score, nothing else.
        """
        response = talk_fast(prompt, max_tokens=200, image_data=front_page_image, client=client)
        try:
            if isinstance(response, dict) and 'content' in response and 'value' in response['content']:
                score = response['content']['value']
            else:
                raise ValueError("Unexpected response format")
            
            if 0 <= score <= 100:
                scores[criterion['key']] = score
            else:
                raise ValueError("Score out of range")
        except Exception as e:
            logging.error(f"Error parsing score for criterion {criterion['name']}: {str(e)}")
            scores[criterion['key']] = 0

        weighted_score = (score * criterion['weight']) / 100
        total_score += weighted_score

    overall_score = int((total_score / total_weight) * 100)  # Normalize to 0-100 scale

    return overall_score

def extract_job_requirements(job_desc, client=None):
    prompt = f"""
    Extract the key requirements from the following job description.

    Job Description:
    {job_desc}

    Provide the output in the following JSON format:
    {{
      "required_experience_years": integer,
      "required_education_level": string,
      "required_skills": [list of strings],
      "optional_skills": [list of strings],
      "certifications_preferred": [list of strings],
      "soft_skills": [list of strings],
      "keywords_to_match": [list of strings],
      "emphasis": {{
        "technical_skills_weight": integer,
        "soft_skills_weight": integer,
        "experience_weight": integer,
        "education_weight": integer,
        "language_proficiency_weight": integer,
        "certifications_weight": integer
      }}
    }}

    Only output valid JSON. 
    You can only speak JSON. You can only output valid JSON. Strictly No explanation, no comments, no intro. No \`\`\`json\`\`\` wrapper.
    """
    response = talk_to_ai(prompt, max_tokens=2000, client=client)
    try:
        if isinstance(response, dict):
            job_requirements = response
        else:
            job_requirements = json5.loads(response)
        return job_requirements
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing job requirements: {str(e)}")
        logging.error(f"Response: {response}")
        return None

import sys  # Make sure this import is at the top of your file
import argparse

def match_resume_to_job(resume_text, job_desc, file_path, resume_images, client=None):
    # Extract job requirements and wait for completion
    job_requirements = extract_job_requirements(job_desc, client)
    if not job_requirements:
        logging.error("Failed to extract job requirements")
        print(colored("Error: Failed to extract job requirements. Exiting program.", 'red'))
        sys.exit(1)  # Exit the script with an error code

    # Check if job_requirements contains expected keys
    if 'emphasis' not in job_requirements:
        logging.error("Job requirements missing 'emphasis' key")
        print(colored("Error: Invalid job requirements format. Exiting program.", 'red'))
        sys.exit(1)  # Exit the script with an error code

    criteria = [
        {
            'name': 'Language Proficiency',
            'key': 'language_proficiency',
            'weight': job_requirements['emphasis'].get('language_proficiency_weight', 5),
            'description': 'Assign points based on the candidate\'s proficiency in languages relevant to the job.',
            'factors': [
                'Proficiency in required languages',
                'Multilingual abilities relevant to the job'
            ]
        },
        {
            'name': 'Education Level',
            'key': 'education_level',
            'weight': job_requirements['emphasis'].get('education_weight', 10),
            'description': 'Assign points based on the candidate\'s highest level of education or equivalent experience.',
            'factors': [
                'Highest education level attained',
                'Relevance of degree to the job',
                'Alternative education paths (certifications, bootcamps, self-learning)'
            ]
        },
        {
            'name': 'Years of Experience',
            'key': 'experience_years',
            'weight': job_requirements['emphasis'].get('experience_weight', 20),
            'description': 'Assign points based on the relevance and quality of experience.',
            'factors': [
                'Total years of relevant experience',
                'Quality and relevance of previous roles',
                'Significant achievements in previous positions'
            ]
        },
        {
            'name': 'Technical Skills',
            'key': 'technical_skills',
            'weight': job_requirements['emphasis'].get('technical_skills_weight', 40),
            'description': 'Assign points for each required and optional skill, considering proficiency level.',
            'factors': [
                'Proficiency in required technical skills',
                'Proficiency in optional technical skills',
                'Transferable skills and learning ability',
                'Keywords matched in resume'
            ]
        },
        {
            'name': 'Certifications',
            'key': 'certifications',
            'weight': job_requirements['emphasis'].get('certifications_weight', 5),
            'description': 'Assign points for each relevant certification.',
            'factors': [
                'Possession of preferred certifications',
                'Equivalent practical experience',
                'Self-learning projects demonstrating expertise'
            ]
        },
        {
            'name': 'Soft Skills',
            'key': 'soft_skills',
            'weight': job_requirements['emphasis'].get('soft_skills_weight', 20),
            'description': 'Assign points for each soft skill demonstrated through examples or achievements.',
            'factors': [
                'Demonstrated soft skills in resume',
                'Examples of teamwork, leadership, problem-solving, etc.'
            ]
        }
    ]

    scores = {}
    total_weight = sum(criterion['weight'] for criterion in criteria)
    
    if total_weight == 0:
        logging.error("Total weight of criteria is zero")
        return {'score': 0, 'match_reasons': "Error: Invalid criteria weights", 'website': ''}

    total_score = 0

    for criterion in criteria:
        prompt = f"""
        Evaluate the candidate's resume based on the criterion: "{criterion['name']}".

        Criterion Description:
        {criterion['description']}

        Factors to consider:
        {', '.join(criterion['factors'])}

        Job Requirements:
        {json.dumps(job_requirements, indent=2)}

        Resume:
        {resume_text}

        Provide your evaluation as an integer score from 0 to 100, where 0 is the lowest and 100 is the highest.
        Only return the integer score, nothing else. No explanation, no comments, no intro. No \`\`\`json\`\`\` wrapper.
        """

        response = talk_fast(prompt, client=client)
        try:
            if isinstance(response, dict) and 'content' in response and 'value' in response['content']:
                score = response['content']['value']
            else:
                raise ValueError("Unexpected response format")
            
            if 0 <= score <= 100:
                criterion['score'] = score
            else:
                raise ValueError("Score out of range")
        except ValueError as ve:
            logging.error(f"Error parsing score for criterion {criterion['name']}: {ve}")
            criterion['score'] = 0
        except Exception as e:
            logging.error(f"Unexpected error for criterion {criterion['name']}: {e}")
            criterion['score'] = 0

        scores[criterion['key']] = criterion['score']
        weighted_score = (criterion['score'] * criterion['weight']) / 100
        total_score += weighted_score

    # Normalize total score to 0 - 100 scale
    final_score = int((total_score / total_weight) * 100)

    # Generate match reasons
    reasons_prompt = f"""
    Based on the evaluation, provide 3-4 key reasons for the match between the candidate's resume and the job requirements.

    Resume:
    {resume_text}

    Job Requirements:
    {json.dumps(job_requirements, indent=2)}

    Provide the reasons in telegraphic English, max 10 words per reason, separated by ' | '.

    Only output the reasons as a single string. No explanation, no comments, no intro. No \`\`\`json\`\`\` wrapper.
    """
    reasons_response = talk_fast(reasons_prompt, max_tokens=100, client=client)
    
    if isinstance(reasons_response, dict) and 'content' in reasons_response:
        match_reasons = reasons_response['content'].get('value', '')
    else:
        logging.error(f"Unexpected format for reasons response: {reasons_response}")
        match_reasons = ''

    # Extract website from resume (simple extraction)
    website = ''
    website_prompt = f"""
    Extract the candidate's personal website URL from the resume if available.

    Resume:
    {resume_text}

    Only output the URL or an empty string.
    You can only speak URL. You can only output valid URL. Strictly No explanation, no comments, no intro. No \`\`\`json\`\`\` wrapper.
    """
    website_response = talk_fast(website_prompt, max_tokens=150, client=client)
    
    if isinstance(website_response, dict) and 'content' in website_response:
        website = website_response['content'].get('value', '')
    else:
        logging.error(f"Unexpected format for website response: {website_response}")
        website = ''

    # Generate email response and subject
    email_prompt = f"""
    Compose a professional email response to the candidate based on their match score.

    Score: {final_score}

    If the score is below 90, politely reject the person. If the score is 90 or above, invite them to the next stage. Use personal details and make it personalized. Omit signature and "best regards". Friendly concise business tone.

    Provide the output in the following JSON format:
    {{
      "email_response": "Email body",
      "subject_response": "Email subject"
    }}

    You can only speak JSON. You can only output valid JSON. Strictly No explanation, no comments, no intro. No \`\`\`json\`\`\` wrapper.
    """
    email_text = talk_to_ai(email_prompt, max_tokens=180, client=client)
    try:
        email_response = json5.loads(email_text)
        email_body = email_response.get('email_response', '')
        email_subject = email_response.get('subject_response', '')
        # Save email to file
        os.makedirs('out', exist_ok=True)
        file_name = f"out/{os.path.splitext(os.path.basename(file_path))[0]}_response.txt"
        with open(file_name, 'w') as f:
            f.write(f"Subject: {email_subject}\n\n{email_body}")
    except ValueError as e:
        logging.error(f"Error parsing email response: {str(e)}")
        logging.error(f"Raw email text: {email_text}")
        email_body = ''
        email_subject = ''

    return {'score': final_score, 'match_reasons': match_reasons, 'website': website}

def get_score_details(score):
    if not isinstance(score, int):
        raise ValueError(f"Invalid score type: {type(score)}. Expected int.")
    
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

# Define font presets
FONT_PRESETS = {
    'sans-serif': "Helvetica, Helvetica-Bold, Helvetica-Oblique, Helvetica-BoldOblique, sans-serif",
    'serif': "Times-Roman, Times-Bold, Times-Italic, Times-BoldItalic, serif",
    'mono': "Courier, Courier-Bold, Courier-Oblique, Courier-BoldOblique, monospace"
}

def unify_format(extracted_data, font_styles, generate_pdf=False):
    resume_text, resume_images = extracted_data
    
    prompt = """
    Given the following raw text extracted from a resume, convert it into a unified format following these guidelines:

Resume Object Model Definition (Markdown):
===
# Full legal name as it appears on official documents or as preferred professionally.        | First and last name; include middle name or initial if commonly used. | Use your professional or legal name.     |
## Specific position or role aimed for, aligned with the job you're applying for to showcase career focus. | Concise title, typically 2-5 words. | Be specific to highlight your career goals. |

Format: Email / Phone / Country / City
| Field      | Description                                                        | Expected Length                | Guidelines                                         |
|------------|--------------------------------------------------------------------|--------------------------------|----------------------------------------------------|
| **Email**  | Professional email address (e.g., name@example.com).               | Standard email format          | Use a professional email; avoid unprofessional addresses. |
| **Phone**  | Primary contact number, including country code if applicable.      | Include country code if applicable | Provide a reliable contact number.                  |
| **Country**| Full country name of current residence.                            | Full country name              | Specify for relocation considerations.             |
| **City**   | Full city name of current residence.                               | Full city name                 | Indicates proximity to job location.               |

## Summary

Format: plain text

| Field      | Description                                                                                                                | Expected Length                | Guidelines                           |
|------------|----------------------------------------------------------------------------------------------------------------------------|--------------------------------|--------------------------------------|
| **Summary**| Brief overview of qualifications and career goals, highlighting key skills, experiences, and achievements aligned with the desired job. | Mention quantifiable data. STAR format, approximately 5-6 sentences or bullet points | Keep it concise and impactful.       |

Format: _skill, skill, skill_   

| Field      | Description                                                                                                                | Expected Length                | Guidelines                           |
|------------|----------------------------------------------------------------------------------------------------------------------------|--------------------------------|--------------------------------------|
| **Skills**| List of skills (1-2 words each), separated by commas. | Mention technical skills, programming languages, frameworks, tools, and any other relevant skills. SCan the original data and find the skills. | 1-2 words each, 6-12 skills      |


## Employment History

**Description**: Chronological list of past employment experiences (**one or more** entries).
Format: Company / Job Title / Location

Start - End Date

Responsibilities (list or description)

| Field            | Description                                                           | Expected Length        | Guidelines                                           |
|------------------|-----------------------------------------------------------------------|------------------------|------------------------------------------------------|
| **Company**      | Name of employer; include brief descriptor if not well-known.         | Full official name     | Provide context for lesser-known companies.          |
| **Job Title**    | Official title held; accurately reflects roles and responsibilities.  | Standard job title     | Use accurate and professional titles.                |
| **Location**     | City, State/Province, Country.                                        | Full location          | Provides context about work environment.             |
| **Start - End Date** | Employment period (e.g., June 2015 - Present).                       | Format as 'Month Year' | Ensure accuracy and consistency in formatting.       |
| **Responsibilities** | Key duties, achievements, contributions (**one or more** bullet points). | ~3-6 bullet points     | Start with action verbs; quantify achievements when possible. |

## Education

**Description**: Academic qualifications and degrees obtained (**one or more** entries).
Format: Institution / Degree / Location

Start - End Date

Description (if any)

| Field            | Description                                                           | Expected Length        | Guidelines                                           |
|------------------|-----------------------------------------------------------------------|------------------------|------------------------------------------------------|
| **Institution**  | Name of educational institution; add location if not widely known.    | Full official name     | Provide context for lesser-known institutions.       |
| **Degree**       | Degree or certification earned; specify field of study.               | Full degree title      | Highlight relevance to desired job.                  |
| **Location**     | City, State/Province, Country.                                        | Full location          | Provides context about institution's setting.        |
| **Start - End Date** | Education period (e.g., August 2004 - May 2008).                     | Format as 'Month Year' | Use consistent formatting.                           |
| **Description**    | Additional information about the education (if any).                  | ~1-2 sentences         | Include if relevant; keep it concise.               |

## Courses (Optional)

**Description**: Relevant courses, certifications, or training programs completed (**one or more** entries).
Format: Course / Platform

Start - End Date

Description (if any)    

| Field            | Description                                                           | Expected Length        | Guidelines                                           |
|------------------|-----------------------------------------------------------------------|------------------------|------------------------------------------------------|
| **Platform**     | Provider or platform name (e.g., Coursera, Udemy).                    | Organization name      | List reputable providers.                            |
| **Title**        | Official course or certification name.                                | Full title             | Use exact title for verification.                    |
| **Start - End Date** | Course period; can omit if not available.                           | Format as 'Month Year' | Include for context if possible.                     |
| **Description**  | Additional information about the course (if any).                    | ~1-2 sentences         | Include if relevant; keep it concise.               |

## Languages

**Description**: Languages known and proficiency levels (**one or more** entries).
Format: Language / Proficiency

| Field            | Description                                | Expected Length    | Guidelines                                   |
|------------------|--------------------------------------------|--------------------|----------------------------------------------|
| **Language**     | Name of the language (e.g., Spanish).      | Full language name | List languages enhancing your profile.       |
| **Proficiency**  | Level of proficiency (e.g., Native, Fluent). | Standard levels    | Use recognized scales like CEFR.             |

## Links (Optional)

**Description**: Online profiles, portfolios, or relevant links (**one or more** entries).
Format: list of links

- [Title](URL)

| Field      | Description                                          | Expected Length | Guidelines                                     |
|------------|------------------------------------------------------|-----------------|------------------------------------------------|
| **Title**  | Descriptive title (e.g., "My GitHub Profile").       | Short phrase    | Make it clear and professional.                |
| **URL**    | Direct hyperlink to the resource.                    | Full URL        | Ensure links are active and professional.      |

## Hobbies (Optional)
Format: list of hobbies

| Field      | Description                          | Expected Length     | Guidelines                                       |
|------------|--------------------------------------|---------------------|--------------------------------------------------|
| **Hobbies**| Personal interests or activities.    | List of 3-5 hobbies | Showcase positive traits; avoid controversial topics. |

## Misc (Optional)
Format: list of misc

| Field      | Description                          | Expected Length     | Guidelines                                       |
|------------|--------------------------------------|---------------------|--------------------------------------------------|
| **Misc**| Any other information.    | List of any other information | Showcase positive traits; avoid controversial topics. |

===

# General Guidelines:

- **Repeatable Sections**: Employment History, Education, Courses, Languages, and Links can contain **one or more** entries.
- **Optional Sections**: Courses, Links, and Hobbies are **optional**. Omit sections not present in the original resume. **Do not add or invent information**.
- **No Invented Information**: The parser must strictly use only the information provided in the original resume. Do not create, infer, or embellish any details.

# Parser Rules:

To convert an original resume into the defined object model, a parser should follow these rules:

1. **Information Extraction**: Extract information exactly as it appears in the original document. Pay attention to details such as names, dates, job titles, and descriptions.

2. **Section Mapping**: Map the content of the resume to the corresponding sections in the object model:
   - **Name**: Extract from the top of the resume or personal details section.
   - **Desired Job Title**: Look for a stated objective or title near the beginning.
   - **Personal Details**: Extract email, phone, country, and city from the contact information.
   - **Summary**: Use the professional summary or objective section.
   - **Employment History**: Identify past job experiences, including company names, job titles, locations, dates, and responsibilities.
   - **Education**: Extract academic qualifications with institution names, degrees, locations, and dates.
   - **Courses**: Include any additional training or certifications listed.
   - **Languages**: Note any languages and proficiency levels mentioned.
   - **Links**: Extract URLs to professional profiles or portfolios.
   - **Hobbies**: Include personal interests if provided.
   - **Misc**: Include any other information if provided.

3. **Consistency and Formatting**:
   - Ensure dates are formatted consistently throughout (e.g., 'Month Year').
   - Use bullet points for lists where applicable.
   - Maintain the order of entries as they appear in the original resume unless a different order enhances clarity.

4. **Accuracy**:
   - Double-check all extracted information for correctness.
   - Preserve the original wording, especially in descriptions and responsibilities, unless minor adjustments are needed for clarity.

5. **Exclusion of Unavailable Information**:
   - If a section or specific detail is not present in the original resume, omit that section or field in the output.
   - Do not fill in default or placeholder values for missing information.

6. **Avoiding Invention or Assumption**:
   - Do not add any information that is not explicitly stated in the original document.
   - Do not infer skills, responsibilities, or qualifications from context or general knowledge.

7. **Enhancements**:
   - Minor rephrasing for grammar or clarity is acceptable but should not alter the original meaning.
   - Do NOT fix typos or grammar mistakes.
   - Quantify achievements where numbers are provided; do not estimate or create figures.

8. **Professional Language**:
   - Ensure all language used is professional and appropriate for a resume.
   - Remove any informal language or slang that may have been present.

9. **Confidentiality**:
   - Handle all personal data with confidentiality.
   - Do not expose sensitive information in the output that was not intended for inclusion.

10. **Validation**:
    - Validate all URLs to ensure they are correctly formatted.
    - Verify that contact information follows standard formats.

11. **Omit Empty Sections**:
    - Omit sections that contain no information from the original resume.

    Raw Resume Text:
~~~
    {resume_text}
~~~

    Please structure the resume information according to the provided format. Only include sections and details that are present in the original text. Do not invent or assume any information. No more then 4000 tokens.
    No intro, no explanations, no comments. 
    Use telegraphic english with no fluff. Keep all the information, do NOT invent data.
    No ```` or ```yaml or ```json or ```json5 or ``` or --- or any other formatting. Just clean text.
You can only speak in clean, concise, Markdown format.     
    """

    unified_resume = talk_to_ai(prompt.format(resume_text=resume_text), max_tokens=4092)
    
    # Create 'out' folder if it doesn't exist
    out_folder = Path('out')
    out_folder.mkdir(exist_ok=True)
    
    # Extract the name from the first line of the unified resume
    first_line = unified_resume.split('\n', 1)[0]
    if first_line.lower().startswith('# '):
        name = first_line[2:].strip()  # Remove '# ' and trim whitespace
    else:
        name = 'Unknown'  # Fallback if name is not found in expected format
    
    # Generate a filename based on the extracted name
    safe_filename = ''.join(c for c in name if c.isalnum() or c in (' ', '.', '_')).rstrip()
    safe_filename = safe_filename[:50]  # Limit filename length
    
    # Save as Markdown
    md_filename = out_folder / f"{safe_filename}_unified.md"
    with open(md_filename, 'w', encoding='utf-8') as md_file:
        md_file.write(unified_resume)
    logging.info(f"Markdown file created: {md_filename}")
    
    if generate_pdf:
        # Convert Markdown to HTML (in memory)
        html_content = markdown.markdown(unified_resume)

        if font_styles.get('serif'):
            font_family = FONT_PRESETS['serif']
        elif font_styles.get('mono'):
            font_family = FONT_PRESETS['mono']
        else:
            font_family = FONT_PRESETS['sans-serif']  # Default to sans-serif

        html_with_style = f"""
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
            * {{
                color: #3A3F53;
                font-family: {font_family};
            }}
            body {{
                font-size: 0.67em;
                letter-spacing: -0.01em;
                line-height: 1.125;
                background-color: #fff;
                padding: 0;
                margin: 0;
            }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """

        # Convert HTML to PDF
        pdf_filename = out_folder / f"{safe_filename}_unified.pdf"
        try:
            HTML(string=html_with_style).write_pdf(pdf_filename)
            logging.info(f"PDF file created: {pdf_filename}")
        except Exception as e:
            logging.error(f"Error creating PDF: {str(e)}")
    
    return unified_resume, resume_images

def worker(args):
    file, job_desc, font_styles, generate_pdf = args
    try:
        extracted_data = extract_text_and_image_from_pdf(file)
        unified_resume, resume_images = unify_format(extracted_data, font_styles, generate_pdf)
        
        if not unified_resume:
            return (os.path.basename(file), 0, "🔴", "red", "Error: Failed to unify resume format", "", "")
        
        result = match_resume_to_job(unified_resume, job_desc, file, resume_images)
        
        # Use json5 to parse the result
        if isinstance(result, str):
            result = json5.loads(result)
        
        score = result.get('score', 0)
        match_reasons = result.get('match_reasons', '')
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
                    
                    # Combine unified_resume and website_text
                    combined_text = f"{unified_resume}\n\nWebsite Content:\n{website_text}"
                    
                    # Re-run match_resume_to_job with combined_text
                    result = match_resume_to_job(combined_text, job_desc, file, resume_images)
                    if isinstance(result, str):
                        result = json5.loads(result)
                    score = result.get('score', 0)
                    match_reasons = result.get('match_reasons', '')
                except Exception as e:
                    logging.error(f"Error fetching website content for {file}: {str(e)}")
        
        emoji, color, label = get_score_details(score)
        return (os.path.basename(file), score, emoji, color, label, match_reasons, website)
    except json.JSONDecodeError as je:
        error_msg = f"JSON Decode Error: {str(je)}"
        logging.error(f"Error processing {file}: {error_msg}")
        return (os.path.basename(file), 0, "🔴", "red", error_msg, "", "")
    except Exception as e:
        error_msg = f"Unexpected Error: {str(e)}"
        logging.error(f"Error processing {file}: {error_msg}")
        return (os.path.basename(file), 0, "🔴", "red", error_msg, "", "")

def process_resumes(job_desc, pdf_files, font_styles, generate_pdf):
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(worker, [(file, job_desc, font_styles, generate_pdf) for file in pdf_files]), total=len(pdf_files), desc=f"Processing {len(pdf_files)} resumes"))
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
        suggestions = talk_to_ai(prompt, max_tokens=1000)
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
    parser = argparse.ArgumentParser(description="Resume Matcher")
    parser.add_argument("--sans-serif", action="store_true", help="Use sans-serif font preset")
    parser.add_argument("--serif", action="store_true", help="Use serif font preset")
    parser.add_argument("--mono", action="store_true", help="Use monospace font preset")
    parser.add_argument("--pdf", action="store_true", help="Generate PDF files")
    parser.add_argument("job_desc_file", nargs="?", default="job_description.txt", help="Path to job description file")
    parser.add_argument("pdf_folder", nargs="?", default="src", help="Folder containing PDF resumes")
    
    args = parser.parse_args()

    font_styles = {
        'sans-serif': args.sans_serif,
        'serif': args.serif,
        'mono': args.mono
    }

    choose_api()
    
    # Ensure the job description file exists
    while not os.path.exists(args.job_desc_file):
        print(f"File not found: {args.job_desc_file}")
        args.job_desc_file = input("Enter the path to the job description file: ")

    # Read job description
    with open(args.job_desc_file, 'r') as file:
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
    pdf_files = glob(os.path.join(args.pdf_folder, "*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {args.pdf_folder}")
        sys.exit(1)

    logging.info(f"Found {len(pdf_files)} PDF files in {args.pdf_folder}")
    logging.info("Starting resume processing...")

    results = process_resumes(job_desc, pdf_files, font_styles, args.pdf)
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