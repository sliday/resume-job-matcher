# Resume Job Matcher

## Overview

**Resume Job Matcher** is a Python script that automates the process of matching resumes to a job description using AI. It leverages the Anthropic Claude API to analyze resumes and provide a match score along with personalized email responses for candidates.

This tool is designed to streamline the recruitment process by efficiently processing multiple resumes and highlighting the best candidates based on customizable criteria.

![Area](https://github.com/user-attachments/assets/1fee4382-7462-4463-9cb1-61704eea218b)

## Features

- **Automated Resume Parsing**: Extracts text from PDF resumes using `PyPDF2`.
- **AI-Powered Matching**: Utilizes the Claude API to compare resumes with job descriptions.
- **Advanced Scoring System**: Implements a comprehensive scoring mechanism based on skills, experience, education, certifications, and more.
- **Multiprocessing Support**: Processes resumes in parallel using all available CPU cores.
- **Personalized Communication**: Generates professional email responses for candidates.
- **Website Content Integration**: Includes personal website content in the evaluation if provided.
- **Detailed Logging and Error Handling**: Provides robust logging and gracefully handles exceptions.

![CleanShot 2024-09-23 at 23 02 45@2x](https://github.com/user-attachments/assets/bc789343-839e-44bc-b3fb-df3cedf869a8)

## Requirements

- **Python**: 3.6 or higher
- **APIs**:
  - [Anthropic Claude API](https://www.anthropic.com/product)
- **Python Packages**:
  - `PyPDF2`
  - `anthropic`
  - `tqdm`
  - `termcolor`
  - `json5`
  - `requests`
  - `beautifulsoup4`

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/sliday/resume-job-matcher.git
cd resume-job-matcher
```

### 2. Set Up a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use 'venv\Scripts\activate'
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not provided, install dependencies manually:

```bash
pip install PyPDF2 anthropic tqdm termcolor json5 requests beautifulsoup4
```

### 4. Set Environment Variables

Set your Claude API key as an environment variable:

**On Linux/macOS:**

```bash
export CLAUDE_API_KEY=your_api_key
```

**On Windows:**

```cmd
set CLAUDE_API_KEY=your_api_key
```

Replace `your_api_key` with your actual Claude API key.

## Usage

### 1. Prepare the Job Description

Create a file named `job_description.txt` in the project directory containing the job description. This file should include:

- Job title
- Required skills
- Optional skills
- Preferred certifications
- Soft skills
- Required education level
- Required years of experience
- Emphasis weights for different criteria

**Example `job_description.txt`:**

```
Senior Machine Learning Engineer

We are looking for a senior machine learning engineer with at least 5 years of experience in machine learning, data analysis, and software development.

Required Skills:
- Python
- Machine Learning
- Data Analysis
- Deep Learning

Optional Skills:
- Cloud Computing
- Big Data
- Computer Vision

Certifications Preferred:
- AWS Certified Solutions Architect
- Certified Data Scientist

Soft Skills:
- Communication
- Team Leadership
- Problem Solving
- Adaptability

Required Education Level:
- Masters

Required Experience Years:
- 5

Emphasis:
- Technical Skills Weight: 50
- Soft Skills Weight: 20
- Experience Weight: 20
- Education Weight: 10
```

### 2. Place Resumes in the Source Directory

Create a directory named `src` (if it doesn't exist) and place all PDF resumes you want to process inside this directory.

### 3. Run the Script

Execute the script using the following command:

```bash
python resume_matcher.py [path_to_job_description] [path_to_resumes]
```

- **`path_to_job_description`**: (Optional) Path to the job description file. Default is `job_description.txt`.
- **`path_to_resumes`**: (Optional) Path to the directory containing resume PDFs. Default is `src`.

**Example:**

```bash
python resume_matcher.py job_description.txt src
```

If you don't provide any arguments, the script will default to `job_description.txt` and `src`.

### 4. View Results

The script will process the resumes and display the results in the terminal, including:

- Candidate name (from the resume filename)
- Match score with an emoji and label
- Key reasons for the match (if score > 80%)
- Any detected personal website status

Personalized email responses are saved in the `out` directory, named after each resume.

## Advanced Features

### Multiprocessing

- Utilizes all available CPU cores to process resumes in parallel, improving efficiency.

### Comprehensive Scoring System

- **Dynamic Weights**: Adjusts scoring weights based on emphasis provided in the job description.
- **Criteria Evaluated**:
  - Language proficiency
  - Education level
  - Years of experience
  - Technical skills (required and optional)
  - Certifications
  - Soft skills
  - Relevance of experience

### Website Content Integration

- **Personal Websites**: If a personal website URL is found in a resume, the script will:
  - Check if the website is active.
  - Fetch and include website content in the evaluation.
  - Penalize the score if the website is inactive.

### Detailed Logging and Error Handling

- **Logging**: Configurable logging levels (`CRITICAL`, `ERROR`, `WARNING`, `INFO`, `DEBUG`).
- **Error Handling**: Gracefully handles exceptions without stopping the entire process.

### Personalized Candidate Communication

- Generates professional, personalized email responses based on the match score.
- Saves responses in the `out` directory for easy access.

## Customization

### Adjust Logging Level

Modify the logging level at the beginning of the script:

```python
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
```

Available levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.

### Change Scoring Model

To change the AI model used, update the `model` parameter in the `match_resume_to_job` function:

```python
message = client.messages.create(
    model="claude-3-sonnet-20240229",
    ...
)
```

### Modify Scoring Criteria

Adjust the scoring logic in the `match_resume_to_job` function's prompt as needed to better fit your specific requirements.

## Output Interpretation

- **Emojis and Labels**: Each candidate is assigned an emoji and label based on their score.
  - Example: `🌟🚀 candidate.pdf : 98% - Cosmic Perfection`

- **Match Reasons**: For candidates scoring above 80%, key reasons for the match are displayed.

- **Website Status**:
  - Active websites are included in the evaluation.
  - Inactive or unreachable websites result in a score penalty and are marked as `(inactive)`.

## Troubleshooting

### Common Issues

- **No Resumes Found**: Ensure that resume PDFs are placed in the correct directory (`src` by default).
- **Job Description Not Found**: Confirm that `job_description.txt` exists in the script's directory or provide the correct path.
- **API Key Errors**: Verify that the `CLAUDE_API_KEY` environment variable is set correctly.
- **Dependency Errors**: Install all required Python packages using `pip`.

### Adjusting Timeouts and Retries

If you experience network-related errors when fetching personal websites, you may adjust the `timeout` parameter in the `check_website` function.

```python
response = requests.get(url, timeout=10)
```

## Best Practices

- **Data Privacy**: Ensure that all candidate data is handled in compliance with relevant data protection laws and regulations.
- **API Usage**: Be mindful of API rate limits and usage policies when using the Anthropic Claude API.

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork the Repository**: Create your own fork on GitHub.
2. **Create a Feature Branch**: Work on your feature or fix in a new branch.
3. **Submit a Pull Request**: Once your changes are ready, submit a pull request for review.

## Acknowledgments

- **Anthropic Claude API**: For providing advanced AI capabilities.

---

Enjoy using the Resume Job Matcher script to streamline your recruitment process!
