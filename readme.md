# Resume Job Matcher

## Overview

**Resume Job Matcher** is a Python script that automates the process of matching resumes to a job description using AI. It leverages the Anthropic Claude API to analyze resumes and provide a match score along with personalized email responses for candidates.

This tool is designed to streamline the recruitment process by efficiently processing multiple resumes and highlighting the best candidates based on customizable criteria. The script uses advanced natural language processing to compare resume content with job requirements, considering factors such as skills, experience, education, and even personal website content when available.

![Area](https://github.com/user-attachments/assets/1fee4382-7462-4463-9cb1-61704eea218b)

## Features

- **Automated Resume Parsing**: Extracts text from PDF resumes using `PyPDF2`.
- **AI-Powered Matching**: Utilizes the Claude API to compare resumes with job descriptions.
- **Advanced Scoring System**: Implements a comprehensive scoring mechanism based on skills, experience, education, certifications, and more.
- **Multiprocessing Support**: Processes resumes in parallel using all available CPU cores.
- **Personalized Communication**: Generates professional email responses for candidates.
- **Website Content Integration**: Includes personal website content in the evaluation if provided.
- **Detailed Logging and Error Handling**: Provides robust logging and gracefully handles exceptions.
- **Interactive Console Output**: Displays a fun "resume matching dance" animation during processing.
- **Comprehensive Result Summary**: Provides a detailed summary of top, average, and bottom scores.

![CleanShot 2024-09-23 at 23 02 45@2x](https://github.com/user-attachments/assets/bc789343-839e-44bc-b3fb-df3cedf869a8)

## Scoring System and Output Interpretation

### Scoring Mechanism

The script uses a sophisticated scoring system that considers various factors:

- Match between resume content and job requirements
- Relevance of skills and experience
- Education level
- Years of experience
- Certifications
- Soft skills
- Personal website content (if available)

The AI model analyzes these factors and assigns a score from 0 to 100.

### Output Interpretation

- **Emojis and Labels**: Each candidate is assigned an emoji and label based on their score. For example:
  - `🌟🚀 98% - Cosmic Perfection`
  - `🦄✨ 95% - Unicorn Candidate`
  - `🌠💫 93% - Superstar Plus`
  - ...
  - `☄️ 10% - Cosmic Mismatch`

- **Match Reasons**: For candidates scoring above 80%, key reasons for the match are displayed.

- **Website Status**:
  - Active websites are included in the evaluation.
  - Inactive or unreachable websites result in a score penalty and are marked as `(inactive)`.

### Result Summary

After processing all resumes, the script provides a summary including:

- Top Score
- Average Score
- Bottom Score
- Number of Processed Resumes
- Number of Errors Encountered (if any)

This summary helps recruiters quickly gauge the overall quality of the candidate pool.

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
  - Re-run the matching process with combined resume and website content for a more comprehensive evaluation.

### Detailed Logging and Error Handling

- **Logging**: Configurable logging levels (`CRITICAL`, `ERROR`, `WARNING`, `INFO`, `DEBUG`).
- **Error Handling**: Gracefully handles exceptions without stopping the entire process.

### Personalized Candidate Communication

- Generates professional, personalized email responses based on the match score.
- Saves responses in the `out` directory for easy access.

### Interactive Console Output

The script provides an engaging user experience with a "resume matching dance" animation displayed in the console during processing. This fun feature uses ASCII art and emojis to show progress and keep users entertained while the script processes resumes.

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
