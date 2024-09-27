# Resume Job Matcher

## Overview

**Resume Job Matcher** is a Python script that automates the process of matching resumes to a job description using AI. It leverages the Anthropic Claude API or OpenAI's GPT API to analyze resumes and provide a match score along with personalized email responses for candidates.

![Area](https://github.com/user-attachments/assets/1fee4382-7462-4463-9cb1-61704eea218b)

## Features

- 🔥 **Multiple Outputs**: PDF and Markdown resume generation
- 🆕 **Font Customization**: Sans-serif, serif, monospace options for PDFs
- 🆕 **Resume Standardization**: Consistent format for fair evaluation
- **AI Matching**: Resume-job comparison via Claude/GPT API
- **Resume Parsing**: PDF text extraction with `PyPDF2`
- **Scoring**: Skills, experience, education, certifications evaluation
- **Parallel Processing**: Multi-core efficiency
- **Email Generation**: Personalized candidate responses
- **Website Integration**: Personal site content inclusion
- **Logging/Error Handling**: Robust exception management
- **Interactive Output**: "Resume matching dance" animation
- **Result Summary**: Top, average, bottom scores
- **Dual AI**: Claude and GPT API support with runtime selection
- **Efficient AI Comm**: Fast model interaction
- **Structured Responses**: Pydantic model for AI output

![CleanShot 2024-09-23 at 23 02 45@2x](https://github.com/user-attachments/assets/bc789343-839e-44bc-b3fb-df3cedf869a8)

## Usage

To run the script with the new features:

```bash
python resume_matcher.py [--sans-serif|--serif|--mono] [--pdf] [job_desc_file] [pdf_folder]
```

- Use `--sans-serif`, `--serif`, or `--mono` to select a font preset.
- Use `--pdf` to generate PDF versions of unified resumes.
- Optionally specify custom paths for the job description file and PDF folder.

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
    model="claude-3-5-sonnet-20240620",
    ...
)
```

### Modify AI Provider

To switch between Anthropic and OpenAI APIs, modify the `choose_api` function call at the beginning of the script:

```python
def choose_api():
    global chosen_api
    prompt = "Use OpenAI API instead of Anthropic? [y/N]: "
    choice = input(colored(prompt, "cyan")).strip().lower()
    
    if choice in ["y", "yes"]:
        chosen_api = "openai"
    else:
        chosen_api = "anthropic"
```

### Adjust AI Model

To change the AI model used, update the `model` parameter in the `talk_fast` function:

```python
response = client.chat.completions.create(
    model="gpt-4o",  # Change this to the desired model
    ...
)
```

## Score Calculation

The final score for each resume is calculated using a combination of two factors:

1. **AI-Generated Match Score (75% weight)**: This score is based on how well the resume matches the job description, considering factors such as skills, experience, education, and other relevant criteria.

2. **Resume Quality Score (25% weight)**: This score assesses the visual appeal and clarity of the resume itself, including formatting, layout, and overall presentation.

The calculation process is as follows:

1. The AI-generated match score and the resume quality score are both normalized to a 0-100 scale.
2. A weighted average is calculated: 
   `(AI_Score * 0.75 + Quality_Score * 0.25)`
3. The result is clamped to ensure it falls within the 0-100 range.

This combined approach ensures that both the content relevance and the presentation quality of the resume are taken into account in the final score.

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

## Python Packages

The following Python packages are required for this project:

- PyPDF2: For extracting text from PDF resumes
- anthropic: To interact with the Anthropic Claude API for AI-powered analysis
- tqdm: For displaying progress bars during processing
- termcolor: To add colored output in the console
- json5: For parsing JSON-like data with added flexibility
- requests: To make HTTP requests for fetching website content
- beautifulsoup4: For parsing HTML content from personal websites
- openai: To interact with the OpenAI API for AI-powered analysis
- pydantic: For data validation and settings management using Python type annotations

To install these packages, you can use pip:

```bash
pip install PyPDF2 anthropic openai tqdm termcolor json5 requests beautifulsoup4 pydantic
```

## Star History

<a href="https://star-history.com/#sliday/resume-job-matcher&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=sliday/resume-job-matcher&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=sliday/resume-job-matcher&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=sliday/resume-job-matcher&type=Date" />
 </picture>
</a>
