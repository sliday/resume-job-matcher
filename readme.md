# Resume Job Matcher

## Overview

**Resume Job Matcher** is a Python script that automates the process of matching resumes to a job description using AI. It leverages **OpenRouter** as a unified API gateway to access multiple LLM providers (OpenAI, Anthropic, Google, and more) to analyze resumes and provide a match score along with personalized email responses for candidates.

![Area](https://github.com/user-attachments/assets/1fee4382-7462-4463-9cb1-61704eea218b)

## 🚀 Why OpenRouter?

We've migrated to **OpenRouter** as our unified LLM gateway, providing significant advantages:

- **🔑 Single API Key** - One key for all providers (OpenAI, Anthropic, Google, etc.)
- **💰 Cost Optimization** - Choose models based on your budget and requirements
- **🛡️ Reliability** - Automatic failover between providers
- **🎯 100+ Models** - Access to latest models from multiple providers
- **📊 Unified Billing** - Single dashboard for usage tracking and billing
- **🔧 Easy Migration** - Full backward compatibility with existing code
- **⚡ Performance** - Smart routing and caching for optimal speed

## Features

- 🔥 **Comprehensive Resume Processing**
  - Multiple outputs: PDF and Markdown generation
  - Standardization for fair evaluation
  - Font customization (sans-serif, serif, monospace)
  - Command-line options for flexibility

- 🧠 **Advanced AI-Powered Analysis**
  - Resume-job comparison using OpenRouter unified API
  - Multi-provider support (OpenAI, Anthropic, Google, DeepSeek)
  - Interactive model selection from 100+ available models
  - Unified API interface with automatic failover
  - Structured data handling with Pydantic

- 📊 **In-depth Evaluation & Scoring**
  - Smart parsing with PyPDF2
  - Multi-factor assessment: skills, experience, education, certifications
  - Visual and content-based quality assessment
  - 🚩 Red flag detection in critical areas
  ![CleanShot 2024-10-09 at 17 08 09@2x](https://github.com/user-attachments/assets/e47b57e1-521a-4b21-aeb3-975af1e0f2ed)
  - Detailed scoring with emoji and color-coded results

- 📈 **Comprehensive Analytics & Reporting**
  - Statistical insights: top, average, median, standard deviation scores
  - Candidate distribution summary
  - Match analysis with improvement suggestions
  - Job description optimization recommendations

- 🌐 **Enhanced Candidate Profiling**
  - Website integration for improved matching
  - Personalized email generation

- 🛠️ **Robust System Management**
  - Advanced logging and error handling
  - Improved user feedback and reliability

![CleanShot 2024-09-23 at 23 02 45@2x](https://github.com/user-attachments/assets/bc789343-839e-44bc-b3fb-df3cedf869a8)

## 🔄 OpenRouter Integration

The Resume Job Matcher now uses OpenRouter as a unified API gateway, providing:

- **Seamless Model Switching**: Switch between GPT-4, Claude, Gemini, and other models with a single interface
- **Cost-Effective Processing**: Use expensive models only when needed, cheaper models for bulk processing
- **Automatic Failover**: If one provider is down, automatically switch to another
- **Real-time Model Selection**: Choose your preferred model interactively when running the script

## Usage

### Quick Start

1. **Get OpenRouter API Key**:
   - Sign up at [OpenRouter.ai](https://openrouter.ai)
   - Get your API key from the dashboard
   - Add credits to your account

2. **Configure Environment**:
   ```bash
   # Copy example environment file
   cp .env.example .env
   
   # Edit .env and add your OpenRouter API key
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```

3. **Run the Script**:
   ```bash
   python resume_matcher.py [--sans-serif|--serif|--mono] [--pdf] [job_desc_file] [pdf_folder]
   ```

### Command Line Options

- Use `--sans-serif`, `--serif`, or `--mono` to select a font preset.
- Use `--pdf` to generate PDF versions of unified resumes.
- Optionally specify custom paths for the job description file and PDF folder.

### Model Selection

When you run the script, you'll see an interactive model selection menu:

```
Available models:
OpenAI Models:
  1. GPT-4o (default)
  2. GPT-4o Mini (fast)
  3. GPT-4

Anthropic Models:
  4. Claude 3.5 Sonnet
  5. Claude 3 Opus
  6. Claude 3 Haiku

Other Models:
  7. Gemini Pro 1.5
  8. DeepSeek Chat

Choose a model (1-8) or press Enter for default:
```

## Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Required - OpenRouter API Key
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional - Model Configuration
DEFAULT_MODEL=openai/gpt-4o           # Main processing model
FAST_MODEL=openai/gpt-4o-mini         # Quick tasks model
DEFAULT_MAX_TOKENS=4000               # Token limit
GPT_4O_CONTEXT_WINDOW=128000          # Context window size
```

### Available Models

- **OpenAI**: `openai/gpt-4o`, `openai/gpt-4o-mini`, `openai/gpt-4`
- **Anthropic**: `anthropic/claude-3.5-sonnet`, `anthropic/claude-3-opus`, `anthropic/claude-3-haiku`
- **Google**: `google/gemini-pro-1.5`
- **Other**: `deepseek/deepseek-chat`

### Logging Level

Modify the logging level at the beginning of the script:

```python
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
```

Available levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.

### Programmatic Model Selection

To set a specific model programmatically:

```python
from resume_matcher import talk_to_ai

# Use a specific model
response = talk_to_ai(
    "Your prompt here",
    model="anthropic/claude-3.5-sonnet",
    max_tokens=1000
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
- **API Key Errors**: Verify that the `OPENROUTER_API_KEY` environment variable is set correctly.
- **Model Access Issues**: Some models may require approval or credits in your OpenRouter account.
- **Dependency Errors**: Install all required Python packages using `pip install -r requirements.txt`.

### Testing Your Setup

Run the test script to verify your configuration:

```bash
python test_openrouter.py
```

This will test:
- API connection to OpenRouter
- Model availability
- Legacy function compatibility

### Adjusting Timeouts and Retries

If you experience network-related errors when fetching personal websites, you may adjust the `timeout` parameter in the `check_website` function.

```python
response = requests.get(url, timeout=10)
```

## Best Practices

- **Data Privacy**: Ensure that all candidate data is handled in compliance with relevant data protection laws and regulations.
- **API Usage**: Be mindful of API rate limits and usage policies when using OpenRouter. Monitor your usage through the OpenRouter dashboard.

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork the Repository**: Create your own fork on GitHub.
2. **Create a Feature Branch**: Work on your feature or fix in a new branch.
3. **Submit a Pull Request**: Once your changes are ready, submit a pull request for review.

## Acknowledgments

- **OpenRouter**: For providing unified access to multiple LLM providers
- **OpenAI**: For GPT models
- **Anthropic**: For Claude models
- **Google**: For Gemini models

---

Enjoy using the Resume Job Matcher script to streamline your recruitment process!

## Python Packages

The following Python packages are required for this project:

- PyPDF2: For extracting text from PDF resumes
- openai: To interact with the OpenRouter API (OpenAI-compatible)
- tqdm: For displaying progress bars during processing
- termcolor: To add colored output in the console
- json5: For parsing JSON-like data with added flexibility
- requests: To make HTTP requests for fetching website content
- beautifulsoup4: For parsing HTML content from personal websites
- python-dotenv: For loading environment variables from .env files
- pydantic: For data validation and settings management using Python type annotations

To install these packages, you can use pip:

```bash
pip install -r requirements.txt
```

## Migration from Previous Versions

If you're upgrading from a previous version that used separate OpenAI/Anthropic APIs:

1. **Read the Migration Guide**: See `MIGRATION.md` for detailed instructions
2. **Update Dependencies**: Run `pip install -r requirements.txt`
3. **Get OpenRouter API Key**: Sign up at [OpenRouter.ai](https://openrouter.ai)
4. **Update Environment**: Copy `.env.example` to `.env` and add your OpenRouter API key
5. **Test Migration**: Run `python test_openrouter.py`

The migration maintains full backward compatibility with existing code.

## Star History

<a href="https://star-history.com/#sliday/resume-job-matcher&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=sliday/resume-job-matcher&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=sliday/resume-job-matcher&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=sliday/resume-job-matcher&type=Date" />
 </picture>
</a>
