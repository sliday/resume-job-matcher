import pytest
import json
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import base64
import PyPDF2
from PIL import Image
import io

from resume_matcher import (
    BaseMessage,
    talk_to_ai,
    talk_to_anthropic,
    talk_to_openai,
    talk_fast,
    extract_text_and_image_from_pdf,
    assess_resume_quality,
    extract_job_requirements,
    match_resume_to_job,
    get_score_details,
    check_website,
    unify_format,
    rank_job_description,
    worker,
    process_resumes,
    analyze_overall_matches,
    improve_job_description
)


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing."""
    client = Mock()
    response = Mock()
    response.content = [Mock(text="Test response from Claude")]
    client.messages.create.return_value = response
    return client


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    client = Mock()
    response = Mock()
    response.choices = [Mock(message=Mock(content="Test response from OpenAI"))]
    client.chat.completions.create.return_value = response
    return client


@pytest.fixture
def sample_pdf_path(tmp_path):
    """Create a temporary PDF file for testing."""
    pdf_path = tmp_path / "test_resume.pdf"
    pdf_path.write_text("Test PDF content")
    return str(pdf_path)


@pytest.fixture
def sample_job_desc():
    """Sample job description for testing."""
    return """
    We are looking for a Senior Python Developer with:
    - 5+ years of Python experience
    - Experience with Django and FastAPI
    - Strong understanding of databases
    - Excellent communication skills
    """


@pytest.fixture
def sample_resume_text():
    """Sample resume text for testing."""
    return """
    John Doe
    Senior Software Engineer
    
    Experience:
    - 6 years of Python development
    - Expert in Django and FastAPI
    - PostgreSQL and MongoDB experience
    
    Skills:
    - Python, JavaScript, SQL
    - Docker, Kubernetes
    - Excellent communication and leadership skills
    """


class TestBaseMessage:
    """Test BaseMessage class functionality."""
    
    def test_init_with_text(self):
        msg = BaseMessage(text="Hello, world!")
        assert len(msg.content) == 1
        assert msg.content[0]["type"] == "text"
        assert msg.content[0]["text"] == "Hello, world!"
    
    def test_init_with_image(self):
        image_data = b"fake_image_data"
        msg = BaseMessage(image_data=image_data)
        assert len(msg.content) == 1
        assert msg.content[0]["type"] == "image_url"
        assert "base64" in msg.content[0]["image_url"]["url"]
    
    def test_add_text(self):
        msg = BaseMessage()
        msg.add_text("Test text")
        assert len(msg.content) == 1
        assert msg.content[0]["text"] == "Test text"
    
    def test_add_image(self):
        msg = BaseMessage()
        image_data = b"fake_image_data"
        msg.add_image(image_data)
        assert len(msg.content) == 1
        assert msg.content[0]["type"] == "image_url"
    
    def test_get_message(self):
        msg = BaseMessage(text="Hello")
        content = msg.get_message()
        assert content == msg.content


class TestAPIFunctions:
    """Test API interaction functions."""
    
    @patch('resume_matcher.chosen_api', 'anthropic')
    def test_talk_to_ai_anthropic(self, mock_anthropic_client):
        result = talk_to_ai("Test prompt", client=mock_anthropic_client)
        assert result == "Test response from Claude"
        mock_anthropic_client.messages.create.assert_called_once()
    
    @patch('resume_matcher.chosen_api', 'openai')
    def test_talk_to_ai_openai(self, mock_openai_client):
        result = talk_to_ai("Test prompt", client=mock_openai_client)
        assert result == "Test response from OpenAI"
        mock_openai_client.chat.completions.create.assert_called_once()
    
    def test_talk_to_anthropic(self, mock_anthropic_client):
        result = talk_to_anthropic("Test prompt", client=mock_anthropic_client)
        assert result == "Test response from Claude"
    
    def test_talk_to_anthropic_with_image(self, mock_anthropic_client):
        image_data = b"fake_image_data"
        result = talk_to_anthropic("Test prompt", image_data=image_data, client=mock_anthropic_client)
        assert result == "Test response from Claude"
    
    def test_talk_to_openai(self, mock_openai_client):
        result = talk_to_openai("Test prompt", client=mock_openai_client)
        assert result == "Test response from OpenAI"
    
    def test_talk_to_openai_with_image(self, mock_openai_client):
        image_data = b"fake_image_data"
        result = talk_to_openai("Test prompt", image_data=image_data, client=mock_openai_client)
        assert result == "Test response from OpenAI"
    
    def test_talk_fast(self, mock_openai_client):
        messages = [{"role": "user", "content": "Test"}]
        result = talk_fast(messages, client=mock_openai_client)
        assert result == "Test response from OpenAI"


class TestPDFExtraction:
    """Test PDF extraction functionality."""
    
    @patch('PyPDF2.PdfReader')
    @patch('PIL.Image.open')
    def test_extract_text_and_image_from_pdf(self, mock_image_open, mock_pdf_reader):
        # Mock PDF reader
        mock_page = Mock()
        mock_page.extract_text.return_value = "Test resume content"
        mock_pdf_reader.return_value.pages = [mock_page]
        
        # Mock image extraction
        mock_image = Mock()
        mock_image.width = 100
        mock_image.height = 100
        mock_image_open.return_value = mock_image
        
        text, images = extract_text_and_image_from_pdf("test.pdf")
        
        assert text == "Test resume content"
        assert len(images) == 0  # No images in our mock


class TestResumeQualityAssessment:
    """Test resume quality assessment functionality."""
    
    def test_assess_resume_quality_no_images(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value.content[0].text = json.dumps({
            "visual_appeal_score": 80,
            "clarity_score": 85,
            "overall_quality_score": 82,
            "feedback": "Good resume"
        })
        
        result = assess_resume_quality([], client=mock_anthropic_client)
        assert result["overall_quality_score"] == 82
        assert result["feedback"] == "Good resume"
    
    def test_assess_resume_quality_with_images(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value.content[0].text = json.dumps({
            "visual_appeal_score": 90,
            "clarity_score": 88,
            "overall_quality_score": 89,
            "feedback": "Excellent resume"
        })
        
        fake_images = [b"fake_image_1", b"fake_image_2"]
        result = assess_resume_quality(fake_images, client=mock_anthropic_client)
        assert result["overall_quality_score"] == 89


class TestJobRequirementsExtraction:
    """Test job requirements extraction functionality."""
    
    def test_extract_job_requirements(self, mock_anthropic_client, sample_job_desc):
        mock_anthropic_client.messages.create.return_value.content[0].text = json.dumps({
            "requirements": [
                "5+ years of Python experience",
                "Experience with Django and FastAPI",
                "Strong understanding of databases",
                "Excellent communication skills"
            ],
            "core_skills": ["Python", "Django", "FastAPI", "Databases"],
            "experience_needed": "5+ years",
            "education_level": "Bachelor's degree preferred"
        })
        
        result = extract_job_requirements(sample_job_desc, client=mock_anthropic_client)
        assert len(result["requirements"]) == 4
        assert "Python" in result["core_skills"]


class TestResumeJobMatching:
    """Test resume-job matching functionality."""
    
    def test_match_resume_to_job(self, mock_anthropic_client, sample_resume_text, sample_job_desc):
        mock_anthropic_client.messages.create.return_value.content[0].text = json.dumps({
            "score": 85,
            "match_details": {
                "skills_match": 90,
                "experience_match": 80,
                "education_match": 85,
                "certifications_match": 0,
                "overall_fit": 85
            },
            "strengths": ["Strong Python experience", "Relevant frameworks"],
            "weaknesses": ["No specific certifications mentioned"],
            "red_flags": [],
            "candidate_name": "John Doe",
            "candidate_email": "john.doe@example.com",
            "phone": "+1234567890",
            "linkedin": "linkedin.com/in/johndoe",
            "github": "github.com/johndoe",
            "personal_website": "johndoe.com",
            "location": "San Francisco, CA",
            "years_of_experience": 6,
            "highest_education": "Bachelor's degree",
            "current_position": "Senior Software Engineer",
            "key_skills": ["Python", "Django", "FastAPI", "PostgreSQL"],
            "languages": ["English"],
            "security_clearance": "None",
            "availability": "2 weeks notice",
            "work_authorization": "US Citizen",
            "suggested_email": {
                "subject": "Your Python Developer Application",
                "body": "Dear John..."
            }
        })
        
        result = match_resume_to_job(
            sample_resume_text, 
            sample_job_desc, 
            "test_resume.pdf",
            [],
            client=mock_anthropic_client
        )
        
        assert result["score"] == 85
        assert result["candidate_name"] == "John Doe"
        assert len(result["strengths"]) > 0


class TestScoreDetails:
    """Test score details generation."""
    
    def test_get_score_details_excellent(self):
        emoji, color, description = get_score_details(95)
        assert emoji == "🌟"
        assert color == "green"
        assert "Exceptional" in description
    
    def test_get_score_details_good(self):
        emoji, color, description = get_score_details(80)
        assert emoji == "✅"
        assert color == "green"
        assert "Strong" in description
    
    def test_get_score_details_fair(self):
        emoji, color, description = get_score_details(65)
        assert emoji == "🤔"
        assert color == "yellow"
        assert "Moderate" in description
    
    def test_get_score_details_poor(self):
        emoji, color, description = get_score_details(45)
        assert emoji == "⚠️"
        assert color == "red"
        assert "Weak" in description
    
    def test_get_score_details_very_poor(self):
        emoji, color, description = get_score_details(20)
        assert emoji == "❌"
        assert color == "red"
        assert "Very weak" in description


class TestWebsiteChecking:
    """Test website checking functionality."""
    
    @patch('requests.get')
    def test_check_website_success(self, mock_get):
        mock_response = Mock()
        mock_response.text = "<html><body><h1>John Doe - Portfolio</h1><p>Python Developer</p></body></html>"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = check_website("https://johndoe.com")
        assert "John Doe" in result
        assert "Python Developer" in result
    
    @patch('requests.get')
    def test_check_website_failure(self, mock_get):
        mock_get.side_effect = Exception("Connection error")
        
        result = check_website("https://invalid-url.com")
        assert result == ""


class TestUnifyFormat:
    """Test format unification functionality."""
    
    def test_unify_format_basic(self):
        extracted_data = {
            "candidate_name": "John Doe",
            "candidate_email": "john@example.com",
            "phone": "+1234567890",
            "location": "San Francisco, CA",
            "current_position": "Senior Developer",
            "years_of_experience": 5,
            "highest_education": "Bachelor's degree",
            "key_skills": ["Python", "JavaScript"],
            "languages": ["English", "Spanish"],
            "certifications": ["AWS Certified"],
            "work_experience": "5 years at Tech Corp",
            "education": "BS in Computer Science",
            "projects": "Built several web applications"
        }
        
        font_styles = {
            "font_family": "Arial, sans-serif",
            "heading_font": "Arial, sans-serif"
        }
        
        md_content, _, html_content = unify_format(extracted_data, font_styles, generate_pdf=False)
        
        assert "# John Doe" in md_content
        assert "john@example.com" in md_content
        assert "Python" in md_content
        assert "<h1>" in html_content


class TestJobDescriptionRanking:
    """Test job description ranking functionality."""
    
    def test_rank_job_description(self, mock_anthropic_client, sample_job_desc):
        mock_anthropic_client.messages.create.return_value.content[0].text = json.dumps({
            "clarity_score": 85,
            "completeness_score": 80,
            "appeal_score": 75,
            "overall_score": 80,
            "strengths": ["Clear requirements", "Good skill list"],
            "improvements": ["Add salary range", "Include benefits"],
            "missing_elements": ["Company culture", "Growth opportunities"]
        })
        
        result = rank_job_description(sample_job_desc, client=mock_anthropic_client)
        assert result["overall_score"] == 80
        assert len(result["strengths"]) > 0
        assert len(result["improvements"]) > 0


class TestWorkerFunction:
    """Test the worker function for multiprocessing."""
    
    @patch('resume_matcher.match_resume_to_job')
    @patch('resume_matcher.extract_text_and_image_from_pdf')
    @patch('resume_matcher.unify_format')
    def test_worker(self, mock_unify, mock_extract, mock_match):
        mock_extract.return_value = ("Resume text", [])
        mock_match.return_value = {
            "score": 85,
            "candidate_name": "John Doe",
            "final_score": 85
        }
        mock_unify.return_value = ("# Resume", None, "<html>Resume</html>")
        
        result = worker((
            "Job description",
            "resume.pdf",
            {"font_family": "Arial"},
            False
        ))
        
        assert result[0] == "resume.pdf"
        assert result[1]["score"] == 85


class TestAnalyzeOverallMatches:
    """Test overall match analysis functionality."""
    
    def test_analyze_overall_matches(self, mock_anthropic_client, sample_job_desc):
        results = [
            ("resume1.pdf", {"final_score": 85, "score": 85}),
            ("resume2.pdf", {"final_score": 75, "score": 75}),
            ("resume3.pdf", {"final_score": 65, "score": 65})
        ]
        
        mock_anthropic_client.messages.create.return_value.content[0].text = json.dumps({
            "top_candidates_analysis": "Top candidates show strong skills",
            "common_strengths": ["Python experience", "Good communication"],
            "common_gaps": ["Limited cloud experience"],
            "recommendations": ["Consider cloud training"],
            "job_market_insights": "Competitive market"
        })
        
        with patch('builtins.print'):
            analyze_overall_matches(sample_job_desc, results)
        
        # This function mainly prints output, so we just verify it runs without error


class TestImproveJobDescription:
    """Test job description improvement functionality."""
    
    def test_improve_job_description(self, mock_anthropic_client, sample_job_desc):
        ranking = {
            "improvements": ["Add salary range", "Include benefits"],
            "missing_elements": ["Company culture"]
        }
        
        mock_anthropic_client.messages.create.return_value.content[0].text = """
        Improved job description with:
        - Salary range: $120k-$150k
        - Benefits package details
        - Company culture section
        """
        
        result = improve_job_description(sample_job_desc, ranking, client=mock_anthropic_client)
        assert "Salary range" in result
        assert "Benefits" in result


@pytest.fixture
def setup_env():
    """Set up environment variables for testing."""
    os.environ['CLAUDE_API_KEY'] = 'test_claude_key'
    os.environ['OPENAI_API_KEY'] = 'test_openai_key'
    os.environ['ANTHROPIC_MODEL'] = 'claude-3-5-sonnet-20240620'
    os.environ['OPENAI_MODEL'] = 'gpt-4o'
    os.environ['OPENAI_FAST_MODEL'] = 'gpt-4o-mini'
    os.environ['DEFAULT_MAX_TOKENS'] = '1000'
    os.environ['GPT_4O_CONTEXT_WINDOW'] = '128000'
    yield
    # Cleanup would go here if needed


def test_environment_setup(setup_env):
    """Test that environment variables are properly loaded."""
    assert os.environ.get('CLAUDE_API_KEY') == 'test_claude_key'
    assert os.environ.get('OPENAI_API_KEY') == 'test_openai_key'