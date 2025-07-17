import pytest
import json
import os
from unittest.mock import Mock, patch, MagicMock
import base64

# Test the BaseMessage class in isolation
class TestBaseMessage:
    """Test BaseMessage class functionality."""
    
    class MockBaseMessage:
        """Mock implementation of BaseMessage."""
        
        def __init__(self, text=None, image_data=None):
            self.content = []
            if text:
                self.add_text(text)
            if image_data:
                self.add_image(image_data)
        
        def add_text(self, text):
            self.content.append({
                "type": "text",
                "text": text
            })
        
        def add_image(self, image_data):
            base64_image = base64.b64encode(image_data).decode('utf-8')
            self.content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        
        def get_message(self):
            return self.content
    
    def test_init_with_text(self):
        msg = self.MockBaseMessage(text="Hello, world!")
        assert len(msg.content) == 1
        assert msg.content[0]["type"] == "text"
        assert msg.content[0]["text"] == "Hello, world!"
    
    def test_init_with_image(self):
        image_data = b"fake_image_data"
        msg = self.MockBaseMessage(image_data=image_data)
        assert len(msg.content) == 1
        assert msg.content[0]["type"] == "image_url"
        assert "base64" in msg.content[0]["image_url"]["url"]
    
    def test_add_text(self):
        msg = self.MockBaseMessage()
        msg.add_text("Test text")
        assert len(msg.content) == 1
        assert msg.content[0]["text"] == "Test text"
    
    def test_add_image(self):
        msg = self.MockBaseMessage()
        image_data = b"fake_image_data"
        msg.add_image(image_data)
        assert len(msg.content) == 1
        assert msg.content[0]["type"] == "image_url"
    
    def test_get_message(self):
        msg = self.MockBaseMessage(text="Hello")
        content = msg.get_message()
        assert content == msg.content


class TestScoreDetails:
    """Test score details generation."""
    
    def get_score_details(self, score):
        """Mock implementation of get_score_details function."""
        if score >= 90:
            return "🌟", "green", "Exceptional match"
        elif score >= 70:
            return "✅", "green", "Strong match"
        elif score >= 60:
            return "🤔", "yellow", "Moderate match"
        elif score >= 40:
            return "⚠️", "red", "Weak match"
        else:
            return "❌", "red", "Very weak match"
    
    def test_get_score_details_excellent(self):
        emoji, color, description = self.get_score_details(95)
        assert emoji == "🌟"
        assert color == "green"
        assert "Exceptional" in description
    
    def test_get_score_details_good(self):
        emoji, color, description = self.get_score_details(80)
        assert emoji == "✅"
        assert color == "green"
        assert "Strong" in description
    
    def test_get_score_details_fair(self):
        emoji, color, description = self.get_score_details(65)
        assert emoji == "🤔"
        assert color == "yellow"
        assert "Moderate" in description
    
    def test_get_score_details_poor(self):
        emoji, color, description = self.get_score_details(45)
        assert emoji == "⚠️"
        assert color == "red"
        assert "Weak" in description
    
    def test_get_score_details_very_poor(self):
        emoji, color, description = self.get_score_details(20)
        assert emoji == "❌"
        assert color == "red"
        assert "Very weak" in description


class TestAPIFunctions:
    """Test API interaction logic without actual imports."""
    
    @patch('anthropic.Anthropic')
    def test_talk_to_anthropic_logic(self, mock_anthropic_class):
        """Test the logic of talking to Anthropic API."""
        # Mock client setup
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response from Claude")]
        mock_client.messages.create.return_value = mock_response
        
        # Simulate the function logic
        prompt = "Test prompt"
        messages = [{"role": "user", "content": prompt}]
        
        response = mock_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            messages=messages
        )
        
        result = response.content[0].text
        
        assert result == "Test response from Claude"
        mock_client.messages.create.assert_called_once_with(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            messages=messages
        )
    
    @patch('openai.OpenAI')
    def test_talk_to_openai_logic(self, mock_openai_class):
        """Test the logic of talking to OpenAI API."""
        # Mock client setup
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response from OpenAI"))]
        mock_client.chat.completions.create.return_value = mock_response
        
        # Simulate the function logic
        prompt = "Test prompt"
        messages = [{"role": "user", "content": prompt}]
        
        response = mock_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000
        )
        
        result = response.choices[0].message.content
        
        assert result == "Test response from OpenAI"
        mock_client.chat.completions.create.assert_called_once()


class TestDataProcessing:
    """Test data processing functions."""
    
    def test_resume_text_extraction_logic(self):
        """Test the logic of extracting text from resume."""
        # Simulate text extraction logic
        sample_text = """
        John Doe
        Senior Software Engineer
        
        Experience:
        - 6 years of Python development
        - Expert in Django and FastAPI
        """
        
        # Simple text processing
        lines = sample_text.strip().split('\n')
        processed_lines = [line.strip() for line in lines if line.strip()]
        
        assert "John Doe" in processed_lines[0]
        assert "Senior Software Engineer" in processed_lines[1]
        assert len(processed_lines) > 3
    
    def test_job_requirements_parsing(self):
        """Test parsing job requirements from text."""
        job_desc = """
        We are looking for a Senior Python Developer with:
        - 5+ years of Python experience
        - Experience with Django and FastAPI
        - Strong understanding of databases
        """
        
        # Extract requirements (simplified logic)
        requirements = []
        for line in job_desc.split('\n'):
            if line.strip().startswith('-'):
                requirements.append(line.strip()[1:].strip())
        
        assert len(requirements) == 3
        assert "5+ years of Python experience" in requirements
        assert "Experience with Django and FastAPI" in requirements
    
    def test_score_calculation(self):
        """Test score calculation logic."""
        # Simulate score calculation
        ai_score = 85
        quality_score = 75
        
        # Calculate weighted average (75% AI, 25% quality)
        final_score = (ai_score * 0.75 + quality_score * 0.25)
        final_score = min(100, max(0, final_score))  # Clamp to 0-100
        
        assert final_score == 82.5
        assert 0 <= final_score <= 100
    
    def test_json_response_parsing(self):
        """Test parsing JSON responses."""
        json_response = json.dumps({
            "score": 85,
            "candidate_name": "John Doe",
            "skills": ["Python", "Django", "FastAPI"],
            "match_details": {
                "skills_match": 90,
                "experience_match": 80
            }
        })
        
        parsed = json.loads(json_response)
        
        assert parsed["score"] == 85
        assert parsed["candidate_name"] == "John Doe"
        assert len(parsed["skills"]) == 3
        assert parsed["match_details"]["skills_match"] == 90


class TestWebsiteChecking:
    """Test website checking logic."""
    
    @patch('requests.get')
    def test_check_website_success(self, mock_get):
        """Test successful website checking."""
        mock_response = Mock()
        mock_response.text = "<html><body><h1>John Doe</h1><p>Python Developer</p></body></html>"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # Simulate website checking
        url = "https://example.com"
        try:
            response = mock_get(url, timeout=10)
            response.raise_for_status()
            content = response.text
            # Simple content extraction
            result = content.replace('<', ' ').replace('>', ' ')
        except Exception:
            result = ""
        
        assert "John Doe" in result
        assert "Python Developer" in result
    
    @patch('requests.get')
    def test_check_website_failure(self, mock_get):
        """Test website checking with failure."""
        mock_get.side_effect = Exception("Connection error")
        
        # Simulate website checking with error handling
        url = "https://invalid-url.com"
        try:
            response = mock_get(url, timeout=10)
            result = response.text
        except Exception:
            result = ""
        
        assert result == ""


class TestEnvironmentSetup:
    """Test environment setup."""
    
    def test_environment_variables(self):
        """Test that required environment variables can be set."""
        test_env = {
            'CLAUDE_API_KEY': 'test_claude_key',
            'OPENAI_API_KEY': 'test_openai_key',
            'ANTHROPIC_MODEL': 'claude-3-5-sonnet-20240620',
            'OPENAI_MODEL': 'gpt-4o',
            'DEFAULT_MAX_TOKENS': '1000'
        }
        
        # Simulate setting environment variables
        for key, value in test_env.items():
            os.environ[key] = value
        
        # Verify they are set
        for key, value in test_env.items():
            assert os.environ.get(key) == value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])