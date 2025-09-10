import unittest
from unittest.mock import patch, MagicMock
import os
import json
from pathlib import Path

# Make sure the script can be imported
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resume_matcher import (
    unify_format,
    match_resume_to_job,
    assess_resume_quality,
)

class TestResumeMatcher(unittest.TestCase):

    def setUp(self):
        # Create a dummy 'out' directory for generated files
        self.out_dir = Path('out')
        self.out_dir.mkdir(exist_ok=True)

    def tearDown(self):
        # Clean up the 'out' directory
        for f in self.out_dir.glob('*'):
            f.unlink()
        if self.out_dir.exists():
            self.out_dir.rmdir()

    @patch('resume_matcher.talk_to_ai')
    @patch('resume_matcher.HTML')
    def test_unify_format(self, mock_resume_matcher_html, mock_talk_to_ai):
        # Arrange
        mock_talk_to_ai.return_value = "# John Doe\n\n## Summary\n\nA great developer."
        mock_html_instance = MagicMock()
        mock_resume_matcher_html.return_value = mock_html_instance

        extracted_data = ("some resume text", [])
        font_styles = {'sans-serif': True}
        file_path = "resumes/john_doe.pdf"

        # Act
        unify_format(extracted_data, font_styles, file_path, generate_pdf=True)

        # Assert
        # Check that the markdown file was created
        md_file = self.out_dir / "john_doe_unified.md"
        self.assertTrue(md_file.exists())
        with open(md_file, 'r') as f:
            content = f.read()
            self.assertIn("# John Doe", content)

        # Check that the PDF was created
        pdf_file = self.out_dir / "john_doe_unified.pdf"
        mock_resume_matcher_html.assert_called_once()
        mock_html_instance.write_pdf.assert_called_with(pdf_file)


    @patch('resume_matcher.talk_to_ai')
    @patch('resume_matcher.talk_fast')
    @patch('resume_matcher.extract_job_requirements')
    def test_match_resume_to_job_red_flags(self, mock_extract_job_requirements, mock_talk_fast, mock_talk_to_ai):
        # Arrange
        mock_extract_job_requirements.return_value = {
            "emphasis": {
                "technical_skills_weight": 50, # high
                "experience_weight": 25, # medium
                "education_weight": 5, # low
                "location_weight": 35, # high
                "language_proficiency_weight": 5, # low
                "certifications_weight": 5, # low
                "soft_skills_weight": 9, # low
            }
        }
        mock_talk_to_ai.return_value = '{"email_response": "body", "subject_response": "subject"}'

        # Test case 1: Low score on high-weight criterion -> 🚩
        # Low score for Technical Skills (idx 3) and Location (idx 6)
        mock_talk_fast.side_effect = ['90', '90', '90', '5', '90', '90', '5', 'reasons', 'website.com']
        result = match_resume_to_job("resume_text", "job_desc", "file_path", [])
        self.assertIn('Technical Skills', result['red_flags']['🚩'])
        self.assertIn('Location', result['red_flags']['🚩'])

        # Test case 2: Low score on medium-weight criterion -> 📍
        # Low score for Years of Experience (idx 2)
        mock_talk_fast.side_effect = ['90', '90', '5', '90', '90', '90', '90', 'reasons', 'website.com']
        result = match_resume_to_job("resume_text", "job_desc", "file_path", [])
        self.assertIn('Years of Experience', result['red_flags']['📍'])

        # Test case 3: Low score on low-weight criterion -> ⛳
        # Low score for Education Level (idx 1)
        mock_talk_fast.side_effect = ['90', '5', '90', '90', '90', '90', '90', 'reasons', 'website.com']
        result = match_resume_to_job("resume_text", "job_desc", "file_path", [])
        self.assertIn('Education Level', result['red_flags']['⛳'])

        # Test case 4: High scores -> no red flags
        mock_talk_fast.side_effect = ['90', '90', '90', '90', '90', '90', '90', 'reasons', 'website.com']
        result = match_resume_to_job("resume_text", "job_desc", "file_path", [])
        self.assertEqual(len(result['red_flags']['🚩']), 0)
        self.assertEqual(len(result['red_flags']['📍']), 0)
        self.assertEqual(len(result['red_flags']['⛳']), 0)

    @patch('resume_matcher.talk_fast')
    def test_assess_resume_quality(self, mock_talk_fast):
        # Arrange
        # Mocking 6 criteria calls
        mock_talk_fast.side_effect = ['80', '70', '90', '85', '95', '75']
        resume_images = [b"dummy_image_data"]

        # Act
        score = assess_resume_quality(resume_images)

        # Assert
        # (10*80 + 15*70 + 25*90 + 20*85 + 20*95 + 10*75) / 100 = 84.5 -> 84
        self.assertEqual(score, 84)

        # Test invalid score
        mock_talk_fast.side_effect = ['80', 'not a score', '90', '85', '95', '75']
        score = assess_resume_quality(resume_images)
        # (10*80 + 15*0 + 25*90 + 20*85 + 20*95 + 10*75) / 100 = 74 -> 74
        self.assertEqual(score, 74)

if __name__ == '__main__':
    unittest.main()
