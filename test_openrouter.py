#!/usr/bin/env python3
"""
Test script for OpenRouter migration.
This script tests the basic functionality of the OpenRouter API integration.
"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()

# Add the current directory to the path to import resume_matcher
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from resume_matcher import talk_to_ai, choose_model, AVAILABLE_MODELS, current_model, DEFAULT_MODEL

def test_api_connection():
    """Test basic API connection to OpenRouter."""
    print("Testing OpenRouter API connection...")
    
    # Test with a simple prompt
    test_prompt = "Hello, please respond with 'OpenRouter connection successful!'"
    
    try:
        response = talk_to_ai(test_prompt, max_tokens=50)
        if response:
            print(f"✅ API connection successful!")
            print(f"Response: {response}")
            return True
        else:
            print("❌ API connection failed - no response received")
            return False
    except Exception as e:
        print(f"❌ API connection failed: {str(e)}")
        return False

def test_model_selection():
    """Test model selection functionality."""
    print(f"\\nTesting model selection...")
    print(f"Current model: {current_model}")
    print(f"Default model: {DEFAULT_MODEL}")
    print(f"Available models: {list(AVAILABLE_MODELS.keys())}")
    
    # Test with different models
    test_models = ['openai/gpt-4o', 'anthropic/claude-3.5-sonnet']
    
    for model in test_models:
        if model.split('/')[0] in ['openai', 'anthropic']:  # Common providers
            print(f"\\nTesting with model: {model}")
            try:
                response = talk_to_ai(
                    "Respond with just the model name you are.",
                    max_tokens=20,
                    model=model
                )
                if response:
                    print(f"✅ Model {model} working: {response}")
                else:
                    print(f"❌ Model {model} failed: no response")
            except Exception as e:
                print(f"❌ Model {model} failed: {str(e)}")

def test_legacy_functions():
    """Test backward compatibility with legacy functions."""
    print(f"\\nTesting legacy function compatibility...")
    
    from resume_matcher import talk_to_anthropic, talk_to_openai
    
    # Test legacy functions
    try:
        response = talk_to_anthropic("Say 'Claude legacy function works'", max_tokens=20)
        if response:
            print(f"✅ Legacy Anthropic function working: {response}")
        else:
            print("❌ Legacy Anthropic function failed")
    except Exception as e:
        print(f"❌ Legacy Anthropic function failed: {str(e)}")
    
    try:
        response = talk_to_openai("Say 'OpenAI legacy function works'", max_tokens=20)
        if response:
            print(f"✅ Legacy OpenAI function working: {response}")
        else:
            print("❌ Legacy OpenAI function failed")
    except Exception as e:
        print(f"❌ Legacy OpenAI function failed: {str(e)}")

def main():
    """Run all tests."""
    print("🚀 OpenRouter Migration Test Suite")
    print("=" * 50)
    
    # Check if API key is set
    if not os.getenv('OPENROUTER_API_KEY'):
        print("❌ OPENROUTER_API_KEY not set!")
        print("Please set your OpenRouter API key in .env file")
        print("Copy .env.example to .env and add your key")
        return
    
    # Run tests
    test_api_connection()
    test_model_selection()
    test_legacy_functions()
    
    print("\\n" + "=" * 50)
    print("✅ Test suite completed!")
    print("\\nTo use the resume matcher:")
    print("1. Set OPENROUTER_API_KEY in .env file")
    print("2. Run: python resume_matcher.py")
    print("3. Choose your preferred model when prompted")

if __name__ == "__main__":
    main()