# OpenRouter Migration Guide

## 🚀 What's New

The Resume Job Matcher now uses **OpenRouter** as a unified API gateway, providing access to multiple LLM providers through a single interface. This means:

- ✅ **Single API key** for all providers (OpenAI, Anthropic, Google, etc.)
- ✅ **More models** available (100+ models)
- ✅ **Unified billing** and usage tracking
- ✅ **Automatic failover** between providers
- ✅ **Cost optimization** with model selection

## 📋 Migration Steps

### 1. Update Dependencies

```bash
# Install updated dependencies
pip install -r requirements.txt
```

### 2. Get OpenRouter API Key

1. Sign up at [OpenRouter.ai](https://openrouter.ai)
2. Get your API key from the dashboard
3. Add credits to your account

### 3. Update Environment Variables

**Option A: New Setup (Recommended)**
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your OpenRouter API key
OPENROUTER_API_KEY=your_openrouter_api_key_here
DEFAULT_MODEL=openai/gpt-4o
FAST_MODEL=openai/gpt-4o-mini
```

**Option B: Keep Existing Keys (Backward Compatible)**
```bash
# Keep your existing keys, add OpenRouter key
OPENROUTER_API_KEY=your_openrouter_api_key_here
CLAUDE_API_KEY=your_claude_key_here  # Optional fallback
OPENAI_API_KEY=your_openai_key_here  # Optional fallback
```

### 4. Test the Migration

```bash
# Run the test script
python test_openrouter.py
```

## 🎯 Available Models

### OpenAI Models
- `openai/gpt-4o` - Latest GPT-4 model
- `openai/gpt-4o-mini` - Fast and efficient
- `openai/gpt-4` - Classic GPT-4

### Anthropic Models
- `anthropic/claude-3.5-sonnet` - Recommended for most tasks
- `anthropic/claude-3-opus` - Highest capability
- `anthropic/claude-3-haiku` - Fastest response

### Google Models
- `google/gemini-pro-1.5` - Google's latest model

### Other Models
- `deepseek/deepseek-chat` - Cost-effective option

## 🔧 Configuration Options

### Model Selection
When you run the script, you'll see a new interactive model selection menu:

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

### Environment Variables
```bash
# Required
OPENROUTER_API_KEY=your_key_here

# Optional - customize default models
DEFAULT_MODEL=openai/gpt-4o           # Main processing model
FAST_MODEL=openai/gpt-4o-mini         # Quick tasks model
DEFAULT_MAX_TOKENS=4000               # Token limit
GPT_4O_CONTEXT_WINDOW=128000          # Context window size
```

## 🔄 Backward Compatibility

The migration maintains full backward compatibility:

- ✅ All existing functions work (`talk_to_ai`, `talk_to_anthropic`, `talk_to_openai`)
- ✅ Existing environment variables are supported
- ✅ Same command-line interface
- ✅ Same output format

## 💡 Benefits

### Cost Optimization
- Choose cheaper models for simple tasks
- Use premium models only when needed
- Unified billing across all providers

### Reliability
- Automatic failover if one provider is down
- Multiple providers in one integration
- Consistent API across all models

### Flexibility
- Easy to switch between models
- Access to latest models as they're released
- No need to manage multiple API keys

## 🐛 Troubleshooting

### Common Issues

**API Key Not Working**
```bash
# Check your API key
python test_openrouter.py
```

**Model Not Available**
```bash
# Check available models at OpenRouter
# Some models may require approval or credits
```

**Legacy Environment Variables**
```bash
# The script will automatically map legacy variables:
# ANTHROPIC_MODEL -> anthropic/claude-3.5-sonnet
# OPENAI_MODEL -> openai/gpt-4o
# OPENAI_FAST_MODEL -> openai/gpt-4o-mini
```

### Getting Help

1. Check the test script output: `python test_openrouter.py`
2. Review your .env file configuration
3. Verify your OpenRouter account has credits
4. Check OpenRouter dashboard for usage and errors

## 📊 Cost Comparison

OpenRouter provides transparent pricing for all models:
- **OpenAI GPT-4o**: $5/1M input tokens, $15/1M output tokens
- **Claude 3.5 Sonnet**: $3/1M input tokens, $15/1M output tokens
- **DeepSeek Chat**: $0.14/1M input tokens, $0.28/1M output tokens

Choose models based on your budget and quality requirements.

## 🚀 Next Steps

1. **Test the migration** with your existing job descriptions
2. **Experiment with different models** to find the best fit
3. **Monitor usage and costs** through OpenRouter dashboard
4. **Consider model optimization** for different tasks (fast model for preprocessing, premium model for final analysis)

Happy matching! 🎯