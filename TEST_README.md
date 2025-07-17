# Test Suite Documentation

## Overview
This project includes comprehensive test coverage for the Resume Job Matcher application.

## Test Files

### 1. `tests.py` - Comprehensive Test Suite
Full test suite covering all major functions of the resume matcher. Note: This requires all dependencies including system libraries for WeasyPrint.

### 2. `test_core_functions.py` - Core Functionality Tests
Isolated tests that verify core logic without requiring problematic system dependencies. This is the recommended test file for quick verification.

## Running Tests

### Prerequisites
```bash
# Install dependencies using uv
uv sync --all-extras
```

### Running Core Tests (Recommended)
```bash
# Run core functionality tests
uv run pytest test_core_functions.py -v

# Run with short traceback
uv run pytest test_core_functions.py -v --tb=short
```

### Running Full Test Suite
```bash
# Note: Requires system dependencies for WeasyPrint
# On macOS: brew install gobject-introspection
# On Ubuntu: apt-get install libgobject-2.0-0

uv run pytest tests.py -v
```

## Test Coverage

The test suite covers:
- BaseMessage class functionality
- API interaction functions (Anthropic & OpenAI)
- PDF text extraction logic
- Resume quality assessment
- Job requirements extraction
- Resume-job matching algorithm
- Score calculation and details
- Website checking functionality
- Format unification
- Job description ranking
- Multi-processing worker functions
- Environment setup

## Test Status
✅ All 19 core tests passing
✅ No deprecation warnings in main code
✅ Comprehensive coverage of key functionality