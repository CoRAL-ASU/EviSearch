#!/bin/bash

# Clinical Trial Data Extraction - Web Interface Startup Script

echo "=========================================="
echo "Clinical Trial Data Extraction"
echo "Web Interface Startup"
echo "=========================================="
echo ""

# Check if virtual environment is activated
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "⚠️  Virtual environment not activated!"
    echo "Activating .venv..."
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    else
        echo "❌ Virtual environment not found. Please create it first:"
        echo "   python -m venv .venv"
        echo "   source .venv/bin/activate"
        echo "   pip install -r web/requirements.txt"
        exit 1
    fi
fi

# Check if GEMINI_API_KEY is set
if [[ -z "${GEMINI_API_KEY}" ]]; then
    echo "⚠️  GEMINI_API_KEY not set in environment"
    
    # Try to load from .env file
    if [ -f ".env" ]; then
        echo "Loading from .env file..."
        export $(cat .env | grep -v '^#' | xargs)
    fi
    
    if [[ -z "${GEMINI_API_KEY}" ]]; then
        echo "❌ GEMINI_API_KEY is required. Please set it:"
        echo "   export GEMINI_API_KEY='your_api_key_here'"
        echo "   Or add it to .env file"
        exit 1
    fi
fi

# Check if required dependencies are installed
echo "Checking dependencies..."
python3 -c "import flask; import google.genai; import dotenv" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Some dependencies are missing. Installing..."
    pip install -r web/requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
fi

echo "✅ All checks passed!"
echo ""
echo "Starting web interface..."
echo "Access at: http://127.0.0.1:8007"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

# Start the web application
python3 web/main_app.py
