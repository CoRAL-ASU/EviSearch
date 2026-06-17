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

# Try to load local env for Vertex config
if [ -f ".env" ]; then
    echo "Loading configuration from .env file..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check Vertex AI auth setup
if [[ -n "${VERTEX_API_KEY}" ]]; then
    echo "Using local Vertex API key auth"
elif [[ -n "${GOOGLE_CLOUD_PROJECT}" ]]; then
    echo "Using Vertex ADC/service-account auth"
else
    echo "❌ Vertex AI is not configured. Set one of:"
    echo "   export VERTEX_API_KEY='your_vertex_api_key_here'   # local development"
    echo "   export GOOGLE_CLOUD_PROJECT='your-project-id'      # ADC / deployed auth"
    echo "   export GOOGLE_CLOUD_LOCATION='us-central1'"
    exit 1
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
