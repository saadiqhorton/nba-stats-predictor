#!/bin/bash

# Streamlit App Runner for NBA Stats Predictor
# Uses Python 3.12 for consistency with test environment

set -e  # Exit on error

echo "🏀 NBA Stats Predictor - Streamlit App"
echo "======================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment with Python 3.12..."
    python3.12 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip3 install -q --upgrade pip
pip3 install -q -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""

# Run Streamlit app
echo "🚀 Starting Streamlit app..."
echo "   Open your browser to: http://localhost:8501"
echo ""

streamlit run app.py

# Deactivate virtual environment
deactivate
