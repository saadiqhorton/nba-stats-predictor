#!/bin/bash

# Test runner script for NBA Stats Predictor
# This script sets up the environment and runs the tests

set -e  # Exit on error

echo "🧪 NBA Stats Predictor - Test Runner"
echo "===================================="
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
pip3 install -q -r requirements-dev.txt

echo ""
echo "✅ Setup complete!"
echo ""

# Run tests
echo "🚀 Running tests..."
echo ""

pytest "$@"

# Deactivate virtual environment
deactivate
