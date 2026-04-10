#!/bin/bash
# Run all tests for FPAD project

echo "======================================"
echo "FPAD - Running Test Suite"
echo "======================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

# Install test dependencies
echo "Installing test dependencies..."
pip install pytest pytest-cov pytest-xdist numpy opencv-python pandas scikit-learn scipy -q

# Run tests
echo ""
echo "Running tests..."
echo ""

# Run with coverage if pytest-cov is available
if python -c "import pytest_cov" 2>/dev/null; then
    pytest tests/ \
        --cov=preprocessing \
        --cov=feature_extraction \
        --cov=feature_engineering \
        --cov=feature_extractor \
        --cov=train \
        --cov-report=term-missing \
        --cov-report=html:coverage_report \
        -v
else
    pytest tests/ -v
fi

# Capture exit code
EXIT_CODE=$?

echo ""
echo "======================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "All tests passed!"
else
    echo "Some tests failed. Exit code: $EXIT_CODE"
fi
echo "======================================"

exit $EXIT_CODE
