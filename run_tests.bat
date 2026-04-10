@echo off
REM Run all tests for FPAD project

echo ======================================
echo FPAD - Running Test Suite
echo ======================================

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install test dependencies
echo Installing test dependencies...
pip install pytest pytest-cov pytest-xdist numpy opencv-python pandas scikit-learn scipy -q

REM Run tests
echo.
echo Running tests...
echo.

REM Run with coverage if pytest-cov is available
python -c "import pytest_cov" 2>nul
if %errorlevel% equ 0 (
    pytest tests/ ^
        --cov=preprocessing ^
        --cov=feature_extraction ^
        --cov=feature_engineering ^
        --cov=feature_extractor ^
        --cov=train ^
        --cov-report=term-missing ^
        --cov-report=html:coverage_report ^
        -v
) else (
    pytest tests/ -v
)

REM Capture exit code
set EXIT_CODE=%errorlevel%

echo.
echo ======================================
if %EXIT_CODE% equ 0 (
    echo All tests passed!
) else (
    echo Some tests failed. Exit code: %EXIT_CODE%
)
echo ======================================

exit /b %EXIT_CODE%
