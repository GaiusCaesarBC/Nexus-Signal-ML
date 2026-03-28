@echo off
REM setup.bat - Windows Setup Script for ML Service

echo ========================================
echo   NEXUS SIGNAL ML SERVICE SETUP
echo ========================================
echo.

echo [1/4] Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo Virtual environment created
echo.

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat
echo Virtual environment activated
echo.

echo [3/4] Installing Python dependencies...
echo This may take 5-10 minutes...
python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    echo.
    echo Try running: pip install -r requirements.txt
    pause
    exit /b 1
)
echo Dependencies installed
echo.

echo [4/4] Creating necessary directories...
if not exist "models\saved_models" mkdir models\saved_models
if not exist "data" mkdir data
if not exist "logs" mkdir logs
echo Directories created
echo.

echo ========================================
echo   SETUP COMPLETE!
echo ========================================
echo.
echo To start the ML service:
echo   1. Run: venv\Scripts\activate.bat
echo   2. Run: python app.py
echo.
echo The service will be available at http://localhost:5001
echo.
pause