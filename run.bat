@echo off
echo ============================================
echo  NBA Prophet Pro - Local Launcher
echo ============================================
echo.

:: Check Python is installed
py --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.11+ from python.org
    pause
    exit /b 1
)

:: Install dependencies if not already installed
echo Checking dependencies...
py -m pip install -r requirements.txt --quiet

:: Create cache directory
if not exist cache mkdir cache

:: Launch the app
echo.
echo Starting app... (browser will open automatically)
echo Press Ctrl+C to stop.
echo.
py -m streamlit run main.py --server.headless false --server.port 8501 --browser.gatherUsageStats false

pause
