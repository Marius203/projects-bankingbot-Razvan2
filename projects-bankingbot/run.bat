@echo off
echo ========================================
echo AI Banking Bot - Startup Script
echo ========================================
echo.

REM Check if venv exists
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found!
    echo Please run: py -3.13 -m venv venv
    pause
    exit /b 1
)

REM Check if Ollama is running
echo [1/4] Checking if Ollama is running...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Ollama doesn't seem to be running!
    echo Please start Ollama and make sure these models are installed:
    echo   - ollama pull llama3.1
    echo   - ollama pull nomic-embed-text
    echo.
    echo Press any key to continue anyway, or Ctrl+C to exit...
    pause >nul
) else (
    echo [OK] Ollama is running
)

REM Check if database exists
echo.
echo [2/4] Checking if vector database exists...
if not exist "chroma_db" (
    echo [WARNING] Vector database not found!
    echo Creating database from PDFs... This may take a few minutes.
    echo.
    venv\Scripts\python.exe Loader.py
    if errorlevel 1 (
        echo [ERROR] Failed to create database!
        pause
        exit /b 1
    )
    echo [OK] Database created successfully
) else (
    echo [OK] Database exists
)

REM Install dependencies if needed
echo.
echo [3/4] Checking dependencies...
venv\Scripts\python.exe -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Flask not installed. Installing dependencies...
    venv\Scripts\pip.exe install flask flask-cors
)
venv\Scripts\python.exe -c "import rank_bm25" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] rank_bm25 not installed. Installing...
    venv\Scripts\pip.exe install rank_bm25
)
echo [OK] All dependencies installed

REM Start Flask server
echo.
echo [4/4] Starting Flask server...
echo.
echo ========================================
echo Server will start at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo ========================================
echo.

venv\Scripts\python.exe app.py
