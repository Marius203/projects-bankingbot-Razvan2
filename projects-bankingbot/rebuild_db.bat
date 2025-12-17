@echo off
echo ========================================
echo Rebuilding Vector Database
echo ========================================
echo.
echo This will delete the existing database and rebuild it from PDFs.
echo Press Ctrl+C to cancel, or any key to continue...
pause >nul

echo.
echo Deleting old database...
if exist "chroma_db" (
    rmdir /s /q chroma_db
    echo [OK] Old database deleted
)

echo.
echo Rebuilding database from PDFs...
echo This may take several minutes...
echo.

venv\Scripts\python.exe Loader.py

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to rebuild database!
    pause
    exit /b 1
)

echo.
echo ========================================
echo [SUCCESS] Database rebuilt successfully!
echo ========================================
pause
