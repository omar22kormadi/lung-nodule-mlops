@echo off
echo ========================================
echo  Lung Nodule AI - Backend API Server
echo ========================================
echo.
echo Starting FastAPI server on http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

cd /d "%~dp0"
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000

pause
