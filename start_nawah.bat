@echo off
cd /d "%~dp0"
echo [NAWAH] Starting API on http://localhost:8000 ...
python -m uvicorn server:app --reload --port 8000
pause
