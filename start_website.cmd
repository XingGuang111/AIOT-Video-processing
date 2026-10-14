@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Project virtual environment missing. Please create .venv first.
  pause
  exit /b 1
)
echo Open http://127.0.0.1:8080 in your browser.
".venv\Scripts\python.exe" web.py
pause
