@echo off
rem One-time setup: virtual environment, dependencies, sample images.
cd /d "%~dp0"
py -3.12 -m venv .venv || python -m venv .venv
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements-gestures.txt
".venv\Scripts\python.exe" tools\generate_samples.py
echo.
echo Done. Start the app with run.bat
pause
