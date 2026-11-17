@echo off
rem Start MathVision 3D with the project's virtual environment.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m mathvision3d %*
