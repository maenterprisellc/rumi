@echo off
setlocal enabledelayedexpansion

:: Set paths
set "PROJECT_DIR=%~dp0"
set "VENV_DIR=%PROJECT_DIR%.venv"
set "PYTHON_SCRIPT=main.py"

:: Check if virtual environment exists
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Virtual environment not found at %VENV_DIR%
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%"
)

:: Activate virtual environment
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo Failed to activate virtual environment
    exit /b 1
)

echo Virtual environment found at %VENV_DIR%
echo Installing dependencies from requirements.txt..
:: Check if requirements.txt exists
if not exist "%PROJECT_DIR%requirements.txt" (
    echo requirements.txt not found in %PROJECT_DIR%
    echo Please create a requirements.txt file with your dependencies.
    exit /b 1
)   
pip install -r "%PROJECT_DIR%requirements.txt"


:: Run the Python script with arguments
echo Running %PYTHON_SCRIPT%...
python "%PROJECT_DIR%%PYTHON_SCRIPT%" %*

:: Handle errors
if errorlevel 1 (
    echo Error running %PYTHON_SCRIPT%
    exit /b 1
)

:: Deactivate virtual environment
deactivate

echo Script completed successfully
exit /b 0
