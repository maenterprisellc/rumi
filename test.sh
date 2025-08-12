#!/bin/bash

# Set paths
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="${PROJECT_DIR}/.venv"
PYTHON_SCRIPT="main.py"

# Check if virtual environment exists
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "Virtual environment not found at ${VENV_DIR}"
    echo "Creating virtual environment..."
    python3 -m venv "${VENV_DIR}"
fi

# Activate virtual environment
source "${VENV_DIR}/bin/activate"
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment"
    exit 1
fi

echo "Virtual environment found at ${VENV_DIR}"
echo "Installing dependencies from requirements.txt..."

# Check if requirements.txt exists
if [ ! -f "${PROJECT_DIR}/requirements.txt" ]; then
    echo "requirements.txt not found in ${PROJECT_DIR}"
    echo "Please create a requirements.txt file with your dependencies."
    exit 1
fi

pip install -r "${PROJECT_DIR}/requirements.txt"

# Run the Python script with arguments
echo "Running ${PYTHON_SCRIPT}..."
python "${PROJECT_DIR}/${PYTHON_SCRIPT}" "$@"

# Handle errors
if [ $? -ne 0 ]; then
    echo "Error running ${PYTHON_SCRIPT}"
    exit 1
fi

# Deactivate virtual environment
deactivate

echo "Script completed successfully"
exit 0
