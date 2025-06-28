#!/bin/bash
# Before running: make this file executable with
#     chmod +x setup_env.sh

echo "Creating virtual environment..."
python3 -m venv .venv_ensemble

echo "Activating environment and installing dependencies..."
source .venv_ensemble/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo "To activate your environment in the future, run:"
echo "    source .venv_ensemble/bin/activate"
echo "Then launch the project with:"
echo "    python mainworkflow.py"
