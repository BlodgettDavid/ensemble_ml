@echo off
echo Creating virtual environment...
python -m venv .venv_ensemble

echo Activating environment and installing dependencies...
call .venv_ensemble\Scripts\activate
python.exe -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Setup complete!
echo To activate your environment in the future, run:
echo     .venv_ensemble\Scripts\activate
echo Then launch the project with:
echo     python mainworkflow.py
pause
