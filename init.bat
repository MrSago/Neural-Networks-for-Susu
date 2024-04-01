@echo off
python -m venv .venv
powershell.exe .\.venv\Scripts\Activate.ps1
python -m pip install -r ./requirements.txt