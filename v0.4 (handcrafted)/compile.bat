@echo off

set VENV=%USERPROFILE%\venv-nostradamus-v0.4

python -m venv %VENV%
call %VENV%\Scripts\activate.bat
pip cache purge
pip install torch==2.7.1 numpy==2.2.4 chess==1.11.2 tqdm==4.67.1 pyinstaller
%VENV%\Scripts\pyinstaller --onefile --add-data "models;models" --name nostradamus-v0.4.exe nostradamus.py

pause