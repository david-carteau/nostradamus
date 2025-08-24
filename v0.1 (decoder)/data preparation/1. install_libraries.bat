@echo off

REM NAME   : 1. install_librairies.bat
REM AUTHOR : David Carteau, France, November 2024
REM LICENSE: MIT (see "license.txt" file content)
REM PROJECT: Nostradamus UCI chess engine
REM PURPOSE: Installation of Python libraries

echo transformers==4.46.0 > requirements.txt
echo tokenizers==0.20.1 >> requirements.txt
echo accelerate==1.0.1 >> requirements.txt
echo chess==1.11.1 >> requirements.txt
echo tqdm==4.66.5 >> requirements.txt

pip cache purge
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip freeze > freeze.txt

pause
