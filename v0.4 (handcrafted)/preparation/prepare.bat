@echo off

REM NAME   : prepare.bat
REM AUTHOR : David Carteau, France, August 2025
REM LICENSE: MIT (see "license.txt" file content)
REM PROJECT: Nostradamus UCI chess engine
REM PURPOSE: Prepare training data

python combine.py

REM NOTE:
REM - The '-M' flag is set by default
REM - It only matches games that end in checkmate
REM - Adjust if needed!
pgn-extract.exe -s -M -Wuci --fencomments --commentlines games.pgn | python prepare.py

python shuffle.py

pause
