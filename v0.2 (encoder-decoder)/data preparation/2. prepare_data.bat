@echo off

REM NAME   : 2. prepare_data.bat
REM AUTHOR : David Carteau, France, November 2024
REM LICENSE: MIT (see "license.txt" file content)
REM PROJECT: Nostradamus UCI chess engine
REM PURPOSE: Data preparation

pgn-extract.exe -s -Wuci --fencomments --commentlines "./pgn/games.pgn" | python prepare_data.py

pause
