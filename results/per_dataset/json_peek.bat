@echo off
REM ==========================================================
REM  json_peek.bat - show the structure of every .json here
REM  Small files: full structure. Huge files: safe header peek.
REM  Requires _json_peek.py in the same folder.
REM ==========================================================
cd /d "%~dp0"
python "%~dp0_json_peek.py"
echo.
pause
