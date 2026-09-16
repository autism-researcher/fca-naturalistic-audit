@echo off
REM ==========================================================
REM  json_browse.bat - step through trajectories one by one
REM  Requires _json_browse.py in the same folder.
REM  Commands inside: [Enter]=next  p=prev  <number>=jump
REM                   e=export current trajectory to CSV  q=quit
REM ==========================================================
cd /d "%~dp0"
python "%~dp0_json_browse.py"
echo.
pause
