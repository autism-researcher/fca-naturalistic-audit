@echo off
REM ==========================================================
REM  json_summary.bat - the whole file at a glance
REM  For every *_features.json here it writes:
REM     <dataset>_summary.csv   (one row per trajectory)
REM     <dataset>_overview.png  (R_max histogram, if matplotlib)
REM  and prints B_d / crossing statistics per tau.
REM  Requires _json_summary.py in the same folder.
REM  NOTE: NGSIM (847 MB) takes 1-2 minutes to load.
REM ==========================================================
cd /d "%~dp0"
python "%~dp0_json_summary.py"
echo.
pause
