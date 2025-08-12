@echo off
title Divina Diagnostics & Live Monitor

REM Create logs directory if it doesn't exist
if not exist "logs\divina" mkdir "logs\divina"

REM Set Python path and run the diagnostics script
python divina_diagnostics.py

REM Pause to see any error messages before closing
pause
