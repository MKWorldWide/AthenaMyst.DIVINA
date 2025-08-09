@echo off
echo Starting Optimized Kraken Trading Bot...
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

:: Make sure we're in the correct directory
cd /d "%~dp0"

:: Copy optimized config to .env if it doesn't exist
if not exist .env.optimized (
    echo Creating optimized configuration...
    copy .env .env.optimized >nul
)

:: Start the optimized bot
python optimized_kraken_bot.py

pause
