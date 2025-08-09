@echo off
echo Starting Kraken Scalping Bot...
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

:: Make sure we're in the correct directory
cd /d "%~dp0"

:: Start the bot
python kraken_scalper_fixed.py

pause
