@echo off
echo Starting Kraken Scalping Bot with enhanced logging...

:: Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

:: Set Python path if needed
set PYTHONPATH=%~dp0

:: Start the bot with log rotation
python kraken_scalper_fixed.py

:: If there's an error, pause so we can see it
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error starting bot. Press any key to exit...
    pause
)
