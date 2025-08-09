@echo off
echo Stopping all trading bots...

taskkill /F /FI "WINDOWTITLE eq Trading Bots Watchdog*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq status_server*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq kraken*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq binanceus*" >nul 2>&1

echo All trading bots have been stopped.
pause
