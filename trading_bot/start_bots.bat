@echo off
echo Starting Trading Bots with Watchdog...

title Trading Bots Watchdog
echo Starting Watchdog...
start "Trading Bots Watchdog" python watchdog.py

echo Watchdog started in a new window.
echo Press any key to exit this window (this will NOT stop the bots).
pause > nul
