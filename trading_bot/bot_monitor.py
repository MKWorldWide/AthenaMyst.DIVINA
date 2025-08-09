#!/usr/bin/env python3
"""Real-time monitoring for the Kraken trading bot."""
import os
import time
import json
import subprocess
from datetime import datetime

def get_bot_process():
    """Get information about the running bot process."""
    try:
        # On Windows, we'll use tasklist to find the process
        result = subprocess.run(
            ['wmic', 'process', 'where', "name='python.exe'" , 'get', 'CommandLine,ProcessId,Name'],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            for line in lines:
                if 'kraken_scalper_fixed.py' in line:
                    parts = line.strip().split()
                    return {
                        'pid': int(parts[-1]),
                        'running': True,
                        'command': ' '.join(parts[:-1])
                    }
    except Exception as e:
        print(f"Error checking process: {e}")
    
    return {'running': False}

def get_latest_signals(log_file='kraken_scalper.log', num_lines=50):
    """Extract and analyze recent trading signals from the log file."""
    if not os.path.exists(log_file):
        return []
    
    try:
        # Use PowerShell to get the last N lines of the log file
        result = subprocess.run(
            ['powershell', '-Command', f'Get-Content -Path "{log_file}" -Tail {num_lines} | Select-String -Pattern "Signal|Confidence|ERROR"'],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        
        if result.returncode == 0 and result.stdout.strip():
            return [line.strip() for line in result.stdout.split('\n') if line.strip()]
            
    except Exception as e:
        print(f"Error reading log file: {e}")
    
    return ["No recent signals found in log"]

def display_dashboard(process_info, signals):
    """Display a clean dashboard of bot status and recent signals."""
    os.system('cls' if os.name == 'nt' else 'clear')
    print("""
    🚀 Kraken Trading Bot Monitor
    ============================
    Status: {status}
    {details}
    
    📊 Recent Signals & Errors:
    {signals}
    
    Press Ctrl+C to exit
    """.format(
        status="🟢 RUNNING" if process_info['running'] else "🔴 STOPPED",
        details=f"PID: {process_info['pid']} | Command: {process_info.get('command', 'N/A')}" 
                if process_info['running'] else "The bot is not currently running.",
        signals='\n'.join(f"    • {sig}" for sig in signals[-10:])  # Show last 10 signals
    ))

def main():
    """Main monitoring loop."""
    try:
        while True:
            process_info = get_bot_process()
            signals = get_latest_signals()
            display_dashboard(process_info, signals)
            time.sleep(5)  # Update every 5 seconds
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    main()
