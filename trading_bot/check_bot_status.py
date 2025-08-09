#!/usr/bin/env python3
"""Script to check the status of the running trading bot."""
import psutil
import os
import json
from datetime import datetime

def check_bot_process():
    """Check if the trading bot process is running."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            # Check if this is our trading bot process
            if proc.info['name'] == 'python.exe' and len(proc.info['cmdline']) > 1:
                if 'kraken_scalper_fixed.py' in ' '.join(proc.info['cmdline']):
                    return {
                        'running': True,
                        'pid': proc.info['pid'],
                        'start_time': datetime.fromtimestamp(proc.info['create_time']).strftime('%Y-%m-%d %H:%M:%S'),
                        'runtime': str(datetime.now() - datetime.fromtimestamp(proc.info['create_time']))
                    }
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return {'running': False}

def check_logs():
    """Check the most recent log entries."""
    log_file = 'kraken_scalper.log'
    if not os.path.exists(log_file):
        return {'log_status': 'No log file found', 'recent_entries': []}
    
    # Get last 10 lines of the log file
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()[-10:]
        return {
            'log_status': 'Log file found',
            'last_modified': datetime.fromtimestamp(os.path.getmtime(log_file)).strftime('%Y-%m-%d %H:%M:%S'),
            'recent_entries': [line.strip() for line in lines if line.strip()]
        }
    except Exception as e:
        return {'log_status': f'Error reading log file: {str(e)}'}

def main():
    print("🔍 Trading Bot Status Check")
    print("=" * 50)
    
    # Check if bot process is running
    process_status = check_bot_process()
    if process_status['running']:
        print(f"✅ Bot is running (PID: {process_status['pid']})")
        print(f"   Started: {process_status['start_time']}")
        print(f"   Uptime:  {process_status['runtime']}")
    else:
        print("❌ Bot is not running")
    
    # Check log file
    log_status = check_logs()
    print(f"\n📋 Log Status: {log_status['log_status']}")
    if 'last_modified' in log_status:
        print(f"   Last modified: {log_status['last_modified']}")
    
    if 'recent_entries' in log_status and log_status['recent_entries']:
        print("\n📜 Recent Log Entries:")
        for entry in log_status['recent_entries']:
            print(f"   {entry}")
    
    print("\n💡 Recommendation:")
    if not process_status['running']:
        print("   - Start the bot using: python kraken_scalper_fixed.py")
    else:
        print("   - Bot appears to be running. Check logs for detailed activity.")
    print("   - For detailed analysis, check the full log file: kraken_scalper.log")

if __name__ == "__main__":
    main()
