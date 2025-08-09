#!/usr/bin/env python3
"""Script to analyze Kraken bot logs with proper encoding handling."""
import os
import re
from datetime import datetime

def analyze_log_file(log_file='kraken_scalper.log'):
    """Analyze the log file and extract important information."""
    if not os.path.exists(log_file):
        print(f"Error: Log file '{log_file}' not found.")
        return
    
    print(f"\n📊 Analyzing log file: {log_file}")
    print(f"   Size: {os.path.getsize(log_file) / (1024*1024):.2f} MB")
    print("=" * 80)
    
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(log_file, 'r', encoding=encoding) as f:
                # Read last 1000 lines for analysis
                lines = f.readlines()[-1000:]
                
                if not lines:
                    print(f"   Warning: Log file is empty or couldn't be read with {encoding} encoding.")
                    continue
                
                print(f"\n🔍 Successfully read log with {encoding} encoding")
                print("-" * 60)
                
                # Find and display the last 5 log entries with timestamps
                timestamp_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})')
                log_entries = []
                
                for line in lines:
                    match = timestamp_pattern.match(line)
                    if match:
                        log_entries.append(line.strip())
                
                print("\n🕒 Most Recent Log Entries:")
                for entry in log_entries[-5:]:  # Show last 5 entries
                    print(f"   {entry[:150]}..." if len(entry) > 150 else f"   {entry}")
                
                # Check for errors
                errors = [line for line in lines if 'ERROR' in line.upper()]
                if errors:
                    print(f"\n❌ Found {len(errors)} ERROR(S) in the log:")
                    for error in errors[-3:]:  # Show last 3 errors
                        print(f"   - {error.strip()}")
                else:
                    print("\n✅ No errors found in the log.")
                
                # Check for signals
                signals = [line for line in lines if any(x in line.upper() for x in ['SIGNAL', 'TRADE', 'EXECUTE'])]
                if signals:
                    print(f"\n📡 Found {len(signals)} trading signals:")
                    for signal in signals[-3:]:  # Show last 3 signals
                        print(f"   - {signal.strip()}")
                
                # Print log time range if possible
                if log_entries:
                    first_time = log_entries[0].split(',')[0]
                    last_time = log_entries[-1].split(',')[0]
                    print(f"\n⏰ Log time range: {first_time} to {last_time}")
                
                return  # Successfully read the log
                
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error with {encoding} encoding: {e}")
            continue
    
    print("\n❌ Failed to read log file with any standard encoding.")

if __name__ == "__main__":
    analyze_log_file()
