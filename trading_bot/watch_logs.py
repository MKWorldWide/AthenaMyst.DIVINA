import os
import time
import sys
from datetime import datetime

def watch_log_file(log_file='trading_bot.log', num_lines=10):
    """Watch the log file and display updates in real-time."""
    print(f"\n{'='*80}")
    print(f"WATCHING LOG FILE: {log_file}")
    print("Press Ctrl+C to exit")
    print("="*80 + "\n")
    
    # Create the log file if it doesn't exist
    if not os.path.exists(log_file):
        open(log_file, 'a').close()
    
    # Get the initial file size
    file_size = os.path.getsize(log_file)
    
    try:
        while True:
            # Check if file has been modified
            current_size = os.path.getsize(log_file)
            
            if current_size > file_size:
                # File has been modified, read new content
                with open(log_file, 'r') as f:
                    # Go to the last read position
                    f.seek(file_size)
                    # Read new content
                    new_content = f.read()
                    if new_content:
                        print(new_content, end='')
                        sys.stdout.flush()
                    
                    # Update file size
                    file_size = current_size
            
            # Small delay to prevent high CPU usage
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nLog watching stopped.")
    except Exception as e:
        print(f"\nError watching log file: {e}")

if __name__ == "__main__":
    watch_log_file()
