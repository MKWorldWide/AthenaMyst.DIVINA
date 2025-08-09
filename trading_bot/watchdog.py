#!/usr/bin/env python3
"""
Watchdog script to ensure trading bots stay running 24/7.
Monitors both Kraken and Binance.US trading bots and restarts them if they crash.
"""
import os
import sys
import time
import logging
import subprocess
import signal
from pathlib import Path
from typing import Dict, Optional, List

# Configure logging
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'watchdog.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('watchdog')

# Create a PnL log file if it doesn't exist
pnl_log = Path('trading_pnl.csv')
if not pnl_log.exists():
    pnl_log.write_text('timestamp,exchange,symbol,side,price,amount,cost,pnl,pnl_pct,fee,fee_currency,status\n')

class ProcessManager:
    def __init__(self):
        self.processes: Dict[str, Optional[subprocess.Popen]] = {
            'status_server': None,
            'kraken': None,
            'binanceus': None
        }
        self.base_cmd = [sys.executable]  # Use the same Python interpreter
        self.scripts_dir = Path(__file__).parent
        self.max_restart_attempts = 3
        self.restart_attempts = {name: 0 for name in self.processes}
        self.startup_delay = 5  # seconds between process starts
        
    def start_process(self, name: str) -> bool:
        """Start a process by name with error handling and logging"""
        # Reset restart attempts if we've had too many failures
        if self.restart_attempts.get(name, 0) >= self.max_restart_attempts:
            logger.error(f"Too many restart attempts for {name}. Manual intervention required.")
            return False
            
        if name == 'status_server':
            cmd = self.base_cmd + ['status_server.py']
        elif name in ['kraken', 'binanceus']:
            cmd = self.base_cmd + ['run_bot.py', '--exchange', name]
        else:
            logger.error(f"Unknown process: {name}")
            return False
        
        try:
            # Set up log files for each process
            log_file = log_dir / f"{name}.log"
            with open(log_file, 'a') as f:
                f.write(f"\n{'='*40} Process Started at {time.ctime()} {'='*40}\n")
                f.flush()
                
                # Start the process with output redirection
                if os.name == 'nt':  # Windows
                    self.processes[name] = subprocess.Popen(
                        cmd,
                        cwd=self.scripts_dir,
                        creationflags=subprocess.CREATE_NEW_CONSOLE,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        text=True
                    )
                else:  # Unix-like
                    self.processes[name] = subprocess.Popen(
                        cmd,
                        cwd=self.scripts_dir,
                        start_new_session=True,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        text=True
                    )
            
            pid = self.processes[name].pid
            logger.info(f"Started {name} with PID {pid}")
            self.restart_attempts[name] += 1
            time.sleep(self.startup_delay)  # Give the process time to start
            return True
            
        except Exception as e:
            logger.error(f"Failed to start {name}: {e}", exc_info=True)
            self.processes[name] = None
            return False
    
    def check_process(self, name: str) -> bool:
        """Check if a process is running and restart if needed"""
        process = self.processes.get(name)
        
        # If process doesn't exist, start it
        if process is None:
            logger.warning(f"Process {name} not found, starting...")
            return self.start_process(name)
        
        # Check if process is still running
        return_code = process.poll()
        if return_code is not None:
            logger.warning(
                f"Process {name} (PID: {process.pid}) died with return code {return_code}. "
                f"Attempting to restart (attempt {self.restart_attempts.get(name, 0) + 1}/{self.max_restart_attempts})"
            )
            
            # Log the last few lines of the process output for debugging
            try:
                log_file = log_dir / f"{name}.log"
                if log_file.exists():
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            logger.debug(f"Last 5 lines of {name} output:")
                            for line in lines[-5:]:
                                logger.debug(f"  {line.strip()}")
            except Exception as e:
                logger.error(f"Error reading log file for {name}: {e}")
            
            return self.start_process(name)
        
        return True
    
    def run(self):
        """Main watchdog loop"""
        logger.info("Starting trading bot watchdog service")
        logger.info(f"Scripts directory: {self.scripts_dir}")
        
        # Create necessary directories
        (self.scripts_dir / 'logs').mkdir(exist_ok=True)
        
        # Initial startup with delay between processes
        for name in self.processes:
            if not self.start_process(name):
                logger.error(f"Failed to start {name} during initialization")
            time.sleep(2)  # Small delay between starting processes
        
        # Main monitoring loop
        check_interval = 10  # seconds between checks
        last_check = time.time()
        
        try:
            while True:
                current_time = time.time()
                
                # Check all processes
                for name in list(self.processes.keys()):
                    if not self.check_process(name):
                        logger.error(f"Critical error with {name}, waiting before retry...")
                        time.sleep(30)  # Longer delay on critical errors
                
                # Log status periodically
                if current_time - last_check >= 300:  # Every 5 minutes
                    self.log_status()
                    last_check = current_time
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested, cleaning up...")
        except Exception as e:
            logger.critical(f"Fatal error in watchdog: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    def log_status(self):
        """Log current status of all processes"""
        status = []
        for name, process in self.processes.items():
            if process is None:
                status.append(f"{name}: Not running")
            elif process.poll() is None:
                status.append(f"{name}: Running (PID: {process.pid})")
            else:
                status.append(f"{name}: Crashed (Exit code: {process.returncode})")
        
        logger.info("Current status: " + ", ".join(status))
    
    def cleanup(self):
        """Clean up all processes"""
        logger.info("Cleaning up processes...")
        for name, process in list(self.processes.items()):
            if process and process.poll() is None:
                logger.info(f"Terminating {name} (PID: {process.pid})")
                try:
                    # Try graceful shutdown first
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                        logger.info(f"{name} terminated successfully")
                        continue
                    except subprocess.TimeoutExpired:
                        logger.warning(f"{name} did not terminate, forcing kill...")
                        process.kill()
                        process.wait()
                        logger.warning(f"{name} force killed")
                except Exception as e:
                    logger.error(f"Error terminating {name}: {e}")

if __name__ == "__main__":
    manager = ProcessManager()
    try:
        manager.run()
    except Exception as e:
        logger.critical(f"Fatal error in watchdog: {e}", exc_info=True)
        manager.cleanup()
        sys.exit(1)
