import win32serviceutil
import win32service
import win32event
import servicemanager
import socket
import sys
import os
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TradingBotService')

class TradingBotService(win32serviceutil.ServiceFramework):
    _svc_name_ = "OANDATradingBot"
    _svc_display_name_ = "OANDA Trading Bot Service"
    _svc_description_ = "Runs the OANDA trading bot as a Windows service"

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        socket.setdefaulttimeout(60)
        self.process = None
        self.script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'oanda_multi_scalper.py')
        
    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)
        if self.process:
            self.process.terminate()
            logger.info("Stopped trading bot process")
        
    def SvcDoRun(self):
        self.ReportServiceStatus(win32service.SERVICE_RUNNING)
        self.main()

    def main(self):
        logger.info("Starting OANDA Trading Bot service...")
        
        # Set working directory to script directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        try:
            # Start the trading bot as a subprocess
            self.process = subprocess.Popen(
                [sys.executable, self.script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            logger.info(f"Trading bot started with PID: {self.process.pid}")
            
            # Monitor the process
            while True:
                if self.process.poll() is not None:
                    # Process has ended, restart it
                    logger.warning("Trading bot process ended, restarting...")
                    self.process = subprocess.Popen(
                        [sys.executable, self.script_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    logger.info(f"Restarted trading bot with PID: {self.process.pid}")
                
                # Check for stop event
                if win32event.WaitForSingleObject(self.hWaitStop, 5000) == win32event.WAIT_OBJECT_0:
                    # Stop signal received
                    if self.process:
                        self.process.terminate()
                        logger.info("Stopped trading bot process")
                    break
                    
        except Exception as e:
            logger.error(f"Error in trading bot service: {e}")
            raise

if __name__ == '__main__':
    if len(sys.argv) == 1:
        # If no arguments, run the service directly (for debugging)
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(TradingBotService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        # Handle service commands (install, start, stop, etc.)
        win32serviceutil.HandleCommandLine(TradingBotService)
