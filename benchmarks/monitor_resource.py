import time
import sys
import os
from utils.system_monitor import SystemMonitor

def main():
    print("Initializing System Monitor...")
    monitor = SystemMonitor()
    
    print("\nStarting monitoring... (Press Ctrl+C to stop)")
    print("-" * 80)
    
    try:
        while True:
            metrics = monitor.get_metrics()
            output = monitor.format_metrics(metrics)
            
            # Clear line and print new stats
            sys.stdout.write(f"\r{output}")
            sys.stdout.flush()
            
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    main()
