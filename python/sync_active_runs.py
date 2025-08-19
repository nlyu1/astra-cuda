#!/usr/bin/env python3
"""
Sync specific active wandb runs continuously.
"""
import subprocess
import time
import os
from pathlib import Path

def sync_active_runs(interval=30):
    """
    Continuously sync the two active runs.
    """
    # The two active runs
    runs = [
        "checkpoints/logs/wandb/offline-run-20250814_213330-3c83jtni",  # main_process
        "checkpoints/logs/wandb/offline-run-20250814_213341-jsi4p107"   # exploiter_process
    ]
    
    print("Starting continuous sync of active runs:")
    print("- Main process: offline-run-20250814_213330-3c83jtni")
    print("- Exploiter process: offline-run-20250814_213341-jsi4p107")
    print(f"Sync interval: {interval} seconds")
    print("Press Ctrl+C to stop\n")
    
    while True:
        try:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n[{timestamp}] Syncing...")
            
            for run_dir in runs:
                run_path = Path(run_dir)
                if run_path.exists():
                    # Get the run name
                    run_name = run_path.name
                    
                    # Check if process is still running
                    if "3c83jtni" in run_name:
                        process_name = "main_process"
                    else:
                        process_name = "exploiter_process"
                    
                    ps_check = subprocess.run(
                        ["pgrep", "-f", f"python {process_name}"],
                        capture_output=True
                    )
                    
                    if ps_check.returncode == 0:
                        status = "🟢 RUNNING"
                    else:
                        status = "🔴 STOPPED"
                    
                    print(f"\n  {process_name} ({run_name}): {status}")
                    
                    # Sync the run
                    result = subprocess.run(
                        ["wandb", "sync", str(run_path)],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        if "Nothing to sync" in result.stderr or "Nothing to sync" in result.stdout:
                            print(f"    Already synced")
                        else:
                            print(f"    ✓ Synced successfully")
                            if result.stdout:
                                print(f"    {result.stdout.strip()}")
                    else:
                        if ".wandb file is empty" in result.stderr:
                            print(f"    ⚠️  Empty .wandb file (run may have just started)")
                        else:
                            print(f"    ❌ Sync failed: {result.stderr.strip()}")
                else:
                    print(f"\n  {run_dir}: Directory not found")
            
            # Wait for next sync
            print(f"\n  Next sync in {interval} seconds...")
            time.sleep(interval)
            
        except KeyboardInterrupt:
            print("\n\nStopping sync...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Retrying in 10 seconds...")
            time.sleep(10)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sync active wandb runs")
    parser.add_argument("--interval", type=int, default=30,
                        help="Sync interval in seconds (default: 30)")
    
    args = parser.parse_args()
    sync_active_runs(args.interval)