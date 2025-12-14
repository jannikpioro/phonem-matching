#!/usr/bin/env python3
"""
Pipeline script to run categorizers and phonem matching.
Runs with logging and can survive SSH disconnection.
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

def log(message):
    """Log with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] {message}"
    print(msg)
    with open("logs/pipeline.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def run_command(command, description):
    """Run a command and log output."""
    log(f"Starting: {description}")
    log(f"Command: {command}")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        
        duration = time.time() - start_time
        log(f"✓ Completed: {description} (took {duration:.1f}s)")
        
        if result.stdout:
            log(f"Output:\n{result.stdout}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        log(f"✗ Failed: {description}")
        log(f"Error code: {e.returncode}")
        if e.stdout:
            log(f"Stdout:\n{e.stdout}")
        if e.stderr:
            log(f"Stderr:\n{e.stderr}")
        return False

def main():
    """Run the full pipeline."""
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    log("="*80)
    log("STARTING PROCESSING PIPELINE")
    log("="*80)
    
    pipeline_start = time.time()
    
    # Step 1: Run English categorizer
    success = run_command(
        "python src/categoriser_en.py",
        "English word categorization"
    )
    if not success:
        log("Pipeline aborted: English categorizer failed")
        sys.exit(1)
    
    # Step 2: Run German categorizer
    success = run_command(
        "python src/categoriser_de.py",
        "German word categorization"
    )
    if not success:
        log("Pipeline aborted: German categorizer failed")
        sys.exit(1)
    
    # Step 3: Run phonem matcher with 0.6 similarity threshold
    log("Updating phonem_matching.py to use 0.6 similarity threshold...")
    
    success = run_command(
        "python src/phonem_matching.py",
        "Phonetic matching (similarity >= 0.6)"
    )
    if not success:
        log("Pipeline aborted: Phonem matching failed")
        sys.exit(1)
    
    # Pipeline completed
    total_duration = time.time() - pipeline_start
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)
    
    log("="*80)
    log(f"✓ PIPELINE COMPLETED SUCCESSFULLY")
    log(f"Total time: {hours}h {minutes}m {seconds}s")
    log("="*80)
    
    # Show results summary
    log("\nResults:")
    log(f"  - Categorized English words: data/grouped_en/")
    log(f"  - Categorized German words: data/grouped_de/")
    log(f"  - Phonetic matches: data/grouped_matches/")
    log(f"  - Match log: logs/phonetic_matching.log")
    log(f"  - Pipeline log: logs/pipeline.log")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        log(f"\nUnexpected error: {e}")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)
