#!/bin/bash
# Run the pipeline in the background, surviving SSH disconnection

# Make sure we're in the right directory
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run with nohup to survive SSH disconnection
# Output goes to logs/pipeline_nohup.log
nohup python run_pipeline.py > logs/pipeline_nohup.log 2>&1 &

# Get the process ID
PID=$!

echo "Pipeline started in background with PID: $PID"
echo "To monitor progress: tail -f logs/pipeline.log"
echo "To check if still running: ps -p $PID"
echo "To stop: kill $PID"
echo ""
echo "Logs:"
echo "  - Main log: logs/pipeline.log"
echo "  - Background log: logs/pipeline_nohup.log"
echo "  - Phonem matching: logs/phonetic_matching.log"
