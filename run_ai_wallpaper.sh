#!/bin/bash
#
# AI Wallpaper Cron Runner Script
# This script is designed to be called from cron for daily wallpaper generation
#

# Set up environment
export DISPLAY=:0
export XAUTHORITY=/home/user/.Xauthority

# Change to script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Log file with date
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/cron_$(date +%Y-%m-%d).log"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

log "===== Starting AI Wallpaper Generation ====="
log "Working directory: $PWD"
log "User: $(whoami)"
log "Display: $DISPLAY"

# Check if virtual environment exists
if [ -d "/home/user/grace/.venv" ]; then
    log "Activating virtual environment"
    source /home/user/grace/.venv/bin/activate
else
    log "WARNING: Virtual environment not found at /home/user/grace/.venv"
fi

# Check environment variables
if [ -z "$OPENAI_API_KEY" ]; then
    log "WARNING: OPENAI_API_KEY not set - DALL-E and GPT models will not work"
fi

# Run the wallpaper generator with random model selection
log "Running ai-wallpaper generate --random-model"

# Execute with error handling
if ./ai-wallpaper generate --random-model >> "$LOG_FILE" 2>&1; then
    log "Wallpaper generation completed successfully"
    EXIT_CODE=0
else
    EXIT_CODE=$?
    log "ERROR: Wallpaper generation failed with exit code $EXIT_CODE"
fi

log "===== AI Wallpaper Generation Finished ====="
log ""

exit $EXIT_CODE