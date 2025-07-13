#!/bin/bash
#
# Daily Wallpaper Generation Wrapper Script for Cron
# Phase 6: Scheduling implementation
#
# This script sets up the environment and runs the wallpaper generator
# Designed to be called from cron at 6am daily
#

# Exit on any error (fail fast, fail loud)
set -e

# Set up logging
SCRIPT_DIR="/home/user/ai-wallpaper"
LOG_FILE="$SCRIPT_DIR/logs/cron_$(date +%Y-%m-%d_%H-%M-%S).log"

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Start logging
log_message "=== STARTING CRON WALLPAPER GENERATION ==="
log_message "Script: $0"
log_message "User: $(whoami)"
log_message "Working directory: $(pwd)"

# Set up environment variables
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/bin"
export DISPLAY=":0"
export HOME="/home/user"
export USER="user"
export LANG="en_US.UTF-8"

log_message "Environment set: DISPLAY=$DISPLAY, PATH=$PATH"

# Change to script directory
cd "$SCRIPT_DIR" || exit 1
log_message "Changed to directory: $SCRIPT_DIR"

# NOTE: Log rotation moved to end of script - only delete old logs after successful completion

# Activate Python virtual environment
VENV_PATH="/home/user/grace/.venv"
log_message "Activating virtual environment: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# Verify Python and required packages
log_message "Python version: $(python3 --version)"
python3 -c "import diffusers, torch, PIL; print('Required packages verified')" 2>&1 | tee -a "$LOG_FILE"

# Run the wallpaper generator
log_message "Starting wallpaper generation..."
log_message "Command: python3 daily_wallpaper.py --run-now"

# Execute with full output capture
python3 daily_wallpaper.py --run-now 2>&1 | tee -a "$LOG_FILE"

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    log_message "Wallpaper generation completed successfully!"
else
    log_message "ERROR: Wallpaper generation failed with exit code ${PIPESTATUS[0]}"
    exit 1
fi

# Deactivate virtual environment
deactivate

# Log rotation - ONLY after successful completion
log_message "Performing log rotation (removing logs older than 30 days)..."
find "$SCRIPT_DIR/logs" -name "*.log" -type f -mtime +30 -delete
LOG_COUNT=$(find "$SCRIPT_DIR/logs" -name "*.log" -type f | wc -l)
log_message "Log rotation complete. Current log count: $LOG_COUNT"

log_message "=== CRON WALLPAPER GENERATION COMPLETE ==="
log_message "Log saved to: $LOG_FILE"

# Exit successfully
exit 0