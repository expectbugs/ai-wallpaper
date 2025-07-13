#!/bin/bash
#
# Cron Setup Script for Daily Wallpaper Generation
# This script helps install the cron job for 6am daily execution
#

echo "=== Daily Wallpaper Cron Setup ==="
echo

# Check if cron service is running - FAIL LOUDLY if not
echo "Checking if cron service is running..."
if ! pgrep -x "cron" > /dev/null && ! pgrep -x "crond" > /dev/null; then
    echo "ERROR: Cron service is not running!"
    echo "Please start the cron service before running this script."
    echo "Try one of these commands:"
    echo "  sudo systemctl start cron"
    echo "  sudo systemctl start crond"
    echo "  sudo service cron start"
    echo "  sudo service crond start"
    exit 1
fi
echo "Cron service is running."
echo

# Check if script exists and is executable
if [ ! -x "/home/user/ai-wallpaper/run_daily_wallpaper.sh" ]; then
    echo "ERROR: Wrapper script not found or not executable!"
    echo "Expected at: /home/user/ai-wallpaper/run_daily_wallpaper.sh"
    exit 1
fi

# Show current crontab
echo "Current crontab entries:"
crontab -l 2>/dev/null || echo "(no existing crontab)"
echo

# Create the cron entry
# NO OUTPUT SUPPRESSION - errors will be emailed by cron (fail loud!)
CRON_ENTRY="0 6 * * * /home/user/ai-wallpaper/run_daily_wallpaper.sh"

echo "Proposed cron entry:"
echo "$CRON_ENTRY"
echo
echo "This will run the wallpaper generator every day at 6:00 AM"
echo

# Ask for confirmation
read -p "Do you want to add this cron entry? (y/N) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Add to crontab - FAIL LOUDLY if it doesn't work
    if ! (crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -; then
        echo "ERROR: Failed to add cron entry!"
        echo "The crontab command failed. Possible reasons:"
        echo "  - No permission to modify crontab"
        echo "  - Crontab syntax error"
        echo "  - System error"
        exit 1
    fi
    
    # Verify the entry was actually added
    if ! crontab -l 2>/dev/null | grep -q "$CRON_ENTRY"; then
        echo "ERROR: Cron entry was not added successfully!"
        echo "The entry does not appear in the current crontab."
        echo "This is a critical failure - manual intervention required."
        exit 1
    fi
    
    echo "Cron entry added successfully!"
    echo
    echo "To verify, run: crontab -l"
    echo "To remove, run: crontab -e and delete the line"
else
    echo "Cron setup cancelled."
    echo
    echo "To add manually, run: crontab -e"
    echo "Then add this line:"
    echo "$CRON_ENTRY"
fi

echo
echo "=== Setup Complete ==="