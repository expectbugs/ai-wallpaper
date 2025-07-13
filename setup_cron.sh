#!/bin/bash
#
# Cron Setup Script for Daily Wallpaper Generation
# This script helps install the cron job for 6am daily execution
#

echo "=== Daily Wallpaper Cron Setup ==="
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
    # Add to crontab
    (crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -
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