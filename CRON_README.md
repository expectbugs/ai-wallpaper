# Cron Setup Instructions for Daily Wallpaper Generation

## Overview
This document explains how to set up automatic daily wallpaper generation at 6:00 AM using cron.

## Components

### 1. Wrapper Script: `run_daily_wallpaper.sh`
- Sets up the complete environment for cron execution
- Activates Python virtual environment
- Handles log rotation (keeps last 30 days)
- Captures all output to log files
- Fails loudly on any error

### 2. Setup Script: `setup_cron.sh`
- Interactive script to install cron job
- Shows current crontab before making changes
- Asks for confirmation before installing

## Installation

### Automatic Setup
```bash
cd /home/user/ai-wallpaper
./setup_cron.sh
```

### Manual Setup
1. Edit your crontab:
   ```bash
   crontab -e
   ```

2. Add this line:
   ```
   0 6 * * * /home/user/ai-wallpaper/run_daily_wallpaper.sh
   ```
   Note: No output redirection - failures will be emailed (fail loud!)

3. Save and exit

## Cron Schedule Explanation
```
0 6 * * * /home/user/ai-wallpaper/run_daily_wallpaper.sh
│ │ │ │ │
│ │ │ │ └─── Day of week (0-7, Sunday is 0 or 7)
│ │ │ └───── Month (1-12)
│ │ └─────── Day of month (1-31)
│ └───────── Hour (0-23, 6 = 6:00 AM)
└─────────── Minute (0-59)
```

## Verification

### Check if cron job is installed:
```bash
crontab -l
```

### Monitor execution:
```bash
# Watch latest cron log
tail -f /home/user/ai-wallpaper/logs/cron_*.log

# Check all logs
ls -la /home/user/ai-wallpaper/logs/
```

### Test the wrapper script manually:
```bash
/home/user/ai-wallpaper/run_daily_wallpaper.sh
```

## Troubleshooting

### Common Issues:

1. **DISPLAY not set**
   - The wrapper script sets `DISPLAY=:0`
   - Ensure X11 is running on display :0

2. **Python packages not found**
   - Wrapper activates venv automatically
   - Test with: `source /home/user/grace/.venv/bin/activate`

3. **Ollama server not running**
   - The Python script starts it automatically
   - Can manually start: `ollama serve &`

4. **No wallpaper change**
   - Check logs in `/home/user/ai-wallpaper/logs/`
   - Verify XFCE4 is running
   - Test manually: `./run_daily_wallpaper.sh`

### Log Files

All cron executions create detailed logs:
- Location: `/home/user/ai-wallpaper/logs/cron_YYYY-MM-DD_HH-MM-SS.log`
- Auto-rotation: Logs older than 30 days are deleted
- Contains full output from Python script

### Disable/Remove

To disable the cron job:
```bash
crontab -e
# Comment out or delete the wallpaper line
```

## Important Notes

1. **Time Zone**: Cron uses system time. Verify with `date`
2. **User Crontab**: This uses user crontab, not system crontab
3. **Environment**: Cron runs with minimal environment; wrapper handles setup
4. **Failures**: Any error causes immediate exit (fail-fast philosophy)
5. **Email Alerts**: Errors will be emailed by cron (fail loud philosophy)

## Testing Different Times

To test at different times, modify the cron schedule:
- Every hour: `0 * * * *`
- Every 5 minutes: `*/5 * * * *`
- Specific time: `30 14 * * *` (2:30 PM)

Remember to change back to `0 6 * * *` for 6:00 AM daily execution.