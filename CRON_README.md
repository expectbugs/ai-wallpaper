# Cron Setup

Instructions for automatic daily wallpaper generation at 6:00 AM using cron.

## Components

### 1. Wrapper Script: `run_daily_wallpaper.sh`
- Sets up environment for cron execution
- Activates Python virtual environment
- Handles log rotation (30 days)
- Captures output to log files

### 2. Setup Script: `setup_cron.sh`
- Installs cron job
- Shows current crontab
- Asks for confirmation

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

Check if cron job is installed:
```bash
crontab -l
```

Monitor execution:
```bash
tail -f /home/user/ai-wallpaper/logs/cron_*.log
ls -la /home/user/ai-wallpaper/logs/
```

Test the wrapper script:
```bash
/home/user/ai-wallpaper/run_daily_wallpaper.sh
```

## Troubleshooting

### Common Issues

1. **DISPLAY not set** - Wrapper script sets `DISPLAY=:0`
2. **Python packages not found** - Wrapper activates venv automatically
3. **Ollama server not running** - Python script starts it automatically
4. **No wallpaper change** - Check logs in `/home/user/ai-wallpaper/logs/`

### Log Files

- Location: `/home/user/ai-wallpaper/logs/cron_YYYY-MM-DD_HH-MM-SS.log`
- Auto-rotation: 30 days
- Contains full Python script output

### Disable

```bash
crontab -e
# Comment out or delete the wallpaper line
```

## Testing Different Times

Modify the cron schedule:
- Every hour: `0 * * * *`
- Every 5 minutes: `*/5 * * * *`
- Specific time: `30 14 * * *` (2:30 PM)