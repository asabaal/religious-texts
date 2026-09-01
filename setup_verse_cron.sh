#!/bin/bash

# Bible Verse Notifier Cron Setup Script
# This script sets up a cron job to run the verse notifier every hour

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/verse_notifier.py"
LOG_FILE="$SCRIPT_DIR/verse_cron.log"

echo "🕊 Setting up Bible Verse Notifier cron job..."
echo "Script location: $PYTHON_SCRIPT"
echo "Log file: $LOG_FILE"

# Create the cron command
CRON_CMD="0 * * * * cd $SCRIPT_DIR && /usr/bin/python3 $PYTHON_SCRIPT --cron >> $LOG_FILE 2>&1"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "verse_notifier.py"; then
    echo "⚠️  Verse notifier cron job already exists. Removing old one..."
    crontab -l 2>/dev/null | grep -v "verse_notifier.py" | crontab -
fi

# Add the new cron job
(crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -

echo "✅ Cron job added! You will receive a Bible verse notification every hour."
echo ""
echo "📋 Cron job details:"
echo "   Schedule: Every hour at minute 0"
echo "   Command: $CRON_CMD"
echo ""
echo "🔧 To manage your cron jobs:"
echo "   View all:    crontab -l"
echo "   Edit:        crontab -e"
echo "   Remove all:  crontab -r"
echo ""
echo "📝 Logs will be saved to: $LOG_FILE"
echo "🧪 To test: python3 $PYTHON_SCRIPT --test"