#!/bin/bash

# Setup script for systemd user service

echo "Setting up Bible Verse Notifier systemd service..."

# Create systemd user directory
mkdir -p ~/.config/systemd/user

# Copy service files
cp verse-notifier.service ~/.config/systemd/user/
cp verse-notifier.timer ~/.config/systemd/user/

# Reload systemd daemon
systemctl --user daemon-reload

# Enable and start the timer
systemctl --user enable verse-notifier.timer
systemctl --user start verse-notifier.timer

# Check status
echo "Service status:"
systemctl --user status verse-notifier.timer

echo "Next run time:"
systemctl --user list-timers verse-notifier.timer

echo "Setup complete! The verse notifier will now run every hour."