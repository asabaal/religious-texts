#!/bin/bash
cp verse-notifier.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user restart verse-notifier.service
echo "Service updated and restarted"