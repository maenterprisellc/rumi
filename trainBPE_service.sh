#!/bin/bash

# Set your project and venv paths
PROJECT_DIR="/home/$(whoami)/rumi"
VENV_DIR="$PROJECT_DIR/.venv"
MAIN_SCRIPT="$PROJECT_DIR/main.py"
PROCESS_ARG="--process buildbpe"  # Change to clean/collect as needed
SERVICE_FILE="/etc/systemd/system/rumi.service"

# Update rumi.service file
sudo tee $SERVICE_FILE > /dev/null <<EOF
[Unit]
Description=Rumiv1 Background Service
After=network.target

[Service]
User=$(whoami)
Group=$(whoami)
WorkingDirectory=$PROJECT_DIR
ExecStart=/bin/bash -c 'source $VENV_DIR/bin/activate && $VENV_DIR/bin/python -m $MAIN_SCRIPT $PROCESS_ARG'
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

echo "Starting rumi.service ..."
sudo systemctl start rumi.service

echo "Enabling rumi.service to start on boot..."
sudo systemctl enable rumi.service

echo "Checking service status:"
sudo systemctl status rumi.service --no-pager