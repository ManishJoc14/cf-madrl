#!/bin/bash
# CF-MADRL Traffic Monitor Setup

echo "================================"
echo "CF-MADRL Monitor - Setup"
echo "================================"

# Update system
echo "[1/4] Updating system..."
sudo apt-get update && sudo apt-get upgrade -y

# Install dependencies
echo "[2/4] Installing dependencies..."
sudo apt-get install -y python3-pip python3-venv libatlas-base-dev

# Create venv
echo "[3/4] Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install packages
echo "[4/4] Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit config.yaml with ESP32 IP addresses"
echo "2. Run: source .venv/bin/activate"
echo "3. Run: python3 monitor.py"
echo ""

