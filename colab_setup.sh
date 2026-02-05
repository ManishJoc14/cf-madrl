#!/bin/bash

# 1. Install System Dependencies (SUMO)
echo "Installing SUMO..."
sudo add-apt-repository ppa:sumo/stable -y
sudo apt-get update -y
sudo apt-get install sumo sumo-tools sumo-doc -y

# 2. Set Environment Variables
echo "Exporting SUMO_HOME..."
export SUMO_HOME="/usr/share/sumo"
echo "export SUMO_HOME=/usr/share/sumo" >> ~/.bashrc

# 3. Install Python Dependencies
echo "Installing Python packages..."
pip install -r requirements.txt

# 4. Verification
echo "Verifying installation..."
sumo --version
python -c "import traci; print('TraCI installed successfully')"

echo "Setup complete! You can now run training."
