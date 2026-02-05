#!/bin/bash

echo "==========================================="
echo "  YOLO Scripts WebUI Installer"
echo "==========================================="
echo ""

# 1. Virtual Environment Prompt
read -p "Do you want to create/use a python virtual environment (yolovenv)? [y/N] " use_venv
use_venv=${use_venv:-N}

if [[ "$use_venv" =~ ^[Yy]$ ]]; then
    if [ ! -d "yolovenv" ]; then
        echo "Creating virtual environment 'yolovenv'..."
        python3 -m venv yolovenv
    else
        echo "Found existing virtual environment 'yolovenv'."
    fi
    
    echo "Activating virtual environment..."
    source yolovenv/bin/activate
else
    echo "Skipping virtual environment setup. Installing globally (or in active venv)."
fi

# 2. Backend Installation
echo ""
echo "-------------------------------------------"
echo "  Installing Backend Dependencies..."
echo "-------------------------------------------"
cd backend || exit
pip install -r requirements.txt
cd ..

# 3. Frontend Installation
echo ""
echo "-------------------------------------------"
echo "  Installing Frontend Dependencies..."
echo "-------------------------------------------"
cd frontend || exit
echo "Cleaning up previous install artifacts..."
rm -rf node_modules package-lock.json
npm cache clean --force
npm config set progress=false
npm config set fetch-retries 5
npm config set fetch-retry-mintimeout 20000
npm config set fetch-retry-maxtimeout 120000
echo "Installing Frontend Dependencies (Verbose mode)..."
npm install --verbose --no-audit --no-fund
# Ensure xterm and addons are installed
npm install xterm xterm-addon-fit xterm-addon-web-links
cd ..

# 4. Completion Instructions
echo ""
echo "==========================================="
echo "  Installation Complete!"
echo "==========================================="
echo ""
echo "To run the WebUI locally:"
echo "  ./webui.sh start --port 4000"
echo "  Then open: http://localhost:4000"
echo ""
echo "To run on a Remote Server:"
echo "  1. On the server: ./webui.sh start --port 4000"
echo "  2. On your local machine: ssh -L 4000:localhost:4000 user@remote-ip"
echo "  3. Open http://localhost:4000"
echo ""
echo "To stop the server:"
echo "  ./webui.sh stop"
echo ""
if [[ "$use_venv" =~ ^[Yy]$ ]]; then
    echo "NOTE: If running manually without webui.sh, remember to activate the environment:"
    echo "  source yolovenv/bin/activate"
fi
echo ""
