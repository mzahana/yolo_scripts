#!/bin/bash

# Configuration
BACKEND_PID_FILE=".webui_backend.pid"
FRONTEND_PID_FILE=".webui_frontend.pid"
BACKEND_LOG="backend.log"
FRONTEND_LOG="frontend.log"
BACKEND_DIR="backend"
FRONTEND_DIR="frontend"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: ./webui.sh {start|stop|restart} [--venv /path/to/venv] [--port PORT]"
    exit 1
}

if [ -z "$1" ]; then
    usage
fi

ACTION="$1"
shift

# Parse Arguments
VENV_PATH=""
FRONTEND_PORT="3000"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --venv) VENV_PATH="$2"; shift ;;
        --port) FRONTEND_PORT="$2"; shift ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

start() {
    echo -e "${GREEN}Starting WebUI...${NC}"

    # 1. Activate Venv if provided
    if [ -n "$VENV_PATH" ]; then
        if [ -f "$VENV_PATH/bin/activate" ]; then
            echo "Activating virtual environment: $VENV_PATH"
            source "$VENV_PATH/bin/activate"
        else
            echo -e "${RED}Error: Virtual environment not found at $VENV_PATH${NC}"
            exit 1
        fi
    fi

    # 2. Start Backend
    if [ -f "$BACKEND_PID_FILE" ] && kill -0 $(cat "$BACKEND_PID_FILE") 2>/dev/null; then
        echo "Backend is already running (PID: $(cat $BACKEND_PID_FILE))"
    else
        echo "Starting Backend..."
        cd "$BACKEND_DIR" || exit
        nohup python -u main.py > "../$BACKEND_LOG" 2>&1 &
        echo $! > "../$BACKEND_PID_FILE"
        cd ..
        echo "Backend started (PID: $(cat $BACKEND_PID_FILE))"
    fi

    # 3. Start Frontend
    if [ -f "$FRONTEND_PID_FILE" ] && kill -0 $(cat "$FRONTEND_PID_FILE") 2>/dev/null; then
        echo "Frontend is already running (PID: $(cat $FRONTEND_PID_FILE))"
    else
        echo "Starting Frontend on port $FRONTEND_PORT..."
        cd "$FRONTEND_DIR" || exit
        nohup npm run dev -- --host --port $FRONTEND_PORT > "../$FRONTEND_LOG" 2>&1 &
        echo $! > "../$FRONTEND_PID_FILE"
        cd ..
        echo "Frontend started (PID: $(cat $FRONTEND_PID_FILE))"
    fi

    echo -e "${GREEN}WebUI started in background.${NC}"
    echo "Logs available at: $BACKEND_LOG, $FRONTEND_LOG"
}

stop() {
    echo -e "${RED}Stopping WebUI...${NC}"

    # Stop Backend
    if [ -f "$BACKEND_PID_FILE" ]; then
        PID=$(cat "$BACKEND_PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            echo "Backend stopped (PID: $PID)"
        else
            echo "Backend process $PID not found."
        fi
        rm "$BACKEND_PID_FILE"
    else
        echo "Backend PID file not found."
    fi

    # Stop Frontend
    if [ -f "$FRONTEND_PID_FILE" ]; then
        PID=$(cat "$FRONTEND_PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Stopping Frontend (PID: $PID)..."
            
            # Recursive function to kill descendants
            kill_descendants() {
                local p=$1
                local children=$(pgrep -P "$p")
                for child in $children; do
                    kill_descendants "$child"
                done
                if [ "$p" != "$PID" ]; then
                    kill "$p" 2>/dev/null
                fi
            }
            
            # Kill descendants first
            kill_descendants "$PID"
            
            # Kill the main process
            kill "$PID"
            echo "Frontend stopped."
        else
            echo "Frontend process $PID not found."
        fi
        rm "$FRONTEND_PID_FILE"
    else
        echo "Frontend PID file not found."
    fi

    # Fallback: Check if port is still in use and offer to kill?
    # Or just kill it if we are sure?
    # Let's check the port provided (default 3000 or --port arg)
    if lsof -i :"$FRONTEND_PORT" -t >/dev/null 2>&1; then
        echo "Warning: Port $FRONTEND_PORT is still in use."
        PID=$(lsof -i :"$FRONTEND_PORT" -t | head -n 1)
        echo "Killing orphaned process on port $FRONTEND_PORT (PID: $PID)..."
        kill -9 "$PID" 2>/dev/null
        echo "Port $FRONTEND_PORT cleared."
    fi
}

case "$ACTION" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        stop
        sleep 2
        start
        ;;
    *)
        usage
        ;;
esac
