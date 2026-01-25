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
    echo "Usage: ./webui.sh {start|stop|restart} [--venv /path/to/venv]"
    exit 1
}

if [ -z "$1" ]; then
    usage
fi

ACTION="$1"
shift

# Parse Arguments
VENV_PATH=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --venv) VENV_PATH="$2"; shift ;;
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
        nohup python main.py > "../$BACKEND_LOG" 2>&1 &
        echo $! > "../$BACKEND_PID_FILE"
        cd ..
        echo "Backend started (PID: $(cat $BACKEND_PID_FILE))"
    fi

    # 3. Start Frontend
    if [ -f "$FRONTEND_PID_FILE" ] && kill -0 $(cat "$FRONTEND_PID_FILE") 2>/dev/null; then
        echo "Frontend is already running (PID: $(cat $FRONTEND_PID_FILE))"
    else
        echo "Starting Frontend..."
        cd "$FRONTEND_DIR" || exit
        nohup npm run dev -- --host > "../$FRONTEND_LOG" 2>&1 &
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
        # For npm, it might spawn child processes (vite).
        # We might need to kill the process group or hope vite dies with parent.
        # Usually npm run dev spawns vite. PID is npm or vite.
        if kill -0 "$PID" 2>/dev/null; then
            # Try to kill process group to ensure children die
            pkill -P "$PID" 2>/dev/null
            kill "$PID"
            echo "Frontend stopped (PID: $PID)"
        else
            echo "Frontend process $PID not found."
        fi
        rm "$FRONTEND_PID_FILE"
    else
        echo "Frontend PID file not found."
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
