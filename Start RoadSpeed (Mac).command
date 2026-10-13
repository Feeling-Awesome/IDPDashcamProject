#!/bin/bash
# Road Speed Grid for macOS - double-click to start (first run installs what it needs).
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Setting up Road Speed Grid (first run only, needs internet)..."
  if ! command -v python3 >/dev/null || ! python3 -c "import tkinter" 2>/dev/null; then
    echo "Please install Python from https://www.python.org/downloads/ and run this again."
    open "https://www.python.org/downloads/"; read -n 1; exit 1
  fi
  python3 -m venv .venv && .venv/bin/python -m pip install -q -r requirements.txt || { echo "Setup failed"; read -n 1; exit 1; }
fi
nohup .venv/bin/python gui.py "$@" >/dev/null 2>&1 &
