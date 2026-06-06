#!/bin/bash
set -e

PROJECT_DIR="$HOME/finmanalphacore-sandbox/FinManAlphaCore"
VENV_DIR="$PROJECT_DIR/.venv"
LOG_DIR="$PROJECT_DIR/logs"

cd "$PROJECT_DIR"

echo "===================================" >> "$LOG_DIR/daily_blog.log"
echo "AlphaCore Daily Blog Run: $(date)" >> "$LOG_DIR/daily_blog.log"
echo "===================================" >> "$LOG_DIR/daily_blog.log"

source "$VENV_DIR/bin/activate"

python generate_daily_blog.py >> "$LOG_DIR/daily_blog.log" 2>&1

echo "Completed: $(date)" >> "$LOG_DIR/daily_blog.log"
