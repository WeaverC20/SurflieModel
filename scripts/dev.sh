#!/bin/bash
# Development helper script

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

function help() {
    echo "Wave Forecast Development Helper"
    echo ""
    echo "Usage: ./scripts/dev.sh [command]"
    echo ""
    echo "Commands:"
    echo "  web       - Start Next.js web app"
    echo "  mobile    - Start Expo mobile app"
    echo "  api       - Start FastAPI backend"
    echo "  worker    - Start background worker"
    echo "  all       - Start all services (requires tmux)"
    echo "  test      - Run all tests"
    echo "  lint      - Run all linters"
    echo "  format    - Format all code"
    echo ""
}

function start_web() {
    echo -e "${GREEN}Starting web app...${NC}"
    pnpm dev:web
}

function start_mobile() {
    echo -e "${GREEN}Starting mobile app...${NC}"
    pnpm dev:mobile
}

function start_api() {
    echo -e "${GREEN}Starting API...${NC}"
    source venv/bin/activate
    cd backend/api
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
}

function start_worker() {
    echo -e "${GREEN}Starting worker...${NC}"
    source venv/bin/activate
    cd backend/worker
    python scheduler.py
}

function run_tests() {
    echo -e "${GREEN}Running tests...${NC}"
    source venv/bin/activate
    pytest backend/ ml/ packages/python/ -v
    pnpm test
}

function run_lint() {
    echo -e "${GREEN}Running linters...${NC}"
    source venv/bin/activate
    black --check backend/ ml/ packages/python/
    isort --check-only backend/ ml/ packages/python/
    pnpm lint
}

function run_format() {
    echo -e "${GREEN}Formatting code...${NC}"
    source venv/bin/activate
    black backend/ ml/ packages/python/
    isort backend/ ml/ packages/python/
    pnpm prettier --write "apps/**/*.{ts,tsx,js,jsx,json,css,md}"
}

# Parse command
case "$1" in
    web)
        start_web
        ;;
    mobile)
        start_mobile
        ;;
    api)
        start_api
        ;;
    worker)
        start_worker
        ;;
    test)
        run_tests
        ;;
    lint)
        run_lint
        ;;
    format)
        run_format
        ;;
    *)
        help
        ;;
esac
