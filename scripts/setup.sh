#!/bin/bash
# Setup script for Wave Forecast development environment

set -e  # Exit on error

echo "🌊 Setting up Wave Forecast development environment..."

# Check prerequisites
echo "Checking prerequisites..."
command -v python3 >/dev/null 2>&1 || { echo "❌ Python 3 is required but not installed."; exit 1; }
command -v node >/dev/null 2>&1 || { echo "❌ Node.js is required but not installed."; exit 1; }
command -v pnpm >/dev/null 2>&1 || { echo "❌ pnpm is required. Install with: npm install -g pnpm"; exit 1; }

echo "✅ Prerequisites check passed"

# Setup Python environment
echo ""
echo "Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Created virtual environment"
else
    echo "✅ Virtual environment already exists"
fi

source venv/bin/activate

# Install Python packages
echo ""
echo "Installing Python packages..."
pip install --upgrade pip
pip install -e packages/python/common
pip install -e backend/api
pip install -e backend/worker
pip install -e ml/inference
pip install -r ml/training/requirements.txt
pip install -r data/pipelines/requirements.txt

echo "✅ Python packages installed"

# Setup Node.js environment
echo ""
echo "Installing Node.js packages..."
pnpm install

echo "✅ Node.js packages installed"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✅ Created .env file - please update with your credentials"
else
    echo "✅ .env file already exists"
fi

# Create necessary directories
echo ""
echo "Creating data directories..."
mkdir -p data/raw data/processed ml/artifacts ml/experiments

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Update .env with your API keys and credentials"
echo "  2. Start the backend: cd backend/api && uvicorn app.main:app --reload"
echo "  3. Start the web app: pnpm dev:web"
echo "  4. Start the mobile app: pnpm dev:mobile"
