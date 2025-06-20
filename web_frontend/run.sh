#!/bin/bash

# Web Frontend Runner Script for Ollama LangChain Playground

echo "🚀 Starting Ollama LangChain Playground Web Frontend..."

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Please run this script from the web_frontend directory."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "../venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    cd ..
    python3 -m venv venv
    cd web_frontend
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source ../venv/bin/activate

# Install dependencies if needed
if [ ! -f ".dependencies_installed" ]; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
    touch .dependencies_installed
    echo "✅ Dependencies installed successfully!"
else
    echo "✅ Dependencies already installed."
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from template..."
    if [ -f "../env.example" ]; then
        cp ../env.example .env
        echo "✅ .env file created from template."
        echo "📝 Please edit .env file with your configuration."
    else
        echo "❌ env.example not found. Please create .env file manually."
    fi
fi

# Check if Ollama is running (optional)
if ! pgrep -x "ollama" > /dev/null; then
    echo "⚠️  Ollama is not running. If you plan to use Ollama models, start it with: ollama serve"
fi

# Check if Langfuse is running (optional)
if ! curl -s http://localhost:3000 > /dev/null 2>&1; then
    echo "⚠️  Langfuse is not running. To enable observability, start it with: cd ../langfuse_setup && ./manage_langfuse.sh start"
fi

echo ""
echo "🎯 Starting Streamlit application..."
echo "📱 The web interface will be available at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the application"
echo ""

# Start Streamlit
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 