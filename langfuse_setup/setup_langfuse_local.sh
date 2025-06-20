#!/bin/bash

# Langfuse Local Setup Script
# This script sets up Langfuse to run locally using Docker

echo "🚀 Setting up Langfuse locally..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose.yml exists
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ docker-compose.yml not found in current directory."
    echo "Please run this script from the langfuse_setup directory."
    exit 1
fi

# Start Langfuse
echo "🔧 Starting Langfuse..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo "✅ Langfuse is running!"
    echo ""
    echo "🌐 Dashboard: http://localhost:3000"
    echo "🔑 Public Key: pk-lf-1234567890abcdef"
    echo "🔐 Secret Key: sk-lf-1234567890abcdef"
    echo ""
    echo "📝 Add these to your .env file:"
    echo "LANGFUSE_PUBLIC_KEY=pk-lf-1234567890abcdef"
    echo "LANGFUSE_SECRET_KEY=sk-lf-1234567890abcdef"
    echo "LANGFUSE_HOST=http://localhost:3000"
    echo ""
    echo "🛑 To stop Langfuse, run: docker-compose down"
    echo "📊 To view logs, run: docker-compose logs -f"
else
    echo "❌ Failed to start Langfuse. Check the logs with: docker-compose logs"
    exit 1
fi 