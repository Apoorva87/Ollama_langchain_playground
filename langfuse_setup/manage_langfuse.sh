#!/bin/bash

# Langfuse Management Script
# This script provides easy management of Langfuse services

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    print_error "docker-compose.yml not found. Please run this script from the langfuse_setup directory."
    exit 1
fi

# Function to check if Docker is running
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
}

# Function to check for port conflicts
check_ports() {
    print_info "Checking for port conflicts..."
    
    # Check port 3000 (Langfuse)
    if lsof -Pi :3000 -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_warning "Port 3000 is already in use. Langfuse dashboard may not start properly."
        print_info "You can change the port in docker-compose.yml if needed."
    fi
    
    # Check port 5433 (PostgreSQL)
    if lsof -Pi :5433 -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_warning "Port 5433 is already in use. PostgreSQL may not start properly."
        print_info "You can change the port in docker-compose.yml if needed."
    fi
}

# Function to start Langfuse
start_langfuse() {
    print_info "Starting Langfuse..."
    check_docker
    check_ports
    
    # Stop any existing containers first
    if docker-compose ps | grep -q "Up"; then
        print_info "Stopping existing containers..."
        docker-compose down
    fi
    
    docker-compose up -d
    
    # Wait for services to be ready
    print_info "Waiting for services to start..."
    sleep 15
    
    # Check if services are running
    if docker-compose ps | grep -q "Up"; then
        print_status "Langfuse started successfully!"
        print_info "Dashboard: http://localhost:3000"
        print_info "Public Key: pk-lf-1234567890abcdef"
        print_info "Secret Key: sk-lf-1234567890abcdef"
        print_info "PostgreSQL: localhost:5433"
    else
        print_error "Failed to start Langfuse. Check the logs:"
        docker-compose logs
        exit 1
    fi
}

# Function to stop Langfuse
stop_langfuse() {
    print_info "Stopping Langfuse..."
    docker-compose down
    print_status "Langfuse stopped successfully!"
}

# Function to restart Langfuse
restart_langfuse() {
    print_info "Restarting Langfuse..."
    docker-compose restart
    print_status "Langfuse restarted successfully!"
}

# Function to show status
show_status() {
    print_info "Langfuse Status:"
    if docker-compose ps | grep -q "Up"; then
        print_status "Langfuse is running"
        echo ""
        print_info "Services:"
        docker-compose ps
        echo ""
        print_info "Dashboard: http://localhost:3000"
        print_info "PostgreSQL: localhost:5433"
    else
        print_warning "Langfuse is not running"
        echo ""
        print_info "Run './manage_langfuse.sh start' to start Langfuse"
    fi
}

# Function to show logs
show_logs() {
    print_info "Showing Langfuse logs (Ctrl+C to exit)..."
    docker-compose logs -f
}

# Function to clean up (remove all data)
cleanup() {
    print_warning "This will remove all Langfuse data. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        print_info "Cleaning up Langfuse data..."
        docker-compose down -v
        print_status "Langfuse data cleaned up successfully!"
    else
        print_info "Cleanup cancelled."
    fi
}

# Function to show help
show_help() {
    echo "Langfuse Management Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start     Start Langfuse services"
    echo "  stop      Stop Langfuse services"
    echo "  restart   Restart Langfuse services"
    echo "  status    Show status of Langfuse services"
    echo "  logs      Show Langfuse logs (follow mode)"
    echo "  cleanup   Remove all Langfuse data (DESTRUCTIVE)"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start    # Start Langfuse"
    echo "  $0 status   # Check if Langfuse is running"
    echo "  $0 logs     # View logs"
    echo ""
    echo "Ports:"
    echo "  Langfuse Dashboard: http://localhost:3000"
    echo "  PostgreSQL: localhost:5433"
}

# Main script logic
case "${1:-help}" in
    start)
        start_langfuse
        ;;
    stop)
        stop_langfuse
        ;;
    restart)
        restart_langfuse
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac 