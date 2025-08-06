#!/bin/bash

# Car Damage Analyzer - Public Server (Port 8080, No Sudo Required)
# Alternative startup script that doesn't require root privileges

echo "🚀 Car Damage Analyzer - Public Server (Port 8080)"
echo "=================================================="

# Get current IP address  
CURRENT_IP=$(ip route get 1.1.1.1 | awk '{print $7; exit}')
echo "📍 Public IP Address: $CURRENT_IP"

# Change to the correct directory
cd /home/user/gemma-3n/gemma3n-car-damage-competition/object_detection

echo ""
echo "🔧 Starting services..."

# Start WebSocket server in background
echo "Starting WebSocket server (port 8765)..."
python3 unified_websocket_server.py &
WEBSOCKET_PID=$!

# Wait a moment for WebSocket to start
sleep 3

# Start HTTP server on port 8080 (no sudo required)
echo "Starting HTTP server on port 8080..."

python3 -m http.server 8080 --bind 0.0.0.0 &
HTTP_PID=$!

echo ""
echo "✅ Services started successfully!"
echo ""
echo "🌐 Access URLs:"
echo "   - Web Interface: http://$CURRENT_IP:8080/"
echo "   - Local Access:  http://localhost:8080/"
echo "   - WebSocket:     ws://$CURRENT_IP:8765"
echo ""
echo "📋 File served:"
echo "   - Main App: http://$CURRENT_IP:8080/car_damage_analyzer.html"
echo ""
echo "🔧 Process IDs:"
echo "   - WebSocket Server PID: $WEBSOCKET_PID"
echo "   - HTTP Server PID: $HTTP_PID"
echo ""
echo "⏹️  To stop servers:"
echo "   kill $HTTP_PID"
echo "   kill $WEBSOCKET_PID"
echo ""
echo "📝 Press Ctrl+C to stop both servers"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $HTTP_PID 2>/dev/null
    kill $WEBSOCKET_PID 2>/dev/null
    echo "✅ Servers stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Keep script running
wait
