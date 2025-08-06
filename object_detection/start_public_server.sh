#!/bin/bash

# Car Damage Analyzer - Public Server Startup Script
# This script starts both the WebSocket server and HTTP server for public access

echo "🚀 Car Damage Analyzer - Public Server Setup"
echo "============================================="

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

# Start HTTP server on port 80 (requires sudo)
echo "Starting HTTP server on port 80 (requires sudo)..."
echo "You may be prompted for your password."

sudo python3 -m http.server 80 --bind 0.0.0.0 &
HTTP_PID=$!

echo ""
echo "✅ Services started successfully!"
echo ""
echo "🌐 Access URLs:"
echo "   - Web Interface: http://$CURRENT_IP/"
echo "   - Local Access:  http://localhost/"
echo "   - WebSocket:     ws://$CURRENT_IP:8765"
echo ""
echo "📋 File served:"
echo "   - Main App: http://$CURRENT_IP/car_damage_analyzer.html"
echo ""
echo "🔧 Process IDs:"
echo "   - WebSocket Server PID: $WEBSOCKET_PID"
echo "   - HTTP Server PID: $HTTP_PID"
echo ""
echo "⏹️  To stop servers:"
echo "   sudo kill $HTTP_PID"
echo "   kill $WEBSOCKET_PID"
echo ""
echo "📝 Logs:"
echo "   - WebSocket logs will appear below"
echo "   - HTTP access logs will appear in terminal"
echo ""
echo "⚠️  Note: Make sure your firewall allows ports 80 and 8765"
echo ""

# Keep script running and show WebSocket logs
wait $WEBSOCKET_PID
