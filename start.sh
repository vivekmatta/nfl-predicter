#!/bin/bash

# Start Flask backend in background
echo "Starting Flask backend on port 5000..."
python app.py &
FLASK_PID=$!

# Wait a moment for Flask to start
sleep 2

# Start Next.js frontend
echo "Starting React frontend on port 3000..."
npm run dev

# When frontend stops, kill Flask
kill $FLASK_PID 2>/dev/null

