@echo off
echo Starting Flask backend on port 5000...
start "Flask Backend" cmd /k "python app.py"
timeout /t 3 /nobreak >nul
echo Starting React frontend on port 3000...
npm run dev

