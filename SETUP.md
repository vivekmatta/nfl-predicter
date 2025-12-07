# Setup Instructions

## Quick Setup for Local Development

### 1. Install Dependencies

```bash
# Install Node.js dependencies
npm install

# Install Python dependencies
pip install flask flask-cors pandas numpy scikit-learn requests kagglehub
```

### 2. Set Up Kaggle (Optional - for model refresh)

1. Go to [Kaggle Account Settings](https://www.kaggle.com/settings)
2. Scroll to "API" section
3. Click "Create New Token"
4. Download `kaggle.json`
5. Place it in `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<username>\.kaggle\kaggle.json` (Windows)

### 3. Run the Application

**Terminal 1 - Start Flask Backend:**
```bash
python app.py
```
Backend will run on http://localhost:5000

**Terminal 2 - Start React Frontend:**
```bash
npm run dev
```
Frontend will run on http://localhost:3000

### 4. Open in Browser

Navigate to: **http://localhost:3000**

## Project Structure

```
nfl-predicter/
├── app.py                 # Flask backend API
├── pages/
│   ├── index.tsx          # Main React frontend
│   └── _app.tsx           # Next.js app wrapper
├── styles/
│   └── globals.css        # Global styles
├── requirements.txt       # Python dependencies
├── package.json           # Node.js dependencies
└── spreadspoke_scores.csv # Training data (optional)
```

## Features

✅ **Game Prediction**: Enter game details and get ML-powered predictions  
✅ **Live Weather**: Automatic weather fetching for game locations  
✅ **Model Refresh**: Download latest data from Kaggle and retrain model  
✅ **Modern UI**: Beautiful, responsive React interface  

## Troubleshooting

### Backend not starting
- Make sure port 5000 is available
- Check Python dependencies are installed: `pip install -r requirements.txt`

### Frontend not connecting
- Ensure Flask backend is running on port 5000
- Check browser console for CORS errors
- Make sure both servers are running

### Model refresh fails
- Verify Kaggle credentials are set up correctly
- Check internet connection
- Dataset `tonycorona/nfl-spreadspoke-scores` must be accessible

### "Model not trained" error
- The model trains automatically on startup from `spreadspoke_scores.csv`
- If file doesn't exist, click "🔄 Refresh Model" to download from Kaggle
- Wait for training to complete (2-5 minutes)

## Environment Variables (Optional)

- `WEATHERAPI_KEY` - WeatherAPI.com key (default key included in code)
- `KAGGLE_USERNAME` - Your Kaggle username
- `KAGGLE_KEY` - Your Kaggle API key

## Next Steps

1. Start both servers (Flask + React)
2. Open http://localhost:3000
3. Select teams and make predictions!
4. Click "Refresh Model" to get latest data from Kaggle

