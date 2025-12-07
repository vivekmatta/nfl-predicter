# 🏈 NFL Game Predictor

A clean, modern React web application for predicting NFL game outcomes using machine learning.

## Features

✅ **Game Prediction** - Predict NFL game winners with probability scores  
✅ **Live Weather Integration** - Automatic weather fetching for game locations  
✅ **Model Refresh** - Download latest data from Kaggle and retrain model  
✅ **Modern UI** - Beautiful, responsive React interface  
✅ **Easy to Run** - Simple local setup  

## Quick Start

### 1. Install Dependencies

**Python (Backend):**
```bash
pip install -r requirements.txt
```

**Node.js (Frontend):**
```bash
npm install
```

### 2. Set Up Kaggle (Optional - for model refresh feature)

**Recommended Method - API Token:**
1. Go to [Kaggle Account Settings](https://www.kaggle.com/settings)
2. Scroll to "API" section and click "Create New Token"
3. Copy the API token (starts with `KGAT_`)
4. Set environment variable:
   - **Windows**: `set KAGGLE_API_TOKEN=your_token_here`
   - **Linux/Mac**: `export KAGGLE_API_TOKEN=your_token_here`

See `KAGGLE_SETUP.md` for detailed instructions.

### 3. Run the Application

You need **two terminal windows**:

**Terminal 1 - Start Flask Backend:**
```bash
python app.py
```
You should see: `Running on http://127.0.0.1:5000`

**Terminal 2 - Start React Frontend:**
```bash
npm run dev
```
You should see: `Ready on http://localhost:3000`

### 4. Open in Browser

Navigate to: **http://localhost:3000**

**Note**: The model trains automatically on startup from `spreadspoke_scores.csv`. If the file doesn't exist, click "🔄 Refresh Model" to download from Kaggle.

## Usage

1. **Select Teams**: Choose home and away teams from dropdowns
2. **Weather**: Weather automatically fetches when you select a home team
3. **Enter Game Details**: Season, week, spread, over/under, etc.
4. **Make Prediction**: Click "Make Prediction" to get ML-powered prediction
5. **Refresh Model**: Click "🔄 Refresh Model" to download latest data from Kaggle and retrain

## Project Structure

```
nfl-predicter/
├── app.py                 # Flask backend API
├── pages/
│   ├── index.tsx         # Main React frontend
│   └── _app.tsx          # Next.js app wrapper
├── styles/
│   └── globals.css       # Global styles
├── package.json          # Node.js dependencies
├── requirements.txt     # Python dependencies
└── spreadspoke_scores.csv # Training data (optional)
```

## API Endpoints

- `GET /api/weather?team=TEAM_ABBR` - Fetch weather for team city
- `POST /api/predict` - Make game prediction
- `POST /api/refresh_model` - Download from Kaggle and retrain model

## Environment Variables (Optional)

- `WEATHERAPI_KEY` - WeatherAPI.com key (default key included)
- `KAGGLE_USERNAME` - Your Kaggle username
- `KAGGLE_KEY` - Your Kaggle API key

## Troubleshooting

**Backend not starting:**
- Make sure port 5000 is available
- Check Python dependencies are installed

**Frontend not connecting:**
- Ensure Flask backend is running on port 5000
- Check browser console for CORS errors

**Model refresh fails:**
- If you see a "403" or "permission" error, you need to set up Kaggle API credentials
- See `KAGGLE_SETUP.md` for detailed instructions
- **Alternative**: Place `spreadspoke_scores.csv` in the project directory - the app will automatically use it as a fallback
- The app will use the local CSV file if Kaggle authentication fails

## Technologies

- **Frontend**: React, Next.js, TypeScript
- **Backend**: Flask, Python
- **ML**: Scikit-learn (Random Forest, Logistic Regression)
- **Weather**: WeatherAPI.com
- **Data**: Kaggle (tonycorona/nfl-spreadspoke-scores)

## License

MIT
