from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import kagglehub

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# =========================================================
# Your existing code (config, cleaning, training functions)
# =========================================================

BASE_DIR = Path(__file__).parent
RAW_CSV = BASE_DIR / "spreadspoke_scores.csv"

WEATHER_API_KEY = os.getenv("WEATHERAPI_KEY", "e685133506aa435ab9252438250612")
WEATHER_BASE_URL = "http://api.weatherapi.com/v1/current.json"

TEAM_ABBR_TO_CITY = {
    "ARI": "Glendale,AZ", "ATL": "Atlanta,GA", "BAL": "Baltimore,MD",
    "BUF": "Buffalo,NY", "CAR": "Charlotte,NC", "CHI": "Chicago,IL",
    "CIN": "Cincinnati,OH", "CLE": "Cleveland,OH", "DAL": "Dallas,TX",
    "DEN": "Denver,CO", "DET": "Detroit,MI", "GB": "Green Bay,WI",
    "HOU": "Houston,TX", "IND": "Indianapolis,IN", "JAX": "Jacksonville,FL",
    "KC": "Kansas City,MO", "LAC": "Inglewood,CA", "LAR": "Inglewood,CA",
    "LV": "Las Vegas,NV", "MIA": "Miami,FL", "MIN": "Minneapolis,MN",
    "NE": "Foxborough,MA", "NO": "New Orleans,LA", "NYG": "East Rutherford,NJ",
    "NYJ": "East Rutherford,NJ", "PHI": "Philadelphia,PA", "PIT": "Pittsburgh,PA",
    "SEA": "Seattle,WA", "SF": "Santa Clara,CA", "TB": "Tampa,FL",
    "TEN": "Nashville,TN", "WAS": "Landover,MD",
}

DOME_TEAMS = {"ATL", "DET", "MIN", "NO", "DAL", "HOU", "IND", "ARI", "LV", "LAC", "LAR"}

# Copy all your helper functions here (week_to_num, clean_for_model, etc.)
def week_to_num(x):
    try:
        return int(x)
    except Exception:
        mapping = {
            "Wildcard": 18, "Division": 19, "Conference": 20,
            "Superbowl": 21, "Super Bowl": 21, "SB": 21,
        }
        return mapping.get(str(x), 0)

def clean_for_model(df):
    df = df.copy()
    df = df.dropna(subset=["score_home", "score_away"])
    df["point_diff"] = df["score_home"] - df["score_away"]
    df["home_win"] = (df["point_diff"] > 0).astype(int)
    df["schedule_date_dt"] = pd.to_datetime(df["schedule_date"], errors="coerce")
    df["game_month"] = df["schedule_date_dt"].dt.month
    df["game_dow"] = df["schedule_date_dt"].dt.weekday
    df["schedule_week_num"] = df["schedule_week"].apply(week_to_num)
    df["is_playoff"] = df["schedule_playoff"].astype(int)
    
    teams = sorted(set(df["team_home"].unique()) | set(df["team_away"].unique()))
    team_to_id = {name: i + 1 for i, name in enumerate(teams)}
    df["home_team_id"] = df["team_home"].map(team_to_id)
    df["away_team_id"] = df["team_away"].map(team_to_id)
    
    fav_vals = sorted(v for v in df["team_favorite_id"].dropna().unique() if v not in ["PICK", "PK"])
    fav_to_id = {name: i + 1 for i, name in enumerate(fav_vals)}
    df["favorite_team_id"] = df["team_favorite_id"].map(fav_to_id)
    df.loc[df["team_favorite_id"].isin(["PICK", "PK"]), "favorite_team_id"] = 0
    df["favorite_team_id"] = df["favorite_team_id"].fillna(0).astype(int)
    
    for col in ["spread_favorite", "over_under_line", "weather_temperature", "weather_wind_mph", "weather_humidity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df["fav_implied_pts"] = np.where(
        df["spread_favorite"].notna() & df["over_under_line"].notna(),
        df["over_under_line"] / 2 - df["spread_favorite"] / 2, np.nan,
    )
    df["dog_implied_pts"] = np.where(
        df["spread_favorite"].notna() & df["over_under_line"].notna(),
        df["over_under_line"] / 2 + df["spread_favorite"] / 2, np.nan,
    )
    df["abs_spread"] = df["spread_favorite"].abs()
    
    stadiums = sorted(df["stadium"].dropna().unique())
    stadium_to_id = {name: i + 1 for i, name in enumerate(stadiums)}
    df["stadium_id"] = df["stadium"].map(stadium_to_id).fillna(0).astype(int)
    df["stadium_neutral"] = df["stadium_neutral"].astype(int)
    
    wd = df["weather_detail"].fillna("").str.lower()
    df["is_indoor"] = (wd.str.contains("indoor") | wd.str.contains("dome")).astype(int)
    df["is_rain"] = wd.str.contains("rain").astype(int)
    df["is_snow"] = wd.str.contains("snow").astype(int)
    
    cols = [
        "schedule_season", "schedule_week_num", "is_playoff", "game_month", "game_dow",
        "home_team_id", "away_team_id", "favorite_team_id", "spread_favorite",
        "abs_spread", "over_under_line", "fav_implied_pts", "dog_implied_pts",
        "stadium_id", "stadium_neutral", "weather_temperature", "weather_wind_mph",
        "weather_humidity", "is_indoor", "is_rain", "is_snow", "home_win",
    ]
    
    cleaned = df[cols].copy()
    for col in cleaned.columns:
        if cleaned[col].dtype.kind in "biufc":
            median_val = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(median_val)
    
    return cleaned, team_to_id, stadium_to_id

def train_models(cleaned):
    X = cleaned.drop(columns=["home_win"])
    y = cleaned["home_win"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logreg = LogisticRegression(max_iter=5000, n_jobs=-1, solver="lbfgs")
    logreg.fit(X_train_scaled, y_train)
    
    rf = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=10,
                                min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    
    from sklearn.metrics import accuracy_score
    acc_lr = accuracy_score(y_test, logreg.predict(X_test_scaled))
    acc_rf = accuracy_score(y_test, rf.predict(X_test))
    
    if acc_rf >= acc_lr:
        return rf, X.columns.tolist(), None, "RandomForest"
    else:
        return logreg, X.columns.tolist(), scaler, "LogisticRegression"

def fetch_weather_for_city(city_q):
    params = {"key": WEATHER_API_KEY, "q": city_q, "aqi": "no"}
    resp = requests.get(WEATHER_BASE_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and "error" in data:
        err = data["error"]
        raise RuntimeError(f"WeatherAPI error {err.get('code')}: {err.get('message')}")
    cur = data["current"]
    cond = cur.get("condition", {})
    return {
        "temp_f": cur.get("temp_f"),
        "wind_mph": cur.get("wind_mph"),
        "humidity": cur.get("humidity"),
        "condition_text": cond.get("text", ""),
    }

def infer_rain_snow_from_condition(condition_text):
    text = (condition_text or "").lower()
    is_rain = int("rain" in text or "drizzle" in text or "shower" in text or "thunderstorm" in text)
    is_snow = int("snow" in text or "sleet" in text or "blizzard" in text or "flurries" in text)
    return is_rain, is_snow

# =========================================================
# Load and train model on startup
# =========================================================

print("Loading data and training model...")
raw_df = pd.read_csv(RAW_CSV)
cleaned, team_to_id, stadium_to_id = clean_for_model(raw_df)
best_model, feature_order, scaler_for_best, best_name = train_models(cleaned)

# Create abbreviation to full name mapping
abbr_to_full = {}
for idx, row in raw_df[["team_favorite_id", "team_home"]].dropna().drop_duplicates().iterrows():
    abbr_to_full[row["team_favorite_id"]] = row["team_home"]

fav_abbrs = sorted(v for v in raw_df["team_favorite_id"].dropna().unique() if v not in ["PICK", "PK"])
favorite_abbr_to_id = {abbr: i + 1 for i, abbr in enumerate(fav_abbrs)}

print(f"Model trained: {best_name}")

# =========================================================
# API Endpoints
# =========================================================

@app.route('/api/weather', methods=['GET'])
def get_weather():
    """Fetch live weather for a team's city"""
    team_abbr = request.args.get('team', '').upper()
    
    if team_abbr not in TEAM_ABBR_TO_CITY:
        return jsonify({"error": "Unknown team abbreviation"}), 400
    
    city_q = TEAM_ABBR_TO_CITY[team_abbr]
    
    try:
        weather = fetch_weather_for_city(city_q)
        is_rain, is_snow = infer_rain_snow_from_condition(weather["condition_text"])
        
        return jsonify({
            "temperature": weather["temp_f"],
            "windSpeed": weather["wind_mph"],
            "humidity": weather["humidity"],
            "condition": weather["condition_text"],
            "isRain": is_rain,
            "isSnow": is_snow,
            "isIndoor": 1 if team_abbr in DOME_TEAMS else 0
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/refresh_model', methods=['POST'])
def refresh_model():
    """Download latest CSV from Kaggle and retrain model"""
    global best_model, feature_order, scaler_for_best, best_name, team_to_id, favorite_abbr_to_id, abbr_to_full
    
    csv_path = None
    source = "local"
    
    # First, try to download from Kaggle
    try:
        import kagglehub
        
        print("Attempting to download dataset from Kaggle...")
        
        # Try to authenticate with Kaggle
        # First check for API token environment variable (new method)
        if os.getenv("KAGGLE_API_TOKEN"):
            print("Using KAGGLE_API_TOKEN for authentication...")
            # kagglehub will automatically use the environment variable
        else:
            # Try to login using kagglehub (will use kaggle.json if available)
            try:
                kagglehub.login()
                print("Authenticated using kagglehub.login()")
            except Exception as auth_error:
                print(f"Authentication attempt failed: {auth_error}")
                # Continue anyway - kagglehub might work without explicit login
        
        dataset_path = kagglehub.dataset_download("tonycorona/nfl-spreadspoke-scores")
        
        # Try to find the CSV file
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.csv') and 'spreadspoke' in file.lower():
                    csv_path = os.path.join(root, file)
                    break
            if csv_path:
                break
        
        if not csv_path:
            # Try common paths
            possible_paths = [
                os.path.join(dataset_path, "spreadspoke_scores.csv"),
                os.path.join(dataset_path, "nfl_spreadspoke_scores.csv"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    csv_path = path
                    break
        
        if csv_path:
            source = "kaggle"
            print(f"Successfully downloaded from Kaggle: {csv_path}")
        
    except Exception as kaggle_error:
        error_str = str(kaggle_error)
        print(f"Kaggle download failed: {error_str}")
        
        # Check if it's an authentication error
        if "403" in error_str or "permission" in error_str.lower() or "authenticated" in error_str.lower() or "401" in error_str:
            # Fall back to local CSV if available
            if os.path.exists(RAW_CSV):
                csv_path = RAW_CSV
                source = "local (Kaggle auth failed)"
                print(f"Using local CSV file: {csv_path}")
            else:
                return jsonify({
                    "success": False,
                    "error": (
                        "Kaggle authentication failed. Please set up Kaggle API token:\n\n"
                        "Method 1 (Recommended - API Token):\n"
                        "1. Go to https://www.kaggle.com/settings\n"
                        "2. Scroll to 'API' section and click 'Create New Token'\n"
                        "3. Copy the API token (starts with KGAT_)\n"
                        "4. Set environment variable:\n"
                        "   - Windows: set KAGGLE_API_TOKEN=your_token_here\n"
                        "   - Linux/Mac: export KAGGLE_API_TOKEN=your_token_here\n\n"
                        "Method 2 (Legacy - kaggle.json file):\n"
                        "1. Download kaggle.json from Kaggle settings\n"
                        "2. Place it in:\n"
                        "   - Linux/Mac: ~/.kaggle/kaggle.json\n"
                        "   - Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json\n\n"
                        "Alternatively, place spreadspoke_scores.csv in the project directory."
                    ),
                }), 500
        else:
            # Other error - try local file as fallback
            if os.path.exists(RAW_CSV):
                csv_path = RAW_CSV
                source = "local (fallback)"
                print(f"Kaggle error, using local CSV: {csv_path}")
            else:
                return jsonify({
                    "success": False,
                    "error": f"Kaggle download failed: {error_str}\n\nNo local CSV file found. Please ensure spreadspoke_scores.csv exists in the project directory.",
                }), 500
    
    # If Kaggle failed, try local file
    if not csv_path and os.path.exists(RAW_CSV):
        csv_path = RAW_CSV
        source = "local"
        print(f"Using local CSV file: {csv_path}")
    
    if not csv_path:
        return jsonify({
            "success": False,
            "error": (
                "No data source available. Please either:\n\n"
                "1. Set up Kaggle API credentials (see instructions above), OR\n"
                "2. Place spreadspoke_scores.csv in the project directory"
            ),
        }), 500
    
    # Load and train model
    try:
        print(f"Loading CSV from: {csv_path} (source: {source})")
        raw_df = pd.read_csv(csv_path)
        
        print("Cleaning data...")
        cleaned, team_to_id, stadium_to_id = clean_for_model(raw_df)
        
        print("Training models...")
        best_model, feature_order, scaler_for_best, best_name = train_models(cleaned)
        
        # Update mappings
        abbr_to_full = {}
        for idx, row in raw_df[["team_favorite_id", "team_home"]].dropna().drop_duplicates().iterrows():
            abbr_to_full[row["team_favorite_id"]] = row["team_home"]
        
        fav_abbrs = sorted(v for v in raw_df["team_favorite_id"].dropna().unique() if v not in ["PICK", "PK"])
        favorite_abbr_to_id = {abbr: i + 1 for i, abbr in enumerate(fav_abbrs)}
        
        print(f"Model retrained: {best_name}")
        
        return jsonify({
            "success": True,
            "message": f"Model refreshed successfully from {source}",
            "model_name": best_name,
            "training_samples": len(cleaned),
            "source": source,
        })
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"Error training model: {error_msg}")
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": f"Error training model: {error_msg}",
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make a game prediction"""
    data = request.json
    
    try:
        # Map abbreviations to full names
        home_full = abbr_to_full.get(data['homeTeam'], data['homeTeam'])
        away_full = abbr_to_full.get(data['awayTeam'], data['awayTeam'])
        
        # Build feature vector
        home_id = team_to_id.get(home_full, 0)
        away_id = team_to_id.get(away_full, 0)
        fav_id = favorite_abbr_to_id.get(data.get('favoriteTeam', ''), 0)
        
        spread_fav = float(data['spread'])
        over_under = float(data['overUnder'])
        fav_implied = over_under / 2 - spread_fav / 2
        dog_implied = over_under / 2 + spread_fav / 2
        abs_spread = abs(spread_fav)
        
        feat = {
            "schedule_season": int(data['season']),
            "schedule_week_num": int(data['week']),
            "is_playoff": 1 if data['isPlayoff'] else 0,
            "game_month": int(data['month']),
            "game_dow": int(data['dayOfWeek']),
            "home_team_id": home_id,
            "away_team_id": away_id,
            "favorite_team_id": fav_id,
            "spread_favorite": spread_fav,
            "abs_spread": abs_spread,
            "over_under_line": over_under,
            "fav_implied_pts": fav_implied,
            "dog_implied_pts": dog_implied,
            "stadium_id": 0,
            "stadium_neutral": 0,
            "weather_temperature": float(data['temperature']),
            "weather_wind_mph": float(data['windSpeed']),
            "weather_humidity": float(data['humidity']),
            "is_indoor": 1 if data['isIndoor'] else 0,
            "is_rain": 1 if data['isRain'] else 0,
            "is_snow": 1 if data['isSnow'] else 0,
        }
        
        x_df = pd.DataFrame([feat], columns=feature_order)
        
        # Make prediction
        if scaler_for_best is not None:
            x_scaled = scaler_for_best.transform(x_df.values)
            proba = best_model.predict_proba(x_scaled)[0, 1]
        else:
            proba = best_model.predict_proba(x_df)[0, 1]
        
        predicted_home_win = proba >= 0.5
        winner = data['homeTeam'] if predicted_home_win else data['awayTeam']
        
        return jsonify({
            "winner": winner,
            "homeWinProb": float(proba),
            "model": best_name
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)