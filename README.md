# 🏈 NFL-PredictER

A machine learning project that predicts NFL game outcomes (home win probability) using:

- Historical game + betting data (spread, O/U)
- Weather conditions (live WeatherAPI integration)
- Team IDs and matchup context
- Game metadata (week, month, playoff flag)
- Logistic Regression (scaled)
- Random Forest (raw features)
- Automatic model selection (best model by accuracy)

You can enter any matchup and get:

- Predicted winner  
- Home win probability  
- Spread & O/U context  
- Weather summary  
- Full game context  

---

## 🚀 Features

### ✔ Automatic Data Cleaning

The script loads `spreadspoke_scores.csv`, cleans it, and generates features such as:

- Week number (regular season + playoffs)
- Home/away team IDs
- Favorite team ID
- Point spread & implied team points
- Over/under total
- Weather features (temperature, wind, humidity, rain/snow flags)
- Game metadata (season, month, day of week, playoff flag)
- Stadium ID and neutral-field flag

The target label is:

- `home_win` — 1 if home team won, 0 otherwise

---

### ✔ Two ML Models + Automatic Selection

The script trains and compares:

1. **Logistic Regression**
   - Uses `StandardScaler` (scaled features)
   - Higher `max_iter` for cleaner convergence
2. **Random Forest**
   - Uses raw numeric features
   - Typically performs better on this dataset

After training, it prints a comparison table and automatically chooses the best model (by accuracy). That model is then used for predictions in the interactive UI.

---

## 🌤 Live WeatherAPI Integration

The project integrates with [WeatherAPI.com](https://www.weatherapi.com/) to automatically pull **current weather** for the home team’s city.

When you enter a matchup:

1. Home team abbreviation → mapped to a city (e.g., `DET` → `Detroit,MI`)
2. The script calls the WeatherAPI `/current.json` endpoint
3. It fills:
   - Temperature (°F)
   - Wind speed (mph)
   - Humidity (%)
   - Rain / snow flags (derived from the text conditions)
4. Indoor teams (dome stadiums) default to `"y"` for indoor games, but you can override

If the API call fails, the script falls back to manual input for weather fields.

---

## 🧠 Interactive Prediction UI

After training, the script prompts you to enter game details:

- Home team (abbreviation or full name)
- Away team (abbreviation or full name)
- Season (year)
- Week number
- Whether it’s a playoff game
- Spread for the favorite (home negative, away positive)
- Over/under total points
- Favorite team abbreviation (optional; defaults to home)
- Weather (auto-fetched, but you can override)
- Indoor / rain / snow flags

Then it prints a clean, formatted summary, for example:

```text
==================== PREDICTION RESULT ====================
Model used : RandomForest
Matchup    : DET (Detroit Lions)  vs  DAL (Dallas Cowboys)
Season     : 2025   Week: 14   Playoff: No
Spread     : -4.5  (favorite: DET)
O/U Total  : 42.0
Weather    : 28.0 F, 10.3 mph wind, 69.0% humidity
Conditions : Indoor, No rain, No snow
-----------------------------------------------------------
🏈 Predicted winner : DET (Detroit Lions)
🏠 Home win prob    : 53.221%
===========================================================
