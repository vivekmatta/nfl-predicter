import os
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
import requests

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =========================================================
# CONFIG: paths
# =========================================================

BASE_DIR = Path(r"C:\Users\Vhas21\Desktop\Northwestern\Projects\nfl-predicter")
RAW_CSV = BASE_DIR / "spreadspoke_scores.csv"

# =========================================================
# CONFIG: WeatherAPI
# =========================================================

WEATHER_API_KEY = os.getenv(
    "WEATHERAPI_KEY",
    "e685133506aa435ab9252438250612"  # your key as fallback
)
WEATHER_BASE_URL = "http://api.weatherapi.com/v1/current.json"

TEAM_ABBR_TO_CITY: Dict[str, str] = {
    "ARI": "Glendale,AZ",
    "ATL": "Atlanta,GA",
    "BAL": "Baltimore,MD",
    "BUF": "Buffalo,NY",
    "CAR": "Charlotte,NC",
    "CHI": "Chicago,IL",
    "CIN": "Cincinnati,OH",
    "CLE": "Cleveland,OH",
    "DAL": "Dallas,TX",
    "DEN": "Denver,CO",
    "DET": "Detroit,MI",
    "GB": "Green Bay,WI",
    "HOU": "Houston,TX",
    "IND": "Indianapolis,IN",
    "JAX": "Jacksonville,FL",
    "KC": "Kansas City,MO",
    "LAC": "Inglewood,CA",
    "LAR": "Inglewood,CA",
    "LV": "Las Vegas,NV",
    "MIA": "Miami,FL",
    "MIN": "Minneapolis,MN",
    "NE": "Foxborough,MA",
    "NO": "New Orleans,LA",
    "NYG": "East Rutherford,NJ",
    "NYJ": "East Rutherford,NJ",
    "PHI": "Philadelphia,PA",
    "PIT": "Pittsburgh,PA",
    "SEA": "Seattle,WA",
    "SF": "Santa Clara,CA",
    "TB": "Tampa,FL",
    "TEN": "Nashville,TN",
    "WAS": "Landover,MD",
    # Legacy
    "OAK": "Oakland,CA",
    "SD": "San Diego,CA",
    "STL": "St. Louis,MO",
}

# Teams with domes / mostly indoor
DOME_TEAMS = {
    "ATL", "DET", "MIN", "NO", "DAL", "HOU", "IND", "ARI", "LV", "LAC", "LAR"
}

# =========================================================
# Data cleaning
# =========================================================

def week_to_num(x):
    try:
        return int(x)
    except Exception:
        mapping = {
            "Wildcard": 18,
            "Division": 19,
            "Conference": 20,
            "Superbowl": 21,
            "Super Bowl": 21,
            "SB": 21,
        }
        return mapping.get(str(x), 0)


def clean_for_model(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
    df = df.copy()

    # Need scores to define label
    df = df.dropna(subset=["score_home", "score_away"])
    df["point_diff"] = df["score_home"] - df["score_away"]
    df["home_win"] = (df["point_diff"] > 0).astype(int)

    # Date features
    df["schedule_date_dt"] = pd.to_datetime(df["schedule_date"], errors="coerce")
    df["game_month"] = df["schedule_date_dt"].dt.month
    df["game_dow"] = df["schedule_date_dt"].dt.weekday

    # Week + playoff
    df["schedule_week_num"] = df["schedule_week"].apply(week_to_num)
    df["is_playoff"] = df["schedule_playoff"].astype(int)

    # Encode teams
    teams = sorted(set(df["team_home"].unique()) | set(df["team_away"].unique()))
    team_to_id = {name: i + 1 for i, name in enumerate(teams)}
    df["home_team_id"] = df["team_home"].map(team_to_id)
    df["away_team_id"] = df["team_away"].map(team_to_id)

    # Encode favorite
    fav_vals = sorted(
        v for v in df["team_favorite_id"].dropna().unique()
        if v not in ["PICK", "PK"]
    )
    fav_to_id = {name: i + 1 for i, name in enumerate(fav_vals)}
    df["favorite_team_id"] = df["team_favorite_id"].map(fav_to_id)
    df.loc[df["team_favorite_id"].isin(["PICK", "PK"]), "favorite_team_id"] = 0
    df["favorite_team_id"] = df["favorite_team_id"].fillna(0).astype(int)

    # Numeric betting + weather
    for col in [
        "spread_favorite",
        "over_under_line",
        "weather_temperature",
        "weather_wind_mph",
        "weather_humidity",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Implied points
    df["fav_implied_pts"] = np.where(
        df["spread_favorite"].notna() & df["over_under_line"].notna(),
        df["over_under_line"] / 2 - df["spread_favorite"] / 2,
        np.nan,
    )
    df["dog_implied_pts"] = np.where(
        df["spread_favorite"].notna() & df["over_under_line"].notna(),
        df["over_under_line"] / 2 + df["spread_favorite"] / 2,
        np.nan,
    )
    df["abs_spread"] = df["spread_favorite"].abs()

    # Stadium encoding
    stadiums = sorted(df["stadium"].dropna().unique())
    stadium_to_id = {name: i + 1 for i, name in enumerate(stadiums)}
    df["stadium_id"] = df["stadium"].map(stadium_to_id).fillna(0).astype(int)
    df["stadium_neutral"] = df["stadium_neutral"].astype(int)

    # Weather detail flags
    wd = df["weather_detail"].fillna("").str.lower()
    df["is_indoor"] = (wd.str.contains("indoor") | wd.str.contains("dome")).astype(int)
    df["is_rain"] = wd.str.contains("rain").astype(int)
    df["is_snow"] = wd.str.contains("snow").astype(int)

    cols = [
        "schedule_season",
        "schedule_week_num",
        "is_playoff",
        "game_month",
        "game_dow",
        "home_team_id",
        "away_team_id",
        "favorite_team_id",
        "spread_favorite",
        "abs_spread",
        "over_under_line",
        "fav_implied_pts",
        "dog_implied_pts",
        "stadium_id",
        "stadium_neutral",
        "weather_temperature",
        "weather_wind_mph",
        "weather_humidity",
        "is_indoor",
        "is_rain",
        "is_snow",
        "home_win",
    ]

    cleaned = df[cols].copy()

    # Fill NaNs with column medians
    for col in cleaned.columns:
        if cleaned[col].dtype.kind in "biufc":
            median_val = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(median_val)

    return cleaned, team_to_id, stadium_to_id

# =========================================================
# Weather helpers
# =========================================================

def fetch_weather_for_city(city_q: str) -> Dict[str, Any]:
    params = {
        "key": WEATHER_API_KEY,
        "q": city_q,
        "aqi": "no",
    }
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


def infer_rain_snow_from_condition(condition_text: str) -> Tuple[int, int]:
    text = (condition_text or "").lower()
    is_rain = int(
        "rain" in text
        or "drizzle" in text
        or "shower" in text
        or "thunderstorm" in text
    )
    is_snow = int(
        "snow" in text
        or "sleet" in text
        or "blizzard" in text
        or "flurries" in text
    )
    return is_rain, is_snow

# =========================================================
# Training (with scaling for logistic regression)
# =========================================================

def train_models(cleaned: pd.DataFrame):
    X = cleaned.drop(columns=["home_win"])
    y = cleaned["home_win"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    print("\n==================== TRAINING SUMMARY ====================")

    # ---- Scale features for logistic regression ----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---- Logistic Regression ----
    print("\n[1] Logistic Regression (scaled features)")
    logreg = LogisticRegression(
        max_iter=5000,   # more iterations so it converges
        n_jobs=-1,
        solver="lbfgs"
    )
    logreg.fit(X_train_scaled, y_train)
    y_pred_lr = logreg.predict(X_test_scaled)
    y_prob_lr = logreg.predict_proba(X_test_scaled)[:, 1]
    acc_lr = accuracy_score(y_test, y_pred_lr)
    auc_lr = roc_auc_score(y_test, y_prob_lr)
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    print(f"  • Accuracy : {acc_lr:.3f}")
    print(f"  • ROC AUC  : {auc_lr:.3f}")
    print("  • Confusion matrix:\n", cm_lr)

    # ---- Random Forest (unscaled) ----
    print("\n[2] Random Forest (raw features)")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    acc_rf = accuracy_score(y_test, y_pred_rf)
    auc_rf = roc_auc_score(y_test, y_prob_rf)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    print(f"  • Accuracy : {acc_rf:.3f}")
    print(f"  • ROC AUC  : {auc_rf:.3f}")
    print("  • Confusion matrix:\n", cm_rf)

    # ---- Comparison table ----
    print("\n---------------- MODEL COMPARISON ----------------")
    print(f"{'Model':<18}{'Accuracy':>10}{'ROC AUC':>10}")
    print("-" * 38)
    print(f"{'LogisticRegression':<18}{acc_lr:>10.3f}{auc_lr:>10.3f}")
    print(f"{'RandomForest':<18}{acc_rf:>10.3f}{auc_rf:>10.3f}")
    print("--------------------------------------------------")

    # Pick best by accuracy
    if acc_rf >= acc_lr:
        print(f"\n➡ Best model: RandomForest (accuracy {acc_rf:.3f})")
        best_model = rf
        best_scaler: Optional[StandardScaler] = None  # RF uses raw features
        best_name = "RandomForest"
    else:
        print(f"\n➡ Best model: LogisticRegression (accuracy {acc_lr:.3f})")
        best_model = logreg
        best_scaler = scaler
        best_name = "LogisticRegression"

    print("==========================================================\n")
    return best_model, X.columns.tolist(), best_scaler, best_name

# =========================================================
# CLI + feature construction
# =========================================================

def build_team_name_lookup(raw_df: pd.DataFrame) -> Dict[str, str]:
    """Map user inputs (abbr or lowercase full name) -> canonical full name."""
    name_to_full: Dict[str, str] = {}

    for name in pd.concat([raw_df["team_home"], raw_df["team_away"]]).dropna().unique():
        key = str(name).lower()
        name_to_full[key] = name

    for abbr, full_name in raw_df[["team_favorite_id", "team_home"]].dropna().drop_duplicates().values:
        key = str(abbr).lower()
        if key not in name_to_full:
            name_to_full[key] = full_name

    return name_to_full


def resolve_team_fullname(user_input: str, name_lookup: Dict[str, str]) -> str:
    key = user_input.strip().lower()
    if key not in name_lookup:
        raise ValueError(f"Unknown team: {user_input}")
    return name_lookup[key]


def input_with_default(prompt: str, default: str) -> str:
    s = input(f"{prompt} [{default}]: ").strip()
    return s if s else default


def prompt_for_game_details(
    team_name_lookup: Dict[str, str],
    team_to_id: Dict[str, int],
    default_season: int,
) -> Dict[str, Any]:
    print("=============== ENTER GAME DETAILS ===============")
    home_raw = input("Home team (abbr or full name): ").strip()
    away_raw = input("Away team (abbr or full name): ").strip()

    home_full = resolve_team_fullname(home_raw, team_name_lookup)
    away_full = resolve_team_fullname(away_raw, team_name_lookup)

    home_abbr_guess = home_raw.upper()
    away_abbr_guess = away_raw.upper()

    print("\n———— Game context ————")
    season_str = input_with_default("Season (year)", str(default_season))
    season = int(season_str)

    week_str = input_with_default("Week number", "8")
    week_num = int(week_str)

    playoff_str = input_with_default("Is this a playoff game? [y/n]", "n")
    is_playoff = 1 if playoff_str.lower().startswith("y") else 0

    month_str = input_with_default("Game month (1-12)", "11")
    game_month = int(month_str)

    dow_str = input_with_default("Day of week (0=Mon ... 6=Sun)", "6")
    game_dow = int(dow_str)

    print("\n———— Betting ————")
    spread_str = input_with_default(
        "Spread for FAVORITE (home negative, away positive)", "-4.5"
    )
    spread_fav = float(spread_str)

    ou_str = input_with_default("Over/under total points", "42.0")
    over_under = float(ou_str)

    fav_team_input = input_with_default(
        "Favorite team abbreviation (or leave blank for home)",
        home_abbr_guess if home_abbr_guess else "",
    ).strip().upper()
    if fav_team_input == "":
        favorite_abbr = home_abbr_guess
    else:
        favorite_abbr = fav_team_input

    print("\n———— Weather ————")
    weather_temperature = None
    weather_wind_mph = None
    weather_humidity = None
    is_rain_auto = 0
    is_snow_auto = 0

    city_q = TEAM_ABBR_TO_CITY.get(home_abbr_guess)
    if city_q and WEATHER_API_KEY:
        try:
            print(f"[INFO] Fetching live weather for {city_q} via WeatherAPI...")
            w = fetch_weather_for_city(city_q)
            weather_temperature = float(w["temp_f"]) if w["temp_f"] is not None else None
            weather_wind_mph = float(w["wind_mph"]) if w["wind_mph"] is not None else None
            weather_humidity = float(w["humidity"]) if w["humidity"] is not None else None
            is_rain_auto, is_snow_auto = infer_rain_snow_from_condition(w["condition_text"])
            print(
                f"  • Temp: {weather_temperature} F\n"
                f"  • Wind: {weather_wind_mph} mph\n"
                f"  • Hum : {weather_humidity}%\n"
                f"  • Cond: {w['condition_text']}"
            )
        except Exception as e:
            print(f"[WARN] Could not fetch weather automatically: {e}")
            print("[WARN] Falling back to manual weather input.")

    if weather_temperature is None:
        temp_str = input_with_default("Temperature (F)", "62.0")
        weather_temperature = float(temp_str)

    if weather_wind_mph is None:
        wind_str = input_with_default("Wind speed (mph)", "8.0")
        weather_wind_mph = float(wind_str)

    if weather_humidity is None:
        hum_str = input_with_default("Humidity (%)", "69.0")
        weather_humidity = float(hum_str)

    default_indoor = "y" if home_abbr_guess in DOME_TEAMS else "n"
    indoor_str = input_with_default(
        "Is this game indoors / in a dome? [y/n]", default_indoor
    )
    is_indoor = 1 if indoor_str.lower().startswith("y") else 0

    rain_str = input_with_default(
        "Is it raining? [y/n]", "y" if is_rain_auto else "n"
    )
    is_rain = 1 if rain_str.lower().startswith("y") else 0

    snow_str = input_with_default(
        "Is it snowing? [y/n]", "y" if is_snow_auto else "n"
    )
    is_snow = 1 if snow_str.lower().startswith("y") else 0

    print("===================================================\n")

    return {
        "home_full": home_full,
        "away_full": away_full,
        "home_abbr": home_abbr_guess,
        "away_abbr": away_abbr_guess,
        "favorite_abbr": favorite_abbr,
        "season": season,
        "week_num": week_num,
        "is_playoff": is_playoff,
        "game_month": game_month,
        "game_dow": game_dow,
        "spread_fav": spread_fav,
        "over_under": over_under,
        "weather_temperature": weather_temperature,
        "weather_wind_mph": weather_wind_mph,
        "weather_humidity": weather_humidity,
        "is_indoor": is_indoor,
        "is_rain": is_rain,
        "is_snow": is_snow,
    }


def build_feature_vector(
    user_inputs: Dict[str, Any],
    team_to_id: Dict[str, int],
    favorite_abbr_to_id: Dict[str, int],
    stadium_to_id: Dict[str, int],
    feature_order: list,
) -> Dict[str, float]:
    home_id = team_to_id[user_inputs["home_full"]]
    away_id = team_to_id[user_inputs["away_full"]]

    fav_id = favorite_abbr_to_id.get(user_inputs["favorite_abbr"], 0)

    spread_fav = user_inputs["spread_fav"]
    over_under = user_inputs["over_under"]
    fav_implied = over_under / 2 - spread_fav / 2
    dog_implied = over_under / 2 + spread_fav / 2
    abs_spread = abs(spread_fav)

    # For now, we don't try to pick a specific stadium; use 0 and non-neutral.
    stadium_id = 0
    stadium_neutral = 0

    feat = {
        "schedule_season": user_inputs["season"],
        "schedule_week_num": user_inputs["week_num"],
        "is_playoff": user_inputs["is_playoff"],
        "game_month": user_inputs["game_month"],
        "game_dow": user_inputs["game_dow"],
        "home_team_id": home_id,
        "away_team_id": away_id,
        "favorite_team_id": fav_id,
        "spread_favorite": spread_fav,
        "abs_spread": abs_spread,
        "over_under_line": over_under,
        "fav_implied_pts": fav_implied,
        "dog_implied_pts": dog_implied,
        "stadium_id": stadium_id,
        "stadium_neutral": stadium_neutral,
        "weather_temperature": user_inputs["weather_temperature"],
        "weather_wind_mph": user_inputs["weather_wind_mph"],
        "weather_humidity": user_inputs["weather_humidity"],
        "is_indoor": user_inputs["is_indoor"],
        "is_rain": user_inputs["is_rain"],
        "is_snow": user_inputs["is_snow"],
    }

    # Ensure all required features exist
    for col in feature_order:
        if col not in feat:
            feat[col] = 0.0

    return {col: float(feat[col]) for col in feature_order}


def predict_game(
    model,
    user_inputs: Dict[str, Any],
    team_to_id: Dict[str, int],
    favorite_abbr_to_id: Dict[str, int],
    stadium_to_id: Dict[str, int],
    feature_order: list,
    scaler_for_best: Optional[StandardScaler],
) -> Tuple[bool, float]:
    feat_dict = build_feature_vector(
        user_inputs,
        team_to_id=team_to_id,
        favorite_abbr_to_id=favorite_abbr_to_id,
        stadium_to_id=stadium_to_id,
        feature_order=feature_order,
    )

    x_df = pd.DataFrame([feat_dict], columns=feature_order)

    # If best model is logistic regression, we need to scale here
    if scaler_for_best is not None:
        x_scaled = scaler_for_best.transform(x_df.values)
        proba = model.predict_proba(x_scaled)[0, 1]
    else:
        # RandomForest path: use raw features (DataFrame is fine)
        proba = model.predict_proba(x_df)[0, 1]

    predicted_home_win = proba >= 0.5
    return predicted_home_win, float(proba)

# =========================================================
# MAIN
# =========================================================

def main():
    print(f"Loading raw CSV from: {RAW_CSV}")
    raw_df = pd.read_csv(RAW_CSV)

    cleaned, team_to_id, stadium_to_id = clean_for_model(raw_df)
    print("Cleaned shape:", cleaned.shape)

    best_model, feature_order, scaler_for_best, best_name = train_models(cleaned)

    # Favorite-team ID map (abbr -> id)
    fav_abbrs = sorted(
        v for v in raw_df["team_favorite_id"].dropna().unique()
        if v not in ["PICK", "PK"]
    )
    favorite_abbr_to_id = {abbr: i + 1 for i, abbr in enumerate(fav_abbrs)}

    # For mapping user input -> canonical team name
    team_name_lookup = build_team_name_lookup(raw_df)
    default_season = int(cleaned["schedule_season"].max())

    user_inputs = prompt_for_game_details(
        team_name_lookup=team_name_lookup,
        team_to_id=team_to_id,
        default_season=default_season,
    )

    predicted_home_win, home_win_proba = predict_game(
        best_model,
        user_inputs=user_inputs,
        team_to_id=team_to_id,
        favorite_abbr_to_id=favorite_abbr_to_id,
        stadium_to_id=stadium_to_id,
        feature_order=feature_order,
        scaler_for_best=scaler_for_best,
    )

    home_label = f"{user_inputs['home_abbr']} ({user_inputs['home_full']})"
    away_label = f"{user_inputs['away_abbr']} ({user_inputs['away_full']})"
    winner_label = home_label if predicted_home_win else away_label

    print("\n==================== PREDICTION RESULT ====================")
    print(f"Model used : {best_name}")
    print(f"Matchup    : {home_label}  vs  {away_label}")
    print(
        f"Season     : {user_inputs['season']}   "
        f"Week: {user_inputs['week_num']}   "
        f"Playoff: {'Yes' if user_inputs['is_playoff'] else 'No'}"
    )
    print(f"Spread     : {user_inputs['spread_fav']}  (favorite: {user_inputs['favorite_abbr']})")
    print(f"O/U Total  : {user_inputs['over_under']}")
    print(
        f"Weather    : {user_inputs['weather_temperature']} F, "
        f"{user_inputs['weather_wind_mph']} mph wind, "
        f"{user_inputs['weather_humidity']}% humidity"
    )
    print(
        "Conditions : "
        f"{'Indoor' if user_inputs['is_indoor'] else 'Outdoor'}, "
        f"{'Rain' if user_inputs['is_rain'] else 'No rain'}, "
        f"{'Snow' if user_inputs['is_snow'] else 'No snow'}"
    )
    print("-----------------------------------------------------------")
    print(f"🏈 Predicted winner : {winner_label}")
    print(f"🏠 Home win prob    : {home_win_proba:.3%}")
    print("===========================================================\n")


if __name__ == "__main__":
    main()
