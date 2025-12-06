import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
BASE_DIR = Path(r"C:\Users\Vhas21\Desktop\Northwestern\Projects\nfl-predicter")
RAW_CSV = BASE_DIR / "spreadspoke_scores.csv"


# ---------------------------------------------------------
# 1. Data cleaning / feature engineering
# ---------------------------------------------------------

def week_to_num(x):
    """Map schedule_week to a numeric value."""
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


def clean_for_model(df: pd.DataFrame):
    """
    Clean raw spreadspoke_scores.csv and return:
      - cleaned_df: numeric features + home_win label
      - team_to_id: mapping FULL team name -> id (for training)
      - fav_to_id: mapping favorite team abbrev -> id (0 = pick'em)
      - abbr_to_team_name: mapping abbrev like 'BUF' -> full team name
      - team_default_stadium_id: mapping full team name -> most common stadium_id
      - feature_cols: list of feature column names
      - feature_medians: median value per feature (for default/empty inputs)
      - default_season: most recent season in dataset
    """
    df = df.copy()

    # Keep only rows with scores (needed to define label)
    df = df.dropna(subset=["score_home", "score_away"])

    # Label: 1 if home team wins, 0 otherwise
    df["point_diff"] = df["score_home"] - df["score_away"]
    df["home_win"] = (df["point_diff"] > 0).astype(int)

    # Date features
    df["schedule_date_dt"] = pd.to_datetime(df["schedule_date"], errors="coerce")
    df["game_month"] = df["schedule_date_dt"].dt.month
    df["game_dow"] = df["schedule_date_dt"].dt.weekday  # 0=Mon ... 6=Sun

    # Week + playoff
    df["schedule_week_num"] = df["schedule_week"].apply(week_to_num)
    df["is_playoff"] = df["schedule_playoff"].astype(int)

    # ---- Team IDs (full names) ----
    teams = sorted(set(df["team_home"].unique()) | set(df["team_away"].unique()))
    team_to_id = {name: i + 1 for i, name in enumerate(teams)}

    df["home_team_id"] = df["team_home"].map(team_to_id)
    df["away_team_id"] = df["team_away"].map(team_to_id)

    # ---- Build abbrev -> full team name map (BUF -> Buffalo Bills) ----
    abbr_to_team_name: dict[str, str] = {}
    for _, row in df.loc[~df["team_favorite_id"].isin(["PICK", "PK"])].iterrows():
        abbr = row["team_favorite_id"]
        if pd.isna(abbr):
            continue
        if abbr not in abbr_to_team_name:
            # Use the home team name for that abbreviation (good enough for mapping)
            abbr_to_team_name[abbr] = row["team_home"]

    # Favorite team IDs (based on abbrevs like BUF, NE, KC, etc.)
    fav_series = df["team_favorite_id"].dropna()
    fav_vals = sorted(
        v for v in fav_series.unique()
        if v not in ["PICK", "PK"]
    )
    fav_to_id = {name: i + 1 for i, name in enumerate(fav_vals)}

    df["favorite_team_id"] = df["team_favorite_id"].map(fav_to_id)
    df.loc[df["team_favorite_id"].isin(["PICK", "PK"]), "favorite_team_id"] = 0
    df["favorite_team_id"] = df["favorite_team_id"].fillna(0).astype(int)

    # Numeric betting + weather columns
    for col in [
        "spread_favorite",
        "over_under_line",
        "weather_temperature",
        "weather_wind_mph",
        "weather_humidity",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Implied points from Vegas line
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

    # Default stadium per home team (most common stadium)
    team_default_stadium_id: dict[str, int] = {}
    for team_name, sub in df.groupby("team_home"):
        counts = Counter(sub["stadium_id"])
        if counts:
            most_common_id, _ = counts.most_common(1)[0]
            team_default_stadium_id[team_name] = int(most_common_id)

    # Weather flags from text
    wd = df["weather_detail"].fillna("").str.lower()
    df["is_indoor"] = (wd.str.contains("indoor") | wd.str.contains("dome")).astype(int)
    df["is_rain"] = wd.str.contains("rain").astype(int)
    df["is_snow"] = wd.str.contains("snow").astype(int)

    feature_cols = [
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
    ]

    cleaned = df[feature_cols + ["home_win"]].copy()

    # Fill NaNs with column medians
    feature_medians = cleaned[feature_cols].median().to_dict()
    for col in feature_cols:
        cleaned[col] = cleaned[col].fillna(feature_medians[col])

    default_season = int(cleaned["schedule_season"].max())

    return (
        cleaned,
        team_to_id,
        fav_to_id,
        abbr_to_team_name,
        team_default_stadium_id,
        feature_cols,
        feature_medians,
        default_season,
    )


# ---------------------------------------------------------
# 2. Training models
# ---------------------------------------------------------

def train_models(cleaned_df: pd.DataFrame, feature_cols):
    """Train Logistic Regression and Random Forest, return best one."""
    X = cleaned_df[feature_cols].values
    y = cleaned_df["home_win"].values

    # Random split; you can change to season-based later if you want
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Logistic Regression pipeline
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, n_jobs=-1)),
    ])

    # Random Forest
    rf = Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ))
    ])

    models = {
        "logreg": logreg,
        "rf": rf,
    }

    best_model = None
    best_acc = -1.0

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        print(f"{name} accuracy: {acc:.3f}")

        if y_proba is not None:
            auc = roc_auc_score(y_test, y_proba)
            print(f"{name} ROC AUC: {auc:.3f}")

        cm = confusion_matrix(y_test, y_pred)
        print(f"{name} confusion matrix:\n{cm}\n")

        if acc > best_acc:
            best_acc = acc
            best_model = model

    print(f"Best model: {best_model.steps[-1][0]} with accuracy {best_acc:.3f}")
    return best_model


# ---------------------------------------------------------
# 3. Prediction helpers
# ---------------------------------------------------------

def get_team_id(code: str, team_to_id: dict, abbr_to_team_name: dict) -> int:
    """
    Accept either a full team name (as in dataset) or an abbreviation like 'BUF'.
    Returns the team_id used for training.
    """
    if code in team_to_id:
        return team_to_id[code]

    if code in abbr_to_team_name:
        full_name = abbr_to_team_name[code]
        return team_to_id[full_name]

    raise KeyError(f"Unknown team code or name: {code}")


def build_feature_vector(
    season,
    week_num,
    is_playoff,
    game_month,
    game_dow,
    home_team_code,
    away_team_code,
    favorite_team_abbr,
    spread_favorite,
    over_under_line,
    stadium_id,
    stadium_neutral,
    temp_f,
    wind_mph,
    humidity,
    is_indoor,
    is_rain,
    is_snow,
    team_to_id,
    fav_to_id,
    abbr_to_team_name,
    feature_cols,
):
    """
    Build a single feature row (1 x n_features) in the same order as training.
    """
    home_team_id = get_team_id(home_team_code, team_to_id, abbr_to_team_name)
    away_team_id = get_team_id(away_team_code, team_to_id, abbr_to_team_name)

    # Favorite team ID
    if not favorite_team_abbr:
        favorite_team_id = 0
    elif favorite_team_abbr in ["PICK", "PK"]:
        favorite_team_id = 0
    else:
        favorite_team_id = fav_to_id.get(favorite_team_abbr, 0)

    abs_spread = abs(spread_favorite)
    fav_implied_pts = over_under_line / 2 - spread_favorite / 2
    dog_implied_pts = over_under_line / 2 + spread_favorite / 2

    row = {
        "schedule_season": season,
        "schedule_week_num": week_num,
        "is_playoff": int(is_playoff),
        "game_month": game_month,
        "game_dow": game_dow,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "favorite_team_id": favorite_team_id,
        "spread_favorite": spread_favorite,
        "abs_spread": abs_spread,
        "over_under_line": over_under_line,
        "fav_implied_pts": fav_implied_pts,
        "dog_implied_pts": dog_implied_pts,
        "stadium_id": stadium_id,
        "stadium_neutral": int(stadium_neutral),
        "weather_temperature": temp_f,
        "weather_wind_mph": wind_mph,
        "weather_humidity": humidity,
        "is_indoor": int(is_indoor),
        "is_rain": int(is_rain),
        "is_snow": int(is_snow),
    }

    # Ensure order matches feature_cols
    features = np.array([[row[col] for col in feature_cols]], dtype=float)
    return features


def predict_game(
    model,
    team_to_id,
    fav_to_id,
    abbr_to_team_name,
    team_default_stadium_id,
    feature_cols,
    feature_medians,
    season,
    week_num,
    is_playoff,
    game_month,
    game_dow,
    home_team_code,
    away_team_code,
    favorite_team_abbr,
    spread_favorite,
    over_under_line,
    temp_f,
    wind_mph,
    humidity,
    is_indoor,
    is_rain,
    is_snow,
):
    """
    High-level helper: returns predicted winner + probability of home win.
    Auto-fills stadium_id from home team if possible.
    """
    # Default stadium = home team's usual stadium
    try:
        full_home_name = abbr_to_team_name.get(home_team_code, home_team_code)
        stadium_id = team_default_stadium_id.get(full_home_name, 0)
        stadium_neutral = 0
    except Exception:
        stadium_id = 0
        stadium_neutral = 1

    x = build_feature_vector(
        season,
        week_num,
        is_playoff,
        game_month,
        game_dow,
        home_team_code,
        away_team_code,
        favorite_team_abbr,
        spread_favorite,
        over_under_line,
        stadium_id,
        stadium_neutral,
        temp_f,
        wind_mph,
        humidity,
        is_indoor,
        is_rain,
        is_snow,
        team_to_id,
        fav_to_id,
        abbr_to_team_name,
        feature_cols,
    )

    proba = model.predict_proba(x)[0, 1]
    home_win = proba >= 0.5
    predicted_winner = home_team_code if home_win else away_team_code
    return predicted_winner, float(proba)


# ---------------------------------------------------------
# 4. Simple, organized CLI UI
# ---------------------------------------------------------

def ask_str(prompt, default=None):
    """Ask for a string, allow empty = use default."""
    if default is not None:
        txt = input(f"{prompt} [{default}]: ").strip()
        return txt if txt != "" else default
    else:
        return input(f"{prompt}: ").strip()


def ask_int(prompt, default=None):
    """Ask for int, allow empty = use default."""
    while True:
        if default is not None:
            txt = input(f"{prompt} [{default}]: ").strip()
            if txt == "":
                return int(default)
        else:
            txt = input(f"{prompt}: ").strip()
            if txt == "":
                print("Please enter a number or Ctrl+C to exit.")
                continue
        try:
            return int(txt)
        except ValueError:
            print("Not a valid integer, try again.")


def ask_float(prompt, default=None):
    """Ask for float, allow empty = use default."""
    while True:
        if default is not None:
            txt = input(f"{prompt} [{default}]: ").strip()
            if txt == "":
                return float(default)
        else:
            txt = input(f"{prompt}: ").strip()
            if txt == "":
                print("Please enter a number or Ctrl+C to exit.")
                continue
        try:
            return float(txt)
        except ValueError:
            print("Not a valid number, try again.")


def ask_yes_no(prompt, default=0):
    """Ask for yes/no, return 1 or 0. Empty = default."""
    default_str = "y" if default == 1 else "n"
    txt = input(f"{prompt} [y/n, default {default_str}]: ").strip().lower()
    if txt == "":
        return int(default)
    if txt in ["y", "yes"]:
        return 1
    if txt in ["n", "no"]:
        return 0
    print("Input not understood, using default.")
    return int(default)


def interactive_ui(
    model,
    team_to_id,
    fav_to_id,
    abbr_to_team_name,
    team_default_stadium_id,
    feature_cols,
    feature_medians,
    default_season,
):
    """
    Simple text UI: user enters teams and (optionally) other parameters.
    Blank inputs use reasonable defaults.
    """
    print("\n" + "=" * 60)
    print("        NFL GAME PREDICTOR (TABULAR ML, PYTHON EDITION)")
    print("=" * 60)
    print("Tips:")
    print("  - Use team abbreviations like: BUF, KC, NE, DAL, SF")
    print("  - Press Enter to skip optional fields (use defaults)")
    print("  - Ctrl+C to quit at any time.\n")

    # Show which abbreviations we know
    known_abbrs = sorted(abbr_to_team_name.keys())
    if known_abbrs:
        print("Known team abbreviations (from dataset):")
        print(", ".join(known_abbrs))
        print()

    while True:
        print("-" * 60)
        print("Enter game details:")

        # Home / away are REQUIRED (but still allow abbreviation or full name)
        home = ask_str("Home team (abbr or full name)", default=None).upper()
        away = ask_str("Away team (abbr or full name)", default=None).upper()

        # Season, week, etc.
        season = ask_int("Season (year)", default=default_season)
        week_num = ask_int("Week number", default=8)
        is_playoff = ask_yes_no("Is this a playoff game?", default=0)

        # Month & day-of-week
        # If user skips, use typical defaults (Nov, Sunday)
        game_month = ask_int("Game month (1-12)", default=11)
        game_dow = ask_int("Day of week (0=Mon ... 6=Sun)", default=6)

        # Spread / total
        median_spread = round(feature_medians["spread_favorite"], 1)
        median_ou = round(feature_medians["over_under_line"], 1)
        spread = ask_float(
            "Spread for FAVORITE (home negative, away positive)",
            default=median_spread,
        )
        ou_total = ask_float("Over/under total points", default=median_ou)

        # Favorite team (optional). Default: home team.
        default_fav = home
        fav_team = ask_str(
            "Favorite team abbreviation (or leave blank for home)",
            default=default_fav,
        ).upper()

        # Weather
        median_temp = round(feature_medians["weather_temperature"], 1)
        median_wind = round(feature_medians["weather_wind_mph"], 1)
        median_humid = round(feature_medians["weather_humidity"], 1)

        temp_f = ask_float("Temperature (F)", default=median_temp)
        wind_mph = ask_float("Wind speed (mph)", default=median_wind)
        humidity = ask_float("Humidity (%)", default=median_humid)

        is_indoor = ask_yes_no("Is this game indoors / in a dome?", default=0)
        is_rain = ask_yes_no("Is it raining?", default=0)
        is_snow = ask_yes_no("Is it snowing?", default=0)

        try:
            winner, prob = predict_game(
                model=model,
                team_to_id=team_to_id,
                fav_to_id=fav_to_id,
                abbr_to_team_name=abbr_to_team_name,
                team_default_stadium_id=team_default_stadium_id,
                feature_cols=feature_cols,
                feature_medians=feature_medians,
                season=season,
                week_num=week_num,
                is_playoff=is_playoff,
                game_month=game_month,
                game_dow=game_dow,
                home_team_code=home,
                away_team_code=away,
                favorite_team_abbr=fav_team,
                spread_favorite=spread,
                over_under_line=ou_total,
                temp_f=temp_f,
                wind_mph=wind_mph,
                humidity=humidity,
                is_indoor=is_indoor,
                is_rain=is_rain,
                is_snow=is_snow,
            )

            print("\n" + "-" * 60)
            print(f"Prediction for {away} @ {home}")
            print("-" * 60)
            print(f"  Predicted winner       : {winner}")
            print(f"  Home win probability   : {prob:.3f}")
            print(f"  Away win probability   : {1.0 - prob:.3f}")
            print("-" * 60 + "\n")

        except KeyError as e:
            print(f"\n[ERROR] {e}")
            print("Make sure you used a known abbreviation or full team name.\n")

        # Ask if user wants another prediction
        again = ask_yes_no("Run another prediction?", default=1)
        if not again:
            print("\nExiting NFL predictor. Goodbye!\n")
            break


# ---------------------------------------------------------
# 5. Main
# ---------------------------------------------------------

def main():
    print(f"Loading raw CSV from: {RAW_CSV}")
    df = pd.read_csv(RAW_CSV)
    (
        cleaned,
        team_to_id,
        fav_to_id,
        abbr_to_team_name,
        team_default_stadium_id,
        feature_cols,
        feature_medians,
        default_season,
    ) = clean_for_model(df)
    print("Cleaned shape:", cleaned.shape)

    best_model = train_models(cleaned, feature_cols)

    # Start interactive UI
    interactive_ui(
        best_model,
        team_to_id,
        fav_to_id,
        abbr_to_team_name,
        team_default_stadium_id,
        feature_cols,
        feature_medians,
        default_season,
    )


if __name__ == "__main__":
    main()
