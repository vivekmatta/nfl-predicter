import os
import json
import requests

# =========================================================
# CONFIG
# =========================================================

# Prefer environment variable so you don't accidentally commit the key
# In PowerShell, you can set it like:
#   $env:WEATHERAPI_KEY = "e685133506aa435ab9252438250612"
#
# If the env var isn't set, we'll fall back to the literal key below.
API_KEY = os.getenv("WEATHERAPI_KEY", "e685133506aa435ab9252438250612")

BASE_URL = "http://api.weatherapi.com/v1/current.json"


def fetch_current_weather(query: str) -> dict:
    """
    Call WeatherAPI current weather endpoint and return the JSON as a dict.

    query examples:
      - "Detroit,MI"
      - "40.71,-74.00"
      - "90210"
      - "London"
    """
    params = {
        "key": API_KEY,
        "q": query,
        "aqi": "no",  # we don't need air quality for now
    }

    print("\n[INFO] Sending request to WeatherAPI...")
    resp = requests.get(BASE_URL, params=params, timeout=10)

    print(f"[INFO] HTTP status: {resp.status_code}")
    print(f"[INFO] Final URL  : {resp.url}")

    # Raise if HTTP error
    resp.raise_for_status()

    data = resp.json()

    # WeatherAPI puts errors inside JSON with "error" key sometimes
    if isinstance(data, dict) and "error" in data:
        err = data["error"]
        code = err.get("code")
        msg = err.get("message")
        raise RuntimeError(f"WeatherAPI error {code}: {msg}")

    return data


def pretty_print_json(data: dict) -> None:
    """Print a Python dict as nicely formatted JSON."""
    print("\n===== RAW JSON RESPONSE =====")
    print(json.dumps(data, indent=2, sort_keys=True))
    print("===== END JSON =====\n")


def main():
    print("======================================")
    print("   WeatherAPI JSON Test (Current)     ")
    print("======================================")
    print("Enter a location for WeatherAPI 'q' parameter.")
    print("Examples:  Detroit,MI   90210   40.71,-74.00   London")
    print("Press Enter with nothing to use default: Detroit,MI\n")

    user_q = input("Location (q): ").strip()
    if not user_q:
        user_q = "Detroit,MI"

    try:
        data = fetch_current_weather(user_q)
        pretty_print_json(data)
    except Exception as e:
        print(f"\n[ERROR] {e}\n")


if __name__ == "__main__":
    main()
