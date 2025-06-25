# config.py
import os

# Gemini API Key
# This will load the value from the environment variable named "GEMINI_API_KEY"
# that you set in Render's dashboard.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Bybit API Keys (assuming your backend uses them for market data/trading)
# These will load values from environment variables named "BYBIT_API_KEY" and "BYBIT_API_SECRET"
# that you set in Render's dashboard.
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

# Custom App ID for Firestore collection paths to separate data for different deployments
# This will load the value from the environment variable named "APP_ID"
# that you set in Render's dashboard (e.g., "aura-trading-dashboard-v2").
APP_ID = os.getenv("APP_ID")

# If you decide to use CoinGecko with a pro plan requiring a key in the future,
# you would add it here. Basic CoinGecko public API typically doesn't need a key.
# COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY") # Example for future use if needed