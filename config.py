# config.py
import os

# Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Or os.environ.get("GEMINI_API_KEY")

# Bybit API Keys (assuming your backend uses them for market data/trading)
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY") # Or os.environ.get("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET") # Or os.environ.get("BYBIT_API_SECRET")

# Custom App ID for Firestore collection paths to separate data for different deployments
APP_ID = os.getenv("APP_ID") # Or os.environ.get("APP_ID")

# If you decide to use CoinGecko with a pro plan requiring a key in the future,
# you would add it here. Basic CoinGecko public API typically doesn't need a key.
# COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY") # Example for future use if needed