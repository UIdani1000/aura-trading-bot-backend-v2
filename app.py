import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
import pandas as pd
# Removed: import pandas_ta as ta # No longer using pandas_ta
from datetime import datetime, timedelta
import random
import json
import numpy as np
import time
import traceback

# Load environment variables from .env file (primarily for local development)
load_dotenv()

app = Flask(__name__)
CORS(app)

# --- Configuration (using os.getenv for consistency with Render setup) ---
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET')
# Corrected: Use GEMINI_API_KEY as per Render environment variable name
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') 

# Initialize Bybit client
bybit_client = None
if BYBIT_API_KEY and BYBIT_API_SECRET:
    try:
        bybit_client = HTTP(
            api_key=BYBIT_API_KEY,
            api_secret=BYBIT_API_SECRET,
            testnet=False, # Set to True for testnet
            timeout=30 # Increased timeout to 30 seconds
        )
        print("--- Bybit client initialized ---")
    except Exception as e:
        print(f"--- ERROR: Failed to initialize Bybit client: {e} ---")
else:
    print("--- Bybit API keys not found. Bybit client not initialized. ---")

# Initialize Google Gemini
gemini_model = None
try:
    if GEMINI_API_KEY:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY) # Corrected API key name
        gemini_model = genai.GenerativeModel('gemini-2.0-flash') # Using gemini-2.0-flash
        print("--- Successfully initialized model to gemini-2.0-flash at startup ---")
    else:
        print("--- WARNING: GEMINI_API_KEY environment variable not set. Gemini model not initialized. ---")
except Exception as e:
    gemini_model = None
    print(f"--- Failed to initialize Gemini model: {e} ---")
    traceback.print_exc()

# --- Health Check / Root Route (ADDED FOR RENDER) ---
@app.route('/')
def home():
    """A basic health check endpoint for Render deployment."""
    return "Aura Trading Bot Backend is running!", 200

# --- Helper Functions ---

def get_indicator_value(df, indicator_name, default_val="N/A"):
    """
    Helper to safely get the most recent indicator value from the DataFrame.
    Assumes DataFrame is ordered oldest to newest (iloc[-1] is most recent).
    """
    # Directly check for indicator_name existence first
    if indicator_name in df.columns:
        last_val = df[indicator_name].iloc[-1]
        if pd.isna(last_val):
            print(f"DEBUG: get_indicator_value for {indicator_name}: Last value is NaN, returning '{default_val}'")
            return default_val
        else:
            print(f"DEBUG: get_indicator_value for {indicator_name}: Raw value found: {last_val}")
            return round(float(last_val), 2)
    
    # If not found, log clearly what was tried
    print(f"DEBUG: get_indicator_value for {indicator_name}: Column '{indicator_name}' NOT found in DataFrame columns: {df.columns.tolist()}, returning '{default_val}'")
    return default_val

# Function to fetch live market data (for dashboard)
@app.route('/all_market_prices', methods=['GET'])
def get_all_market_prices():
    # Caching mechanism (if you want to keep it, otherwise remove these global vars)
    global last_market_data, last_fetch_time # Ensure these are defined globally if used
    # last_market_data = None # Uncomment and initialize if not done elsewhere
    # last_fetch_time = 0 # Uncomment and initialize if not done elsewhere
    CACHE_DURATION = 30 # Seconds

    if 'last_market_data' in globals() and last_market_data and (time.time() - last_fetch_time < CACHE_DURATION):
        app.logger.info("Serving dashboard market data from cache.")
        return jsonify(last_market_data)

    app.logger.info("Fetching new dashboard market data from Bybit.")
    
    if not bybit_client:
        app.logger.error("Error: Bybit client not initialized in get_all_market_prices. Check API keys.")
        return jsonify({"error": "Bybit client not initialized"}), 500

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"]
    market_data = {}

    for symbol in symbols:
        app.logger.info(f"\n--- Fetching daily kline data for {symbol} ---")
        try:
            kline_response = bybit_client.get_kline(
                category="spot", # Assuming spot for general market prices. Adjust if using linear/inverse.
                symbol=symbol,
                interval="D", # Daily interval for general market overview
                limit=1000 # Max limit supported by Bybit
            )
            kline_data = kline_response.get('result', {}).get('list', [])

            if kline_data and len(kline_data) >= 2: # Need at least 2 candles for percent change
                df = pd.DataFrame(kline_data, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                
                # --- CRITICAL FIX: Ensure all OHLCV and volume columns are numeric ---
                numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover']
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce') # Convert non-numeric to NaN
                
                df['start'] = pd.to_numeric(df['start']) # Explicitly cast to numeric to avoid FutureWarning
                df['start'] = pd.to_datetime(df['start'], unit='ms') 
                df.set_index('start', inplace=True)
                df = df.iloc[::-1] # Reverse DataFrame to be OLDEST TO NEWEST

                # --- DEBUG: Verify dtypes after conversion ---
                app.logger.info(f"\n--- DEBUG: Dashboard {symbol} dtypes after numeric conversion ---")
                df.info(verbose=True, buf=sys.stdout) # Use verbose=True to see all dtypes
                app.logger.info("------------------------------------------------------------------")
                
                # Check for NaNs immediately after conversion in critical columns
                if df[numeric_cols].isnull().any().any():
                    app.logger.warning(f"WARNING: Dashboard {symbol} has NaN values in OHLCV or volume columns after conversion.")

                last_close = df['close'].iloc[-1]
                prev_close = df['close'].iloc[-2] if len(df) > 1 else last_close
                percent_change = ((last_close - prev_close) / prev_close) * 100 if prev_close != 0 else 0

                # --- REMOVED PANDAS_TA CALLS from get_all_market_prices ---
                # These lines previously used df.ta.rsi, df.ta.macd, etc.
                # Since pandas_ta is not installed, these calls are removed.
                # Indicator values will now be "N/A" by default.
                rsi_val = "N/A"
                macd_val = "N/A"
                stoch_k_val = "N/A"
                orscr_signal = "N/A" # Default signal

                market_data[symbol] = {
                    "price": round(float(last_close), 2),
                    "percent_change": round(float(percent_change), 2),
                    "rsi": rsi_val,
                    "macd": macd_val,
                    "stoch_k": stoch_k_val, # Now correctly defined, defaulting to N/A
                    "volume": round(float(df['volume'].iloc[-1]), 2),
                    "orscr_signal": orscr_signal # Will be N/A for now
                }
                app.logger.info(f"Final market_data[{symbol}]: {market_data[symbol]}")
            else:
                app.logger.warning(f"No kline data or error for {symbol} from Bybit: {kline_response.get('retMsg', 'Unknown error')}")
                market_data[symbol] = {
                    "price": "N/A", "percent_change": "N/A", "rsi": "N/A",
                    "macd": "N/A", "stoch_k": "N/A", "volume": "N/A", "orscr_signal": "N/A"
                }

        except Exception as e:
            app.logger.error(f"Error fetching or processing data for {symbol} from Bybit: {e}")
            traceback.print_exc()
            market_data[symbol] = {
                "price": "N/A", "percent_change": "N/A", "rsi": "N/A",
                "macd": "N/A", "stoch_k": "N/A", "volume": "N/A", "orscr_signal": "N/A"
            }
        
        time.sleep(0.5) # Small delay to avoid hitting Bybit API rate limits

    # Update cache
    globals()['last_market_data'] = market_data
    globals()['last_fetch_time'] = time.time()

    return jsonify(market_data)

# --- Chat Endpoint (Existing) ---
@app.route('/chat', methods=['POST'])
def chat():
    # Access global gemini_model
    global gemini_model 
    if not gemini_model:
        app.logger.error("Gemini model not initialized for chat endpoint.")
        return jsonify({"error": "Gemini model not initialized"}), 500

    data = request.json
    user_message_data = data.get('message')
    user_name = data.get('userName', 'Trader')
    ai_name = data.get('aiName', 'Aura')

    if not user_message_data:
        return jsonify({"error": "No message provided"}), 400

    chat_history_from_frontend = data.get('chatHistory', [])
    latest_user_message = user_message_data

    gemini_chat_history = []
    for chat_turn in chat_history_from_frontend:
        gemini_chat_history.append({'role': chat_turn['role'], 'parts': [{'text': chat_turn['text']}]})

    if not chat_history_from_frontend and latest_user_message:
        gemini_chat_history.append({'role': 'user', 'parts': [{'text': latest_user_message}]})
    elif chat_history_from_frontend and latest_user_message:
        if gemini_chat_history and gemini_chat_history[-1]['role'] == 'user' and gemini_chat_history[-1]['parts'][0]['text'] == latest_user_message:
            pass
        else:
            gemini_chat_history.append({'role': 'user', 'parts': [{'text': latest_user_message}]})

    # Fetch live market data for context
    market_data_response = get_all_market_prices()
    # Check if get_all_market_prices returned an error (it can return 500 now)
    if isinstance(market_data_response, tuple) and market_data_response[1] != 200:
        app.logger.error(f"Failed to get market data for chat context: {market_data_response[0].json.get('error', 'Unknown error')}")
        market_data = {} # Fallback to empty data
    else:
        market_data = market_data_response.json


    # Trade history will now be managed on frontend, so we don't fetch it from local file
    mock_trade_history = [] 

    context_prefix = f"""
    You are Aura, an AI trading assistant. Your goal is to provide insightful, helpful, and **friendly** responses to the user's trading-related questions.
    Adopt a **conversational, approachable, and encouraging tone**. Avoid overly formal or robotic language.
    You can use emojis if appropriate to convey friendliness (e.g., 🚀🌟).

    Here is the current live market data:
    {json.dumps(market_data, indent=2)}

    (Note: Trade history is now managed by the user on the frontend, so it's not provided in real-time here for chat context.)
    """

    if gemini_chat_history:
        if gemini_chat_history[0]['role'] == 'user':
            gemini_chat_history[0]['parts'][0]['text'] = context_prefix + "\n" + gemini_chat_history[0]['parts'][0]['text']
        else: 
            found_user_turn = False
            for turn in gemini_chat_history:
                if turn['role'] == 'user':
                    turn['parts'][0]['text'] = context_prefix + "\n" + turn['parts'][0]['text']
                    found_user_turn = True
                    break
            if not found_user_turn:
                gemini_chat_history = [{'role': 'user', 'parts': [{'text': context_prefix + "\n" + latest_user_message}]}] + gemini_chat_history
    else:
        gemini_chat_history.append({'role': 'user', 'parts': [{'text': context_prefix + "\n" + latest_user_message}]})

    try:
        app.logger.info("\n--- DEBUG: Chat history sent to Gemini API ---")
        for entry in gemini_chat_history:
            app.logger.info(f"Role: {entry['role']}, Text: {entry['parts'][0]['text'][:200]}...")
        app.logger.info("----------------------------------------------")

        response = gemini_model.generate_content(gemini_chat_history)
        ai_response = response.candidates[0].content.parts[0].text
        return jsonify({"response": ai_response})
    except Exception as e:
        app.logger.error(f"Error generating content with Gemini: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Failed to get AI response: {e}"}), 500

# --- Removed Trade Log Endpoints (now handled by frontend Firestore, or can be re-added if needed for backend storage) ---
# TRADES_FILE = 'trades.json' # These were for local JSON storage
# ANALYSIS_FILE = 'analysis_results.json' # These were for local JSON storage
# def init_db(): ... # No longer needed if moving to Firestore
# def read_trades(): ...
# def write_trades(trades): ...
# def read_analysis_results(): ...
# def write_analysis_results(results): ...
# @app.route('/log_trade', methods=['POST']) ...
# @app.route('/get_trades', methods=['GET']) ...
# @app.route('/get_trade_summary', methods=['GET']) ...


# --- ORMCR Analysis Endpoint ---

# Mapping for Bybit interval strings and data limits
BYBIT_INTERVAL_MAP = {
    "M1": {"interval": "1", "limit": 1000}, # Max limit supported by Bybit
    "M5": {"interval": "5", "limit": 1000}, # Max limit supported by Bybit
    "M15": {"interval": "15", "limit": 1000}, # Max limit supported by Bybit
    "M30": {"interval": "30", "limit": 1000}, # Max limit supported by Bybit
    "H1": {"interval": "60", "limit": 1000}, # Max limit supported by Bybit
    "H4": {"interval": "240", "limit": 1000}, # Max limit supported by Bybit
    "D1": {"interval": "D", "limit": 1000}, # Max limit supported by Bybit
}

def fetch_real_ohlcv(symbol, interval_key):
    """Fetches real OHLCV data from Bybit for a given symbol and interval."""
    if not bybit_client:
        app.logger.error("Bybit client not initialized. Cannot fetch real data for ORMCR.")
        return pd.DataFrame()

    bybit_interval = BYBIT_INTERVAL_MAP[interval_key]["interval"]
    limit = BYBIT_INTERVAL_MAP[interval_key]["limit"]

    try:
        kline_response = bybit_client.get_kline(
            category="spot", # Assuming spot. Adjust if using linear/inverse for ORMCR.
            symbol=symbol,
            interval=bybit_interval,
            limit=limit
        )
        kline_data = kline_response.get('result', {}).get('list', [])

        if not kline_data:
            app.logger.warning(f"No kline data found for {symbol} on {interval_key} interval.")
            return pd.DataFrame()

        df = pd.DataFrame(kline_data, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        
        # --- CRITICAL FIX: Ensure all OHLCV and volume columns are numeric ---
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['start'] = pd.to_numeric(df['start'])
        df['start'] = pd.to_datetime(df['start'], unit='ms')
        df.set_index('start', inplace=True)
        df = df.iloc[::-1] # Reverse to be OLDEST TO NEWEST

        # --- DEBUG: Verify dtypes after conversion in fetch_real_ohlcv ---
        app.logger.info(f"\n--- DEBUG: fetch_real_ohlcv for {symbol} ({interval_key}) dtypes after numeric conversion ---")
        df.info(verbose=True, buf=sys.stdout)
        app.logger.info("------------------------------------------------------------------")

        if df[numeric_cols].isnull().any().any():
            app.logger.warning(f"WARNING: fetch_real_ohlcv for {symbol} ({interval_key}) has NaN values in OHLCV or volume columns after conversion.")

        return df

    except Exception as e:
        app.logger.error(f"Error fetching real OHLCV data for {symbol} ({interval_key}): {e}")
        traceback.print_exc()
        return pd.DataFrame()


def calculate_indicators_for_df(df, indicators):
    """
    Calculates selected indicators for a DataFrame.
    NOTE: Removed .ta calls as pandas_ta is not installed.
    Indicators will currently not be calculated via pandas_ta.
    """
    df_copy = df.copy()
    
    # --- REMOVED PANDAS_TA CALLS from calculate_indicators_for_df ---
    # These functions would normally call df_copy.ta.rsi(), etc.
    # Since pandas_ta is not installed, these calls are removed.
    # If you want to re-introduce indicator calculations without pandas_ta,
    # you would add custom logic here or integrate a different library.

    # if "RSI" in indicators:
    #     df_copy.ta.rsi(append=True)
    # if "MACD" in indicators:
    #     df_copy.ta.macd(append=True)
    # if "Moving Averages" in indicators:
    #     df_copy.ta.ema(length=9, append=True)
    #     df_copy.ta.sma(length=20, append=True)
    #     df_copy.ta.ema(length=50, append=True)
    # if "Bollinger Bands" in indicators:
    #     df_copy.ta.bbands(append=True)
    # if "Stochastic Oscillator" in indicators:
    #     df_copy.ta.stoch(append=True)
    # if "ATR" in indicators:
    #     df_copy.ta.atr(append=True)

    # For now, ensure these columns exist even if empty/NaN to avoid KeyError later in get_indicator_value
    placeholder_cols = [
        'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 
        'EMA_9', 'SMA_20', 'EMA_50',
        'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0', 'BBB_5_2.0', 'BBP_5_2.0',
        'STOCHk_14_3_3', 'STOCHd_14_3_3',
        'ATR_14', 'ATRr_14'
    ]
    for col in placeholder_cols:
        if col not in df_copy.columns:
            df_copy[col] = np.nan # Add as NaN if not already present from Bybit data (e.g., 'volume')

    app.logger.info(f"\n--- DEBUG: ORMCR Analysis ALL COLUMNS after (no) pandas_ta: ---")
    app.logger.info(df_copy.columns.tolist())
    app.logger.info("------------------------------------------------------------------")

    app.logger.info(f"\n--- DEBUG: ORMCR Analysis DataFrame info after (no) pandas_ta for columns: {df_copy.columns.tolist()} ---")
    df_copy.info(verbose=False, buf=sys.stdout)
    app.logger.info("------------------------------------------------------------------")

    app.logger.info("\n--- DEBUG: ORMCR Analysis DataFrame Tail for Critical Indicators after (no) pandas_ta calculation ---")
    selected_cols = ['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9', 'ATR_14', 'RSI_14', 'STOCHk_14_3_3', 'EMA_9']
    existing_cols = [col for col in selected_cols if col in df_copy.columns]
    if existing_cols:
        tail_data = df_copy[existing_cols].tail(20)
        app.logger.info(tail_data)
        if tail_data.iloc[-1].isnull().any().any():
            app.logger.warning("WARNING: ORMCR Analysis Last row of selected indicators contains NaN values.")
    else:
        app.logger.info("No selected indicator columns found in ORMCR Analysis DataFrame after (no) calculation.")
    app.logger.info("------------------------------------------------------------------")

    return df_copy

def apply_ormcr_logic(analysis_data, selected_indicators_from_frontend):
    """
    Applies ORMCR logic to the analysis data, dynamically considering selected indicators
    and handling N/A values more gracefully for confidence score calculation.
    NOTE: Indicator values from df will be N/A since pandas_ta is not used.
    The logic will primarily rely on price action and 'N/A' for indicators.
    """
    overall_bias = "NEUTRAL"
    confirmation_status = "PENDING"
    confirmation_reason = "Initial analysis." 
    entry_suggestion = "MONITOR"
    sl_price = "N/A"
    tp1_price = "N/A"
    tp2_price = "N/A"
    risk_in_points = "N/A"
    position_size_suggestion = "User to calculate"
    sl_percent_change = "N/A"
    tp1_percent_change = "N/A"
    tp2_percent_change = "N/A"
    
    calculated_confidence_score = 0
    calculated_signal_strength = "NEUTRAL"

    sorted_timeframes = sorted(analysis_data.keys(), key=lambda x: int(BYBIT_INTERVAL_MAP.get(x, {"interval": "0"}).get("interval")) if BYBIT_INTERVAL_MAP[x]["interval"].isdigit() else 999999, reverse=True)
    
    lowest_tf_key_for_calc = sorted_timeframes[-1] if sorted_timeframes else None

    app.logger.info(f"\n--- Starting ORMCR Logic ---")
    app.logger.info(f"Sorted Timeframes: {sorted_timeframes}")
    app.logger.info(f"Lowest Timeframe for Calc: {lowest_tf_key_for_calc}")
    app.logger.info(f"Selected Indicators from Frontend: {selected_indicators_from_frontend}")

    trend_analysis = {}
    for tf in sorted_timeframes:
        df = analysis_data[tf]['df']
        if df.empty or len(df) < 2: 
            trend_analysis[tf] = {"trend": "No sufficient data", "last_close": "N/A", "ema9": "N/A", "rsi": "N/A", "macd_hist": "N/A"}
            app.logger.info(f"  {tf}: No sufficient data for trend analysis.")
            continue

        last_close = get_indicator_value(df, 'close')
        ema9 = get_indicator_value(df, 'EMA_9') # Will be N/A
        rsi = get_indicator_value(df, 'RSI_14') # Will be N/A
        macd_hist = get_indicator_value(df, 'MACDh_12_26_9') # Will be N/A

        tf_trend = "Neutral"
        # Trend logic will primarily result in Neutral if indicators are N/A
        if "Moving Averages" in selected_indicators_from_frontend and isinstance(ema9, (int, float)) and isinstance(last_close, (int, float)) and 'EMA_9' in df.columns:
            if len(df) > 1 and not pd.isna(df['EMA_9'].iloc[-2]) and not pd.isna(df['close'].iloc[-2]):
                prev_ema9 = df['EMA_9'].iloc[-2]
                prev_close = df['close'].iloc[-2]

                if last_close > ema9 and prev_close > prev_ema9:
                    tf_trend = "Uptrend"
                elif last_close < ema9 and prev_close < prev_ema9:
                    tf_trend = "Downtrend"
            elif last_close > ema9:
                tf_trend = "Uptrend (weak)"
            elif last_close < ema9:
                tf_trend = "Downtrend (weak)"
        else:
            tf_trend = "Neutral (EMA9 data not available or not selected)"

        trend_analysis[tf] = {
            "trend": tf_trend,
            "last_close": last_close,
            "ema9": ema9,
            "rsi": rsi,
            "macd_hist": macd_hist
        }
        app.logger.info(f"  {tf} Trend Analysis: {trend_analysis[tf]}")
    
    for tf_key in ["D1", "H4", "H1", "M30", "M15", "M5", "M1"]:
        if tf_key in trend_analysis and trend_analysis[tf_key]["trend"] not in ["Neutral", "No sufficient data", "Uptrend (weak)", "Downtrend (weak)", "Neutral (EMA9 data not available or not selected)"]:
            overall_bias = trend_analysis[tf_key]["trend"].upper()
            app.logger.info(f"  Overall Bias determined from {tf_key}: {overall_bias}")
            break
    app.logger.info(f"Final Overall Bias: {overall_bias}")

    confirmation_details = []
    
    if lowest_tf_key_for_calc and lowest_tf_key_for_calc in analysis_data and not analysis_data[lowest_tf_key_for_calc]['df'].empty:
        df_lowest = analysis_data[lowest_tf_key_for_calc]['df']
        
        if len(df_lowest) < 2:
            confirmation_reason = f"Not enough historical data ({len(df_lowest)} candles) for lowest timeframe ({lowest_tf_key_for_calc}) for detailed confirmation. At least 2 candles are needed."
            calculated_confidence_score = 30
            calculated_signal_strength = "NEUTRAL"
            entry_suggestion = "MONITOR"
            confirmation_status = "PENDING"
            app.logger.info(f"  Confirmation Reason: {confirmation_reason}")
            return {
                "overall_bias": overall_bias,
                "confirmation_status": confirmation_status,
                "confirmation_reason": confirmation_reason,
                "entry_suggestion": entry_suggestion,
                "sl_price": sl_price, "tp1_price": tp1_price, "tp2_price": tp2_price, "risk_in_points": risk_in_points,
                "position_size_suggestion": position_size_suggestion,
                "calculated_confidence_score": calculated_confidence_score,
                "calculated_signal_strength": calculated_signal_strength,
                "sl_percent_change": sl_percent_change,
                "tp1_percent_change": tp1_percent_change,
                "tp2_percent_change": tp2_percent_change,
                "trend_analysis_by_tf": {tf: {k: v for k, v in data.items() if k != 'df'} for tf, data in analysis_data.items()}
            }

        last_close_lowest = get_indicator_value(df_lowest, 'close')
        prev_close_lowest = df_lowest['close'].iloc[-2] if len(df_lowest) > 1 and not pd.isna(df_lowest['close'].iloc[-2]) else np.nan

        ema9_lowest = get_indicator_value(df_lowest, 'EMA_9') # N/A
        rsi_lowest = get_indicator_value(df_lowest, 'RSI_14') # N/A
        macd_lowest = get_indicator_value(df_lowest, 'MACD_12_26_9') # N/A
        macds_lowest = get_indicator_value(df_lowest, 'MACDs_12_26_9') # N/A
        macdh_lowest = get_indicator_value(df_lowest, 'MACDh_12_26_9') # N/A
        stoch_k_lowest = get_indicator_value(df_lowest, 'STOCHk_14_3_3') # N/A
        stoch_d_lowest = get_indicator_value(df_lowest, 'STOCHd_14_3_3') # N/A
        atr_lowest = get_indicator_value(df_lowest, 'ATR_14') # N/A

        app.logger.info(f"  Lowest TF ({lowest_tf_key_for_calc}) Indicator Values:")
        app.logger.info(f"    Last Close: {last_close_lowest}, Prev Close: {prev_close_lowest}")
        app.logger.info(f"    EMA9: {ema9_lowest}, RSI: {rsi_lowest}")
        app.logger.info(f"    MACD: {macd_lowest}, MACDS: {macds_lowest}, MACDH: {macdh_lowest}, STOCH_K: {stoch_k_lowest}, STOCH_D: {stoch_d_lowest}, ATR: {atr_lowest}")

        directional_signals_count = 0
        total_relevant_indicators = 0 

        # Condition 1: Price Action (Strong Candle)
        if isinstance(last_close_lowest, (int, float)) and isinstance(prev_close_lowest, (int, float)):
            total_relevant_indicators += 1
            if last_close_lowest > prev_close_lowest * 1.001:
                directional_signals_count += 1
                confirmation_details.append("Price action: Strong bullish candle detected.")
            elif last_close_lowest < prev_close_lowest * 0.999:
                directional_signals_count += 1
                confirmation_details.append("Price action: Strong bearish candle detected.")
            else:
                confirmation_details.append("Price action: Neutral candle (no strong directional movement).")
        else:
            confirmation_details.append("Price action: Data missing or invalid for candle analysis.")


        # Conditions 2-6 (EMA9, RSI, MACD, Stochastic, Bollinger Bands) will be N/A due to pandas_ta removal.
        # The logic here will mainly default to 'Data missing or not selected'
        # or rely on the `isinstance` checks returning False for non-numeric 'N/A' values.
        # This will reduce `directional_signals_count` if these indicators were selected.

        # Condition 2: Price vs. EMA9
        if "Moving Averages" in selected_indicators_from_frontend and isinstance(ema9_lowest, (int, float)) and isinstance(last_close_lowest, (int, float)):
            total_relevant_indicators += 1
            if last_close_lowest > ema9_lowest:
                directional_signals_count += 1
                confirmation_details.append("EMA9: Price above EMA9 (bullish alignment).")
            elif last_close_lowest < ema9_lowest:
                directional_signals_count += 1
                confirmation_details.append("EMA9: Price below EMA9 (bearish alignment).")
            else:
                confirmation_details.append("EMA9: Price is neutral around EMA9.")
        elif "Moving Averages" in selected_indicators_from_frontend:
            confirmation_details.append("EMA9: Data missing or not selected.")


        # Condition 3: RSI
        if "RSI" in selected_indicators_from_frontend and isinstance(rsi_lowest, (int, float)):
            total_relevant_indicators += 1
            if rsi_lowest > 60:
                directional_signals_count += 1
                confirmation_details.append("RSI: Overbought (>60), indicating strong recent buying pressure.")
            elif rsi_lowest < 40:
                directional_signals_count += 1
                confirmation_details.append("RSI: Oversold (<40), indicating strong recent selling pressure.")
            else:
                confirmation_details.append("RSI: Neutral (40-60).")
        elif "RSI" in selected_indicators_from_frontend:
            confirmation_details.append("RSI: Data missing or not selected.")


        # Condition 4: MACD Crossover / Momentum
        if "MACD" in selected_indicators_from_frontend and isinstance(macd_lowest, (int, float)):
            total_relevant_indicators += 1
            if isinstance(macds_lowest, (int, float)):
                if macd_lowest > macds_lowest:
                    directional_signals_count += 1
                    confirmation_details.append("MACD: Bullish crossover (MACD line > Signal line).")
                elif macd_lowest < macds_lowest:
                    directional_signals_count += 1
                    confirmation_details.append("MACD: Bearish crossover (MACD line < Signal line).")
                else:
                    confirmation_details.append("MACD: No clear crossover signal.")
            elif isinstance(macdh_lowest, (int, float)):
                if macdh_lowest > 0:
                    directional_signals_count += 1
                    confirmation_details.append("MACD: Histogram bullish (>0).")
                elif macdh_lowest < 0:
                    directional_signals_count += 1
                    confirmation_details.append("MACD: Histogram bearish (<0).")
                else:
                    confirmation_details.append("MACD: Histogram neutral.")
            else:
                confirmation_details.append("MACD: Data missing or invalid for crossover/histogram check.")
        elif "MACD" in selected_indicators_from_frontend:
            confirmation_details.append("MACD: Data missing or not selected.")


        # Condition 5: Stochastic Oscillator
        if "Stochastic Oscillator" in selected_indicators_from_frontend and \
           isinstance(stoch_k_lowest, (int, float)) and isinstance(stoch_d_lowest, (int, float)):
            total_relevant_indicators += 1
            if stoch_k_lowest > stoch_d_lowest and stoch_k_lowest < 80: 
                directional_signals_count += 1
                confirmation_details.append("Stochastic: Bullish crossover (K above D), not overbought.")
            elif stoch_k_lowest < stoch_d_lowest and stoch_k_lowest > 20: 
                directional_signals_count += 1
                confirmation_details.append("Stochastic: Bearish crossover (K below D), not oversold.")
            elif stoch_k_lowest >= 80:
                confirmation_details.append("Stochastic: Overbought (>=80), potential reversal or strong trend continuation.")
            elif stoch_k_lowest <= 20:
                confirmation_details.append("Stochastic: Oversold (<=20), potential reversal or strong trend continuation.")
            else:
                confirmation_details.append("Stochastic: Neutral (no clear crossover or in extreme zone).")
            
        elif "Stochastic Oscillator" in selected_indicators_from_frontend:
            confirmation_details.append("Stochastic: Data missing or not selected.")

        # Condition 6: Bollinger Bands
        if "Bollinger Bands" in selected_indicators_from_frontend:
            bb_upper = get_indicator_value(df_lowest, 'BBU_5_2.0')
            bb_lower = get_indicator_value(df_lowest, 'BBL_5_2.0')
            if isinstance(bb_upper, (int, float)) and isinstance(bb_lower, (int, float)) and isinstance(last_close_lowest, (int, float)):
                total_relevant_indicators += 1
                if last_close_lowest > bb_upper:
                    directional_signals_count += 1
                    confirmation_details.append("Bollinger Bands: Price broke above Upper Band (strong bullish move).")
                elif last_close_lowest < bb_lower:
                    directional_signals_count += 1
                    confirmation_details.append("Bollinger Bands: Price broke below Lower Band (strong bearish move).")
                else:
                    confirmation_details.append("Bollinger Bands: Price within bands (neutral/ranging).")
            else:
                confirmation_details.append("Bollinger Bands: Data missing or not selected.")

        # Condition 7: Volume - from raw data, but logic here assumes indicator usage
        if "Volume" in selected_indicators_from_frontend:
            volume_val = get_indicator_value(df_lowest, 'volume')
            if isinstance(volume_val, (int, float)) and volume_val > 0:
                if len(df_lowest) >= 5 and 'volume' in df_lowest.columns:
                    recent_avg_volume = df_lowest['volume'].iloc[-5:-1].mean()
                    if volume_val > recent_avg_volume * 1.5:
                        directional_signals_count += 1
                        confirmation_details.append(f"Volume: Significant increase ({volume_val} > 1.5x average).")
                    elif volume_val < recent_avg_volume * 0.5:
                        directional_signals_count += 1
                        confirmation_details.append(f"Volume: Significant decrease ({volume_val} < 0.5x average).")
                    else:
                        confirmation_details.append("Volume: Normal levels.")
                else:
                    confirmation_details.append("Volume: Insufficient data for comparative analysis.")
            else:
                confirmation_details.append("Volume: Data missing or not selected.")
        elif "Volume" in selected_indicators_from_frontend:
            confirmation_details.append("Volume: Indicator not selected or data not available.")
        
        if total_relevant_indicators > 0:
            calculated_confidence_score = int((directional_signals_count / total_relevant_indicators) * 100)
        else:
            calculated_confidence_score = 30
            confirmation_details.append("No relevant ORMCR indicators available or selected for lowest timeframe confirmation, defaulting confidence to 30%.")

        app.logger.info(f"  Directional Signals Count: {directional_signals_count}")
        app.logger.info(f"  Total Relevant Indicators: {total_relevant_indicators}")
        app.logger.info(f"  Calculated Confidence Score (based on directional signals): {calculated_confidence_score}%")

        if overall_bias == "NEUTRAL":
            if calculated_confidence_score >= 60:
                num_bullish_details = sum(1 for detail in confirmation_details if "bullish" in detail.lower() or "overbought" in detail.lower() or "above EMA9" in detail)
                num_bearish_details = sum(1 for detail in confirmation_details if "bearish" in detail.lower() or "oversold" in detail.lower() or "below EMA9" in detail)

                if num_bullish_details > num_bearish_details:
                    calculated_signal_strength = "CAUTIOUS BUY"
                    entry_suggestion = "MONITOR (Cautious bullish signal on lowest TF)"
                    confirmation_status = "NEUTRAL BIAS, CAUTIOUS SIGNAL"
                elif num_bearish_details > num_bullish_details:
                    calculated_signal_strength = "CAUTIOUS SELL"
                    entry_suggestion = "MONITOR (Cautious bearish signal on lowest TF)"
                    confirmation_status = "NEUTRAL BIAS, CAUTIOUS SIGNAL"
                else:
                    calculated_signal_strength = "NEUTRAL"
                    entry_suggestion = "MONITOR"
                    confirmation_status = "PENDING"
            else:
                calculated_signal_strength = "NEUTRAL"
                entry_suggestion = "MONITOR"
                confirmation_status = "PENDING"
        elif calculated_confidence_score < 40:
            calculated_signal_strength = "NEUTRAL"
            entry_suggestion = "MONITOR"
            confirmation_status = "PENDING"
        elif overall_bias == "UPTREND":
            if calculated_confidence_score >= 80:
                calculated_signal_strength = "STRONG BUY"
                entry_suggestion = "ENTER NOW"
                confirmation_status = "STRONG CONFIRMATION"
            elif calculated_confidence_score >= 60:
                calculated_signal_strength = "BUY"
                entry_suggestion = "ENTER NOW"
                confirmation_status = "CONFIRMED"
            else:
                calculated_signal_strength = "MONITOR"
                entry_suggestion = "MONITOR"
                confirmation_status = "PENDING"
        elif overall_bias == "DOWNTREND":
            if calculated_confidence_score >= 80:
                calculated_signal_strength = "STRONG SELL"
                entry_suggestion = "ENTER NOW"
                confirmation_status = "STRONG CONFIRMATION"
            elif calculated_confidence_score >= 60:
                calculated_signal_strength = "SELL"
                entry_suggestion = "ENTER NOW"
                confirmation_status = "CONFIRMED"
            else:
                calculated_signal_strength = "MONITOR"
                entry_suggestion = "MONITOR"
                confirmation_status = "PENDING"

        if entry_suggestion == "ENTER NOW" and isinstance(last_close_lowest, (int, float)):
            if isinstance(atr_lowest, (int, float)) and atr_lowest > 0: # This will be N/A, so fallback will be used
                app.logger.info(f"DEBUG: Using ATR ({atr_lowest}) for SL/TP calculation. (This will not run due to N/A ATR)")
                pass # This block is skipped as ATR will be N/A
            else: # Fallback to fixed percentages if ATR is not available or invalid
                confirmation_details.append("ATR data not available or invalid for dynamic SL/TP, using fixed percentages.")
                app.logger.info(f"DEBUG: Falling back to fixed percentages for SL/TP as ATR is not valid ({atr_lowest}).")
                if calculated_signal_strength in ["BUY", "STRONG BUY", "CAUTIOUS BUY"]:
                    sl_price = round(last_close_lowest * 0.995, 2)
                    tp1_price = round(last_close_lowest * 1.01, 2)
                    tp2_price = round(last_close_lowest * 1.02, 2)
                elif calculated_signal_strength in ["SELL", "STRONG SELL", "CAUTIOUS SELL"]:
                    sl_price = round(last_close_lowest * 1.005, 2)
                    tp1_price = round(last_close_lowest * 0.99, 2)
                    tp2_price = round(last_close_lowest * 0.98, 2)
                
                if last_close_lowest != 0:
                    sl_percent_change = round(((sl_price - last_close_lowest) / last_close_lowest) * 100, 2)
                    tp1_percent_change = round(((tp1_price - last_close_lowest) / last_close_lowest) * 100, 2)
                    tp2_percent_change = round(((tp2_price - last_close_lowest) / last_close_lowest) * 100, 2)
                else:
                    sl_percent_change, tp1_percent_change, tp2_percent_change = "N/A", "N/A", "N/A"

                risk_in_points = round(abs(last_close_lowest - sl_price), 2)
                position_size_suggestion = "Calculated based on 2.5% risk (example)"

        else: # If not "ENTER NOW" signal, SL/TP are N/A
            sl_price = "N/A"
            tp1_price = "N/A"
            tp2_price = "N/A"
            risk_in_points = "N/A"
            position_size_suggestion = "User to calculate"
            sl_percent_change, tp1_percent_change, tp2_percent_change = "N/A", "N/A", "N/A"
        
        app.logger.info(f"  Confidence Score: {calculated_confidence_score}%")
        app.logger.info(f"  Signal Strength: {calculated_signal_strength}")
        app.logger.info(f"  Confirmation Status: {confirmation_status}")
    else:
        confirmation_reason = "No valid data could be fetched for the lowest selected timeframe for detailed confirmation."
        calculated_confidence_score = 30
        calculated_signal_strength = "NEUTRAL"
        entry_suggestion = "MONITOR"
        confirmation_status = "PENDING"
        sl_price = "N/A"
        tp1_price = "N/A"
        tp2_price = "N/A"
        risk_in_points = "N/A"
        position_size_suggestion = "User to calculate"
        sl_percent_change, tp1_percent_change, tp2_percent_change = "N/A", "N/A", "N/A"

        app.logger.info(f"  Confirmation Reason: {confirmation_reason}")
        
    app.logger.info(f"--- ORMCR Logic Finished ---")
    
    final_confirmation_reason = ". ".join([detail for detail in confirmation_details if detail])
    if not final_confirmation_reason:
        final_confirmation_reason = "No specific confirmation details generated."

    return {
        "overall_bias": overall_bias,
        "confirmation_status": confirmation_status,
        "confirmation_reason": final_confirmation_reason,
        "entry_suggestion": entry_suggestion,
        "sl_price": sl_price,
        "tp1_price": tp1_price,
        "tp2_price": tp2_price,
        "risk_in_points": risk_in_points,
        "position_size_suggestion": position_size_suggestion,
        "calculated_confidence_score": calculated_confidence_score,
        "calculated_signal_strength": calculated_signal_strength,
        "sl_percent_change": sl_percent_change, 
        "tp1_percent_change": tp1_percent_change,
        "tp2_percent_change": tp2_percent_change,
        "trend_analysis_by_tf": {tf: {k: v for k, v in data.items() if k != 'df'} for tf, data in analysis_data.items()}
    }

@app.route('/run_ormcr_analysis', methods=['POST'])
def run_ormcr_analysis():
    # Access global models
    global gemini_model, bybit_client

    if not gemini_model:
        app.logger.error("Gemini model not initialized for ORMCR analysis endpoint.")
        return jsonify({"error": "Gemini model not initialized"}), 500
    if not bybit_client:
        app.logger.error("Bybit client not initialized for ORMCR analysis endpoint.")
        return jsonify({"error": "Bybit client not initialized"}), 500

    data = request.json
    currency_pair = data.get('currencyPair')
    timeframes = data.get('timeframes', [])
    trade_type = data.get('tradeType')
    indicators = data.get('indicators', [])
    available_balance = data.get('availableBalance')
    leverage = data.get('leverage')

    if not currency_pair or not timeframes:
        return jsonify({"error": "Currency pair and at least one timeframe are required"}), 400

    valid_timeframes = [tf for tf in timeframes if tf in BYBIT_INTERVAL_MAP]
    if not valid_timeframes:
        return jsonify({"error": "No valid timeframes provided"}), 400
    
    valid_timeframes.sort(key=lambda x: int(BYBIT_INTERVAL_MAP[x]["interval"]) if BYBIT_INTERVAL_MAP[x]["interval"].isdigit() else 999999, reverse=True)


    analysis_data_by_tf = {}
    bybit_symbol = currency_pair.replace('/', '') + 'T'

    for tf in valid_timeframes:
        df = fetch_real_ohlcv(bybit_symbol, tf)
        
        if df.empty:
            app.logger.warning(f"Skipping {tf} due to empty DataFrame after fetching real data.")
            continue

        # df_with_indicators will contain placeholder columns, not calculated values
        df_with_indicators = calculate_indicators_for_df(df, indicators)
        
        last_price_val = get_indicator_value(df_with_indicators, 'close')
        volume_val = get_indicator_value(df_with_indicators, 'volume')


        analysis_data_by_tf[tf] = {
            "df": df_with_indicators,
            "last_price": last_price_val,
            "volume": volume_val
        }
        time.sleep(0.2) # Small delay to avoid hitting API rate limits

    if not analysis_data_by_tf:
        return jsonify({"error": "No valid market data could be fetched for analysis. Please check currency pair and timeframes."}), 500


    ormcr_results = apply_ormcr_logic(analysis_data_by_tf, indicators)

    detailed_tf_data_for_prompt = {}
    for tf, data in analysis_data_by_tf.items():
        indicators_snapshot_dict = {}
        # Ensure these are now correctly referencing placeholder columns that will be N/A
        indicator_cols = [
            'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 
            'EMA_9', 'SMA_20', 'EMA_50',
            'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0', 'BBB_5_2.0', 'BBP_5_2.0',
            'STOCHk_14_3_3', 'STOCHd_14_3_3',
            'ATR_14', 'ATRr_14',
            'volume'
        ]
        for ind_col in indicator_cols:
            indicators_snapshot_dict[ind_col] = get_indicator_value(data["df"], ind_col)
        
        detailed_tf_data_for_prompt[tf] = {
            "last_price": data["last_price"],
            "volume": data["volume"],
            "indicators_snapshot": indicators_snapshot_dict
        }
    
    app.logger.info(f"\n--- Indicators Snapshot before sending to Gemini ---")
    app.logger.info(json.dumps(detailed_tf_data_for_prompt, indent=2))
    app.logger.info("-" * 40)


    prompt_parts = [
        "You are Aura, an advanced AI trading assistant specializing in the ORMCR strategy.",
        "Your goal is to provide insightful, helpful, and **friendly** analysis results. Adopt a **conversational, approachable, and encouraging tone**. Avoid overly formal or robotic language.",
        "You can use emojis if appropriate to convey friendliness (e.g., 🚀🌟).",
        "The user has requested an analysis for:",
        f"- Currency Pair: {currency_pair}",
        f"- Selected Timeframes (highest to lowest): {', '.join(valid_timeframes)}",
        f"- Intended Trade Type: {trade_type}",
        f"- Selected Indicators: {', '.join(indicators) if indicators else 'None'}",
        f"- Available Balance: ${available_balance}",
        f"- Leverage: {leverage}",
        "\nHere is the detailed market data and ORMCR analysis results from our internal system:",
        json.dumps({
            "ormcr_analysis": ormcr_results,
            "detailed_timeframe_data": detailed_tf_data_for_prompt
        }, indent=2),
        "\nBased on this information, provide a comprehensive AI ANALYSIS RESULTS. Follow this exact structure:",
        """
        {
            "confidence_score": "X%",
            "signal_strength": "STRONG BUY/BUY/NEUTRAL/SELL/STRONG SELL/CAUTIOUS BUY/CAUTIOUS SELL",
            "market_summary": "Detailed summary based on multi-timeframe trend, price action, and key indicators. Explain the overall bias (e.g., 'The D1 timeframe shows an Uptrend...'), and then discuss the lowest timeframe's alignment with this bias or its own short-term signals. Be aware that indicator values may be N/A if not calculated by the backend.",
            "ai_suggestion": {
                "entry_type": "BUY ORDER/SELL ORDER/WAIT",
                "recommended_action": "ENTER NOW/MONITOR/AVOID",
                "position_size": "X% of balance (e.g., based on 2.5% risk tolerance, or User to calculate)"
            },
            "stop_loss": {
                "price": "$X.XX",
                "percentage_change": "X.XX%"
            },
            "take_profit_1": {
                "price": "$X.XX",
                "percentage_change": "X.XX%"
            },
            "take_profit_2": {
                "price": "$X.XX",
                "percentage_change": "X.XX%"
            },
            "technical_indicators_analysis": "Based on the 'indicators_snapshot' provided in 'detailed_timeframe_data', interpret the values for all selected indicators (RSI, MACD, Stochastic, Moving Averages, Bollinger Bands, Volume, ATR, Fibonacci Retracements if applicable) across the relevant timeframes, focusing on the lowest timeframe for potential entry signals. Explain what each indicator suggests about market conditions (e.g., overbought/oversold for RSI, momentum for MACD, volatility for Bollinger Bands, etc.). For each indicator that was selected, explain its reading and what it implies for the current market state. **Crucially, if an indicator's value is 'N/A' (because pandas_ta is not used), clearly state that it was not calculated by the backend and therefore not included in the numerical analysis, but you can still provide general context for what it normally signifies.**",
            "next_step_for_user": "What the user should do next (e.g., 'Monitor for confirmation', 'Proceed with the trade', 'Review other timeframes', 'Consider a cautious entry')."
        }
        """,
        "\n**IMPORTANT GUIDELINES FOR YOUR RESPONSE:**",
        f"- The 'confidence_score' should be '{ormcr_results['calculated_confidence_score']}%' based on the backend's calculation.",
        f"- The 'signal_strength' should be '{ormcr_results['calculated_signal_strength']}' based on the backend's calculation.",
        f"- The 'entry_type' and 'recommended_action' in 'ai_suggestion' should directly map to '{ormcr_results['entry_suggestion']}'.",
        f"- For stop_loss and take_profit, use the calculated 'sl_price', 'tp1_price', 'tp2_price' and their corresponding 'sl_percent_change', 'tp1_percent_change', 'tp2_percent_change' from 'ormcr_analysis'. If these are 'N/A', represent them as 'N/A'.",
        f"- If 'ormcr_confirmation_status' is 'PENDING', ensure 'entry_type' is 'WAIT' and 'recommended_action' is 'MONITOR', and provide 'N/A' for SL/TP prices and percentages.",
        f"- Ensure 'market_summary' and 'next_step_for_user' are coherent and actionable based on the analysis.",
        "\n**IMPORTANT: Maintain a friendly, conversational, and encouraging tone throughout your response. Use simple, clear language and feel free to include relevant emojis to enhance friendliness (e.g., 🚀🌟).**"
    ]

    try:
        response = gemini_model.generate_content(
            prompt_parts,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "OBJECT",
                    "properties": {
                        "confidence_score": {"type": "STRING"},
                        "signal_strength": {"type": "STRING"},
                        "market_summary": {"type": "STRING"},
                        "ai_suggestion": {
                            "type": "OBJECT",
                            "properties": {
                                "entry_type": {"type": "STRING"},
                                "recommended_action": {"type": "STRING"},
                                "position_size": {"type": "STRING"}
                            }
                        },
                        "stop_loss": {
                            "type": "OBJECT",
                            "properties": {
                                "price": {"type": "STRING"},
                                "percentage_change": {"type": "STRING"}
                            }
                        },
                        "take_profit_1": {
                            "type": "OBJECT",
                            "properties": {
                                "price": {"type": "STRING"},
                                "percentage_change": {"type": "STRING"}
                            }
                        },
                        "take_profit_2": {
                            "type": "OBJECT",
                            "properties": {
                                "price": {"type": "STRING"},
                                "percentage_change": {"type": "STRING"}
                            }
                        },
                        "technical_indicators_analysis": {"type": "STRING"},
                        "next_step_for_user": {"type": "STRING"}
                    },
                    "required": ["confidence_score", "signal_strength", "market_summary", "ai_suggestion", "stop_loss", "take_profit_1", "take_profit_2", "technical_indicators_analysis", "next_step_for_user"]
                }
            }
        )
        
        ai_analysis_results = json.loads(response.candidates[0].content.parts[0].text)

        # Append ORMCR backend results for frontend to display
        ai_analysis_results['ormcr_confirmation_status'] = ormcr_results['confirmation_status']
        ai_analysis_results['ormcr_overall_bias'] = ormcr_results['overall_bias']
        ai_analysis_results['ormcr_reason'] = ormcr_results['confirmation_reason'] # This is the detailed reason from backend

        # Ensure SL/TP values match backend calculations, especially if N/A
        ai_analysis_results['stop_loss']['price'] = ormcr_results['sl_price']
        ai_analysis_results['stop_loss']['percentage_change'] = ormcr_results['sl_percent_change']
        ai_analysis_results['take_profit_1']['price'] = ormcr_results['tp1_price']
        ai_analysis_results['take_profit_1']['percentage_change'] = ormcr_results['tp1_percent_change']
        ai_analysis_results['take_profit_2']['price'] = ormcr_results['tp2_price']
        ai_analysis_results['take_profit_2']['percentage_change'] = ormcr_results['tp2_percent_change']
        
        # Ensure position_size reflects backend logic
        ai_analysis_results['ai_suggestion']['position_size'] = ormcr_results['position_size_suggestion']


        return jsonify(ai_analysis_results)

    except Exception as e:
        print(f"Error generating ORMCR analysis with Gemini: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Failed to get AI analysis: {e}"}), 500


if __name__ == '__main__':
    # This block is primarily for local development.
    # For Render deployment, Gunicorn will handle starting the Flask application.
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)), debug=False)