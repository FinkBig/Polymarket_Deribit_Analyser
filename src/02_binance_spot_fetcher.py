import requests
import pandas as pd
import time
from datetime import datetime, timezone, timedelta
import logging
import pathlib
import sys
import argparse # Keep for standalone execution
import pytz # Needed for standalone execution timezone handling
import re # Added import
import numpy as np # Added import
from typing import Union, Optional # Import Union and Optional

# --- Setup Logger ---
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s') # Set level in main script

# --- Constants ---
# PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1] # Define dynamically or pass paths
# BINANCE_DIR_DEFAULT = PROJECT_ROOT / 'data' / 'binance_data' # Define dynamically or pass paths
BINANCE_API_URL = "https://api.binance.com"
BINANCE_SYMBOL_DEFAULT = "BTCUSDT"
BINANCE_INTERVAL_DEFAULT = "1h"
REQUEST_DELAY_S = 0.25
API_RETRY_ATTEMPTS = 3
API_RETRY_WAIT_S = 3 # Longer wait for Binance potentially stricter limits
LIMIT_PER_REQUEST = 1000

# --- Helper Functions ---

def ensure_dir(dir_path: pathlib.Path):
    """Ensure the directory exists."""
    dir_path.mkdir(parents=True, exist_ok=True)

def _timestamp_to_milliseconds(dt_obj: datetime) -> int:
    """Converts a datetime object to milliseconds timestamp."""
    return int(dt_obj.timestamp() * 1000)

# *** CORRECTED TYPE HINT FOR PYTHON < 3.10 ***
def _interval_to_milliseconds(interval: str) -> Union[int, None]:
    """Converts Binance interval string to milliseconds duration."""
    pattern = re.compile(r'^(\d+)([mhd])$') # Simple pattern for m, h, d
    match = pattern.match(interval)
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        multipliers = {'m': 60, 'h': 3600, 'd': 86400}
        if unit in multipliers:
            return value * multipliers[unit] * 1000
    logger.error(f"Could not parse interval string: {interval}. Use format like '1h', '5m', '1d'.")
    return None


def _fetch_binance_klines_batch(symbol, interval, start_time_ms, end_time_ms, limit):
    """Fetches a single batch of klines from Binance API with retries."""
    url = f"{BINANCE_API_URL}/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time_ms,
        # Note: Binance endTime is inclusive, adjust if needed depending on how range is defined
        'endTime': end_time_ms -1 if end_time_ms > start_time_ms else end_time_ms , # Use end_time_ms - 1 to make it exclusive matching pandas ranges
        'limit': limit
    }
    # logger.debug(f"Requesting Binance URL: {url} with params: {params}")
    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            response = requests.get(url, params=params, timeout=20) # Consider adjusting timeout
            # Check for specific error codes before raising for status
            if response.status_code >= 400:
                try:
                    data = response.json()
                    if isinstance(data, dict) and 'code' in data:
                        logger.warning(f"Binance API Error: Code={data.get('code')}, Msg={data.get('msg')}")
                        if data.get('code') == -1121: # Invalid symbol
                             logger.error(f"Invalid symbol: {symbol}")
                             return None # Fatal error for this request
                        if data.get('code') == -1003: # Rate limit / IP ban
                            wait = API_RETRY_WAIT_S * (2 ** attempt) # Exponential backoff
                            logger.warning(f"Rate limit/IP block hit (-1003). Retrying in {wait:.1f}s...")
                            time.sleep(wait)
                            continue # Retry the request
                        # Add other specific error codes if needed
                        return None # Assume other errors are not retryable for now
                except requests.exceptions.JSONDecodeError:
                    logger.warning(f"Non-JSON error response (Status: {response.status_code}). Retrying...")
                    # Fall through to general retry logic

            response.raise_for_status() # Raise for other HTTP errors (e.g., 404, 500)
            data = response.json()
            if isinstance(data, list):
                return data
            else:
                # This case should ideally be caught by the error check above
                logger.error(f"Unexpected non-list successful response from Binance API: {data}")
                return None # Treat as error
        except requests.exceptions.Timeout:
            logger.warning(f"Request timed out (Attempt {attempt+1}/{API_RETRY_ATTEMPTS}).")
        except requests.exceptions.ConnectionError:
            logger.warning(f"Connection error (Attempt {attempt+1}/{API_RETRY_ATTEMPTS}).")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed (Attempt {attempt+1}/{API_RETRY_ATTEMPTS}): {e}.")
            if e.response is not None:
                logger.warning(f"Response status: {e.response.status_code}, Text: {e.response.text[:200]}...")

        # Wait before retrying (except for rate limit which has its own wait)
        if attempt + 1 < API_RETRY_ATTEMPTS:
            wait_time = API_RETRY_WAIT_S * (1.5 ** attempt) # Slightly gentler backoff
            logger.info(f"Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)
        else:
            logger.error(f"Max retries ({API_RETRY_ATTEMPTS}) reached for Binance request.")
            # Raise the last exception encountered to signal failure clearly
            if 'e' in locals(): raise e
            else: raise requests.exceptions.RequestException(f"Request failed after {API_RETRY_ATTEMPTS} retries.")

    return None # Should only be reached if loop completes without returning/raising

# --- Main Function (Callable by Orchestrator) ---

# *** CORRECTED TYPE HINT FOR PYTHON < 3.10 ***
def run_fetch(symbol: str, interval: str, start_dt: datetime, end_dt: datetime, output_dir: pathlib.Path) -> Union[pathlib.Path, None]:
    """
    Fetches Binance Klines (OHLCV) for a given symbol, interval, and date range.

    Args:
        symbol: Binance symbol (e.g., "BTCUSDT").
        interval: Binance interval string (e.g., "1h", "1d").
        start_dt: Start datetime (UTC, inclusive).
        end_dt: End datetime (UTC, exclusive).
        output_dir: Directory to save the output CSV file.

    Returns:
        Path to the output CSV file if successful, None otherwise.
    """
    logger.info(f"--- Starting Binance Spot Data Fetch ---")
    logger.info(f"Symbol: {symbol}, Interval: {interval}")
    logger.info(f"Range: {start_dt.strftime('%Y-%m-%d %H:%M')} to {end_dt.strftime('%Y-%m-%d %H:%M')} (exclusive)")
    logger.info(f"Output Directory: {output_dir}")

    # Validate inputs
    if start_dt >= end_dt:
        logger.error("Start date must be before end date.")
        return None
    if start_dt.tzinfo is None or end_dt.tzinfo is None:
        logger.warning("Input datetimes are timezone naive. Assuming UTC.")
        start_dt = pytz.utc.localize(start_dt) # Use pytz for localization
        end_dt = pytz.utc.localize(end_dt)
    elif start_dt.tzinfo != timezone.utc or end_dt.tzinfo != timezone.utc:
         logger.warning("Input datetimes are not UTC. Converting to UTC.")
         start_dt = start_dt.astimezone(pytz.utc) # Use pytz for conversion
         end_dt = end_dt.astimezone(pytz.utc)


    interval_ms = _interval_to_milliseconds(interval)
    if interval_ms is None:
        return None # Error already logged

    # Convert to milliseconds for API
    start_time_ms = _timestamp_to_milliseconds(start_dt)
    end_time_ms = _timestamp_to_milliseconds(end_dt)

    # Ensure output directory exists
    ensure_dir(output_dir)

    all_klines = []
    current_start_time_ms = start_time_ms

    # --- Fetching Loop ---
    # Calculate approximate total requests for progress bar
    total_duration_ms = end_time_ms - start_time_ms
    # Ensure interval_ms is not zero before division
    est_requests = max(1, int(np.ceil(total_duration_ms / (LIMIT_PER_REQUEST * interval_ms)))) if interval_ms > 0 else 1

    # Use tqdm only if import works
    pbar = None
    try:
        from tqdm.auto import tqdm
        pbar = tqdm(total=est_requests, desc=f"Fetching {symbol} {interval}", unit="batch")
    except ImportError:
        logger.info("tqdm not installed, progress bar disabled.")


    try: # Wrap fetching loop in try/finally for progress bar closing
        while current_start_time_ms < end_time_ms:
            # Calculate end time for this batch request
            batch_end_time_ms = min(current_start_time_ms + LIMIT_PER_REQUEST * interval_ms, end_time_ms)

            klines_batch = _fetch_binance_klines_batch(
                symbol,
                interval,
                current_start_time_ms,
                batch_end_time_ms, # Pass calculated end time for batch
                LIMIT_PER_REQUEST
            )
            time.sleep(REQUEST_DELAY_S) # Apply delay *after* request

            if klines_batch is None:
                logger.error("Failed to fetch a Binance batch after retries, stopping fetch.")
                # Return None here as fetch failed, even if we have partial data
                return None
            elif not klines_batch:
                logger.debug("Received empty batch, check start/end or assume no more data.")
                current_start_time_ms = batch_end_time_ms
                if pbar: pbar.update(1)
                if current_start_time_ms >= end_time_ms: break
                else: continue

            all_klines.extend(klines_batch)
            last_kline_open_time_ms = klines_batch[-1][0]
            next_start_time = last_kline_open_time_ms + interval_ms

            if next_start_time <= current_start_time_ms:
                logger.warning(f"Timestamp did not advance correctly (last: {last_kline_open_time_ms}, next_start: {next_start_time}). Check interval logic or API behavior. Breaking loop.")
                break

            if pbar: pbar.update(1)
            current_start_time_ms = next_start_time
    finally:
        if pbar: pbar.close() # Ensure progress bar closes


    if not all_klines:
        logger.warning("No Binance kline data fetched for the specified period.")
        output_path = output_dir / f"binance_spot_{symbol}_{interval}_{start_dt.strftime('%Y%m%d')}_{(end_dt - timedelta(days=1)).strftime('%Y%m%d')}_empty.csv"
        try:
            cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'number_of_trades']
            pd.DataFrame(columns=cols).to_csv(output_path, index=False)
            logger.info(f"Created empty output file: {output_path}")
            return output_path # Return path to empty file
        except Exception as e:
            logger.error(f"Failed to create empty output file: {e}")
            return None

    # --- Process and Save ---
    logger.info(f"Processing {len(all_klines)} fetched klines...")
    try:
        columns = [ # Standard Binance Kline columns
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        df = pd.DataFrame(all_klines, columns=columns)
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)

        # Ensure data is within the originally requested bounds (API might return slightly more)
        df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] < end_dt)]

        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'number_of_trades',
                        'quote_asset_volume', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        output_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'number_of_trades']
        df_output = df[output_cols].copy()
        df_output = df_output.drop_duplicates(subset=['timestamp'], keep='first')
        df_output = df_output.sort_values('timestamp').reset_index(drop=True)

        # Generate output filename
        start_date_str = start_dt.strftime('%Y%m%d')
        # Use end_dt - 1 day for inclusive range name if end_dt was exclusive
        end_date_str = (end_dt - timedelta(days=1)).strftime('%Y%m%d')
        out_name = f"binance_spot_{symbol}_{interval}_{start_date_str}_{end_date_str}.csv"
        output_path = output_dir / out_name

        df_output.to_csv(output_path, index=False, float_format="%.8f")
        logger.info(f"Saved {len(df_output)} processed klines to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to process or save Binance data: {e}", exc_info=True)
        return None

# --- Standalone Execution Logic ---
if __name__ == "__main__":
    # Setup logging for standalone run
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

    # Define PROJECT_ROOT for standalone execution default path
    try:
         # Assumes script is in src/, so parent.parent is thesis_project/
         # Adjust if script location is different relative to project root
         PROJECT_ROOT_STANDALONE = pathlib.Path(__file__).resolve().parent.parent
    except NameError:
         # Fallback if __file__ is not defined (e.g., interactive)
         PROJECT_ROOT_STANDALONE = pathlib.Path.cwd()
         logger.info(f"Could not determine project root reliably in standalone mode, using cwd: {PROJECT_ROOT_STANDALONE}")

    # Define default output dir based on project root (assuming thesis2.0 structure)
    BINANCE_DIR_DEFAULT = PROJECT_ROOT_STANDALONE / 'data' / 'binance_data'

    parser = argparse.ArgumentParser(description="Fetch Binance historical Klines (OHLCV).")
    parser.add_argument("start_date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("end_date", help="End date (YYYY-MM-DD), exclusive.")
    parser.add_argument("-s", "--symbol", default=BINANCE_SYMBOL_DEFAULT, help=f"Binance symbol (default: {BINANCE_SYMBOL_DEFAULT})")
    parser.add_argument("-i", "--interval", default=BINANCE_INTERVAL_DEFAULT, help=f"Kline interval (default: {BINANCE_INTERVAL_DEFAULT})")
    # Corrected default path definition using PROJECT_ROOT_STANDALONE
    parser.add_argument("-o", "--output-dir", default=str(BINANCE_DIR_DEFAULT), help=f"Output directory (default: {BINANCE_DIR_DEFAULT})")

    args = parser.parse_args()

    try:
        # Assume UTC for dates if no timezone specified
        start_dt = pytz.utc.localize(datetime.strptime(args.start_date, "%Y-%m-%d"))
        end_dt = pytz.utc.localize(datetime.strptime(args.end_date, "%Y-%m-%d"))
        output_dir = pathlib.Path(args.output_dir)

        # Call the main fetching function
        result_path = run_fetch(
            symbol=args.symbol,
            interval=args.interval,
            start_dt=start_dt,
            end_dt=end_dt,
            output_dir=output_dir
        )

        if result_path:
            logger.info(f"Standalone execution successful. Output: {result_path}")
            sys.exit(0)
        else:
            logger.error("Standalone execution failed.")
            sys.exit(1)

    except ValueError as e:
        logger.error(f"Error parsing dates. Please use YYYY-MM-DD format. Details: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)
