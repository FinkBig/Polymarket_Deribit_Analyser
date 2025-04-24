import requests
import pandas as pd
import logging
import pathlib
import sys
import time
import re # Import re
from datetime import datetime, timedelta, timezone
import argparse # Keep for potential standalone execution
import pytz # Import pytz
from typing import Union, List, Optional # Import Union and Optional

# --- Setup Logger ---
# Use __name__ for module-specific logger
logger = logging.getLogger(__name__)
# Default level, can be overridden by main script's config
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# --- Constants ---
# Assume PROJECT_ROOT is defined relative to this file's location if run standalone
# If imported, paths might need adjustment or be passed explicitly. Let's use explicit paths.
# PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1] # Get thesis2.0/
# DERIBIT_RAW_DIR_DEFAULT = PROJECT_ROOT / 'data' / 'deribit_data' / 'raw'
DERIBIT_CURRENCY_DEFAULT = "BTC"
DERIBIT_HISTORY_URL = "https://history.deribit.com/api/v2"
API_RETRY_ATTEMPTS = 3
API_RETRY_WAIT_S = 2 # Increased wait slightly
REQUEST_DELAY_S = 0.25 # Slightly increased delay
TRADES_COUNT_LIMIT = 1000
# Standard Deribit Expiry Time (can be adjusted if needed)
DERIBIT_EXPIRY_HOUR_UTC = 8

# --- Helper Functions ---

def ensure_dir(dir_path: pathlib.Path):
    """Ensure the directory exists."""
    dir_path.mkdir(parents=True, exist_ok=True)

class DeribitApiError(Exception):
    """Custom exception for Deribit API errors."""
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code

def make_api_request_with_retry(url, params, retries=API_RETRY_ATTEMPTS, wait=API_RETRY_WAIT_S):
    """Makes API request with retry logic."""
    for attempt in range(retries):
        try:
            # logger.debug(f"Attempt {attempt+1}: Requesting {url} with params: {params}")
            response = requests.get(url, params=params, timeout=30) # Increased timeout
            data = response.json()

            # Check for specific Deribit error structure
            if isinstance(data, dict) and 'error' in data:
                error_data = data['error']
                error_message = error_data.get('message', 'Unknown error') if isinstance(error_data, dict) else str(error_data)
                error_code = error_data.get('code', 'N/A') if isinstance(error_data, dict) else 'N/A'

                # Handle rate limit specifically
                if 'rate limit' in error_message.lower():
                    wait_time = wait * (2 ** attempt) # Exponential backoff
                    logger.warning(f"Rate limit hit (Attempt {attempt+1}). Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                # Handle other specific errors if needed, e.g., 'not_found'
                # elif error_code == some_code: ...
                else:
                    # Raise for other defined Deribit errors
                    raise DeribitApiError(f"Deribit API Error (Code: {error_code}): {error_message}", code=error_code)

            # Check for general HTTP errors if no Deribit error structure found
            response.raise_for_status()
            return data
        except DeribitApiError as e:
            # Log Deribit API errors and re-raise if needed, or break if fatal
            logger.error(f"Deribit API Error (Attempt {attempt+1}): {e}")
            # Decide if retry makes sense for this specific Deribit error code (e.g., maybe not for invalid params)
            if attempt + 1 >= retries: raise # Re-raise if max retries hit
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            logger.warning(f"Network/Timeout error (Attempt {attempt+1}/{retries}): {e}.")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed (Attempt {attempt+1}/{retries}): {e}.")

        if attempt + 1 < retries:
            wait_time = wait * (2 ** attempt) # Exponential backoff
            logger.info(f"Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)
        else:
            logger.error("Max retries reached. Request failed.")
            if 'e' in locals(): raise e # Re-raise last exception
            else: raise requests.exceptions.RequestException(f"Request failed after {retries} retries.")

    return None # Should not be reached if exception is raised

def get_historical_option_instruments(currency, target_expiry_ts_ms):
    """Fetches expired option instruments, filters for a specific expiry timestamp."""
    endpoint = f"{DERIBIT_HISTORY_URL}/public/get_instruments"
    params = {'currency': currency, 'kind': 'option', 'expired': 'true', 'include_old': 'true'}
    logger.info(f"Fetching expired {currency} option instruments list...")
    try:
        data = make_api_request_with_retry(endpoint, params)
        if not data or 'result' not in data or not isinstance(data['result'], list):
            logger.error("Unexpected response format or empty result from get_instruments.")
            return []

        all_instruments = data['result']
        logger.info(f"Received {len(all_instruments)} total expired instruments. Filtering for expiry ts {target_expiry_ts_ms}...")

        matching_instruments = []
        # --- MODIFIED REGEX ---
        # Allow 1 or 2 digits for the day: \d{1,2} instead of \d{2}
        instrument_pattern = re.compile(f"^{currency}-(\\d{{1,2}}[A-Z]{{3}}\\d{{2}})-\\d+-[CP]$")
        # --- END MODIFICATION ---

        instruments_checked_count = 0
        instruments_matched_expiry = 0
        instruments_matched_pattern = 0
        instruments_failed_pattern = []

        for inst in all_instruments:
            instruments_checked_count += 1
            # Check if expiration_timestamp exists and matches
            if inst.get('expiration_timestamp') == target_expiry_ts_ms:
                instruments_matched_expiry += 1
                instrument_name = inst.get('instrument_name', '')
                # Basic sanity check on instrument name format using the updated pattern
                if instrument_pattern.match(instrument_name):
                    instruments_matched_pattern += 1
                    matching_instruments.append(instrument_name)
                else:
                    # Log only once per unique failed pattern if it becomes noisy
                    if instrument_name not in instruments_failed_pattern:
                         logger.warning(f"Instrument {instrument_name} matched expiry but has unexpected format.")
                         instruments_failed_pattern.append(instrument_name) # Track logged patterns
                    # Optionally limit the number of warnings logged
                    # if len(instruments_failed_pattern) > 10: break # Example limit

        # Log summary counts
        logger.debug(f"Checked: {instruments_checked_count}, Matched Expiry: {instruments_matched_expiry}, Matched Pattern: {instruments_matched_pattern}")
        if len(instruments_failed_pattern) > 0 and instruments_matched_pattern == 0 :
             logger.error(f"Regex pattern failed for ALL instruments matching expiry {target_expiry_ts_ms}. Check pattern and instrument names. First few failures: {instruments_failed_pattern[:5]}")
        elif len(instruments_failed_pattern) > 0 :
             logger.warning(f"{len(instruments_failed_pattern)} instruments matched expiry but failed format check.")


        logger.info(f"Found {len(matching_instruments)} instruments matching expiry timestamp {target_expiry_ts_ms} AND format pattern.")
        return matching_instruments
    except Exception as e:
        logger.error(f"Error fetching or processing instruments: {e}", exc_info=True)
        return []

def get_historical_trades(instrument_name, start_ts_ms, end_ts_ms):
    """Fetches raw trades for an instrument within a time range (timestamps in ms)."""
    endpoint = f"{DERIBIT_HISTORY_URL}/public/get_last_trades_by_instrument_and_time"
    all_trades = []
    current_start_ts_ms = start_ts_ms
    request_count = 0

    logger.debug(f"Fetching trades for {instrument_name} from {datetime.fromtimestamp(start_ts_ms/1000, tz=timezone.utc)} to {datetime.fromtimestamp(end_ts_ms/1000, tz=timezone.utc)}")

    while current_start_ts_ms < end_ts_ms:
        request_count += 1
        params = {
            'instrument_name': instrument_name,
            'start_timestamp': current_start_ts_ms,
            'end_timestamp': end_ts_ms,
            'count': TRADES_COUNT_LIMIT,
            'include_old': 'true'
        }
        try:
            # Apply delay *before* request to manage rate limits proactively
            if request_count > 1: # No delay needed for the very first request
                time.sleep(REQUEST_DELAY_S)

            data = make_api_request_with_retry(endpoint, params)
            if not data or 'result' not in data or 'trades' not in data['result']:
                logger.warning(f"Unexpected data format or empty result for {instrument_name} at ts {current_start_ts_ms}: {data}")
                break # Stop fetching for this instrument if data is bad

            trades = data['result']['trades']
            has_more = data['result'].get('has_more', False)

            if not trades:
                logger.debug(f"No more trades found for {instrument_name} after ts {current_start_ts_ms}.")
                break # No trades in this range

            all_trades.extend(trades)
            last_trade_ts_ms = trades[-1]['timestamp']
            # logger.debug(f"Fetched {len(trades)} trades for {instrument_name}. Last ts: {last_trade_ts_ms}")

            if not has_more:
                logger.debug(f"'has_more' is false for {instrument_name}. Fetching complete.")
                break # API indicates no more data

            # Prepare for next request - crucial to avoid infinite loops
            next_start_ts_ms = last_trade_ts_ms + 1 # Start *after* the last trade
            if next_start_ts_ms <= current_start_ts_ms:
                logger.warning(f"Timestamp did not advance for {instrument_name} (last_ts={last_trade_ts_ms}, current_start={current_start_ts_ms}). Check for duplicate timestamps or API issues. Breaking.")
                break # Prevent potential infinite loop

            current_start_ts_ms = next_start_ts_ms

        except Exception as e:
            logger.error(f"Failed to fetch trades batch for {instrument_name} starting at {current_start_ts_ms}: {e}")
            # Optionally: implement logic to save partial results here if needed
            return pd.DataFrame() # Return empty on error for this instrument

    if all_trades:
        df = pd.DataFrame(all_trades)
        # Filter again just in case API returned slightly outside range
        df = df[(df['timestamp'] >= start_ts_ms) & (df['timestamp'] <= end_ts_ms)]
        # Deduplicate based on trade_id, keep the first occurrence if any overlap (unlikely with correct pagination)
        df = df.drop_duplicates(subset=['trade_id'], keep='first').sort_values(by='timestamp')
        logger.debug(f"Found {len(df)} unique trades for {instrument_name} within time range.")
        return df

    logger.debug(f"No trades found for {instrument_name} in the specified time range.")
    return pd.DataFrame()

# --- Main Function (Callable by Orchestrator) ---

def run_fetch(currency: str, start_dt: datetime, end_dt: datetime, expiry_dt: datetime, output_dir: pathlib.Path) -> Union[pathlib.Path, None]:
    """
    Fetches Deribit option trades for a given currency, date range, and expiry.

    Args:
        currency: Asset currency (e.g., "BTC", "ETH").
        start_dt: Start datetime (UTC, inclusive) for fetching trades.
        end_dt: End datetime (UTC, exclusive) for fetching trades.
        expiry_dt: The exact expiry date (UTC) of the options market.
        output_dir: The directory to save the raw output CSV file.

    Returns:
        Path to the output CSV file if successful, None otherwise.
    """
    try:
        logger.info(f"Running Deribit Fetcher for {currency}, Expiry: {expiry_dt.strftime('%Y-%m-%d')}, Range: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")

        # Ensure output directory exists
        ensure_dir(output_dir)

        # Ensure datetimes are timezone-aware (assume UTC if needed)
        if start_dt.tzinfo is None: start_dt = pytz.utc.localize(start_dt)
        if end_dt.tzinfo is None: end_dt = pytz.utc.localize(end_dt)
        if expiry_dt.tzinfo is None: expiry_dt = pytz.utc.localize(expiry_dt)

        # Convert to UTC if needed
        start_dt = start_dt.astimezone(pytz.utc)
        end_dt = end_dt.astimezone(pytz.utc)
        expiry_dt = expiry_dt.astimezone(pytz.utc)

        # Calculate timestamps in milliseconds
        start_ts_ms = int(start_dt.timestamp() * 1000)
        end_ts_ms = int(end_dt.timestamp() * 1000)
        # Target expiry timestamp (set time to standard Deribit expiry time)
        target_expiry_dt = expiry_dt.replace(hour=DERIBIT_EXPIRY_HOUR_UTC, minute=0, second=0, microsecond=0)
        target_expiry_ts_ms = int(target_expiry_dt.timestamp() * 1000)
        market_id = target_expiry_dt.strftime("%d%b%y").upper() # e.g., 14MAR25
        # Handle potential single digit day in market_id for filename consistency if needed
        # Although %d usually produces two digits, strftime behaviour can vary.
        # Example: ensure two digits for day in market_id if required downstream
        # market_id = f"{target_expiry_dt.day:02d}{target_expiry_dt.strftime('%b%y').upper()}"


        logger.info(f"Target expiry: {target_expiry_dt} (Timestamp ms: {target_expiry_ts_ms})")
        logger.info(f"Fetch time range ms: {start_ts_ms} -> {end_ts_ms}")


        # Fetch instruments for the target expiry
        instruments = get_historical_option_instruments(currency, target_expiry_ts_ms)
        if not instruments:
            # This log now occurs *after* attempting to filter with the regex
            logger.warning(f"No instruments found matching format and expiry for {currency} {market_id}. Creating empty output.")
            # Define standard empty output columns
            empty_cols = ["trade_seq","trade_id","timestamp","ticker","order_id","price","instrument_name","side","paid_change","liquidation","fee","fee_usd","amount","label","settlement_type","block_trade_id","index_price","underlying_price","mark_price","timestamp_ms"]
            output_filename = f"deribit_trades_{market_id.lower()}.csv" # Use market_id derived from expiry_dt
            output_path = output_dir / output_filename
            pd.DataFrame(columns=empty_cols).to_csv(output_path, index=False)
            logger.info(f"Created empty output file: {output_path}")
            return output_path # Return path even if empty, indicates process ran

        # Fetch raw trades for each instrument
        all_trades_list = []
        logger.info(f"Fetching trades for {len(instruments)} instruments...")
        # Use tqdm for progress bar if desired (might clutter logs less than per-instrument logging)
        # from tqdm.auto import tqdm
        # for instrument in tqdm(instruments, desc="Fetching Instrument Trades"):
        for i, instrument in enumerate(instruments, 1):
             # logger.info(f"[{i}/{len(instruments)}] Fetching trades for {instrument}") # Reduce verbosity
             trades_df = get_historical_trades(instrument, start_ts_ms, end_ts_ms)
             if not trades_df.empty:
                 all_trades_list.append(trades_df)
             # Optional: Short delay even if last request didn't hit rate limit
             # time.sleep(REQUEST_DELAY_S / 5)


        # Save raw trades
        output_filename = f"deribit_trades_{market_id.lower()}.csv" # Use market_id derived from expiry_dt
        output_path = output_dir / output_filename

        if all_trades_list:
            final_df = pd.concat(all_trades_list, ignore_index=True)
            # Final sort and deduplication across instruments (unlikely needed but safe)
            final_df = final_df.drop_duplicates(subset=['trade_id'], keep='first')
            final_df = final_df.sort_values(by=['timestamp', 'instrument_name']).reset_index(drop=True)

            # Add timestamp_ms column for consistency if it doesn't exist from API
            if 'timestamp_ms' not in final_df.columns:
                 # Assume 'timestamp' column holds ms timestamp from API
                 if not final_df.empty and (final_df['timestamp'] > 1e12).all(): # Simple check if values look like ms
                     final_df['timestamp_ms'] = final_df['timestamp']
                 elif not final_df.empty: # If it looks like seconds, convert
                      logger.warning("Deribit API 'timestamp' column doesn't look like milliseconds. Converting assuming seconds.")
                      final_df['timestamp_ms'] = (final_df['timestamp'] * 1000).astype(int)
                 else: # If dataframe is empty after concat (shouldn't happen if list not empty)
                      final_df['timestamp_ms'] = pd.Series(dtype='int64')


            final_df.to_csv(output_path, index=False, float_format="%.8f")
            logger.info(f"Saved {len(final_df)} raw trades ({len(all_trades_list)} instruments had trades) to {output_path}")
            return output_path
        else:
            logger.warning("No trades collected for any matching instrument in the time range.")
            # Create empty file with headers (use headers from a sample non-empty response if possible)
            empty_cols = ["trade_seq","trade_id","timestamp","ticker","order_id","price","instrument_name","side","paid_change","liquidation","fee","fee_usd","amount","label","settlement_type","block_trade_id","index_price","underlying_price","mark_price","timestamp_ms"]
            pd.DataFrame(columns=empty_cols).to_csv(output_path, index=False)
            logger.info(f"Created empty output file: {output_path}")
            return output_path

    except Exception as e:
        logger.error(f"An error occurred during the fetch process: {e}", exc_info=True)
        return None # Indicate failure

# --- Standalone Execution Logic ---
if __name__ == "__main__":
    # Setup for standalone execution (e.g., basic logging, argument parsing)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    # --- Define PROJECT_ROOT for standalone ---
    try:
         # Assumes script is in src/, so parent.parent is thesis2.0/
         PROJECT_ROOT_STANDALONE = pathlib.Path(__file__).resolve().parent.parent
    except NameError:
         # Fallback if __file__ is not defined (e.g., interactive)
         PROJECT_ROOT_STANDALONE = pathlib.Path.cwd()
         logger.info(f"Could not determine project root reliably in standalone mode, using cwd: {PROJECT_ROOT_STANDALONE}")

    # Define default output dir based on project root (assuming thesis2.0 structure)
    DERIBIT_RAW_DIR_DEFAULT = PROJECT_ROOT_STANDALONE / 'data' / 'deribit_data' / 'raw' # Define default using standalone root


    parser = argparse.ArgumentParser(description="Fetch Deribit historical option trades for a specific expiry.")
    parser.add_argument("start_date", help="Start date for fetching trades (YYYY-MM-DD)")
    parser.add_argument("end_date", help="End date for fetching trades (YYYY-MM-DD), exclusive.")
    parser.add_argument("expiry_date", help="Expiry date of the options (YYYY-MM-DD). Time defaults to 08:00 UTC.")
    parser.add_argument("-c", "--currency", default=DERIBIT_CURRENCY_DEFAULT, help=f"Currency (default: {DERIBIT_CURRENCY_DEFAULT})")
    parser.add_argument("-o", "--output-dir", default=str(DERIBIT_RAW_DIR_DEFAULT), help=f"Output directory for raw data (default: {DERIBIT_RAW_DIR_DEFAULT})")
    args = parser.parse_args()

    try:
        start_dt = pytz.utc.localize(datetime.strptime(args.start_date, "%Y-%m-%d"))
        end_dt = pytz.utc.localize(datetime.strptime(args.end_date, "%Y-%m-%d"))
        expiry_dt = pytz.utc.localize(datetime.strptime(args.expiry_date, "%Y-%m-%d"))
        output_dir = pathlib.Path(args.output_dir)

        # Call the main fetching function
        result_path = run_fetch(
            currency=args.currency,
            start_dt=start_dt,
            end_dt=end_dt,
            expiry_dt=expiry_dt,
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