import pandas as pd
import pathlib
import logging
import re
from datetime import datetime
import sys
import argparse # Keep for standalone
import pytz # Added import
from typing import Union, Optional # Import Union and Optional

# --- Setup Logger ---
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s') # Set level in main script

# --- Constants ---
# Expected columns from raw Deribit fetcher (Script 01)
RAW_COLS_EXPECTED = ['timestamp', 'instrument_name', 'price', 'amount', 'index_price', 'iv']
# Output columns for cleaned file (used by scripts 04, 05)
CLEAN_COLS_OUTPUT = ['timestamp', 'timestamp_ms', 'instrument_name', 'strike', 'type',
                     'option_price', 'volume', 'spot_price', 'implied_volatility']

# --- Helper Functions ---
def ensure_dir(dir_path: pathlib.Path):
    """Ensure the directory exists."""
    dir_path.mkdir(parents=True, exist_ok=True)

# --- Main Cleaning Function (Callable by Orchestrator) ---

def run_clean(input_raw_file: pathlib.Path, output_clean_dir: pathlib.Path, expiry_dt: datetime) -> Union[pathlib.Path, None]:
    """
    Cleans raw Deribit options trade data for a specific expiry, processing both Calls and Puts.

    Args:
        input_raw_file: Path to the raw Deribit trades CSV file (from script 01).
        output_clean_dir: Directory to save the cleaned output CSV file.
        expiry_dt: The exact expiry date (UTC, time usually 08:00) of the options market.

    Returns:
        Path to the cleaned output CSV file if successful, None otherwise.
    """
    # Ensure expiry_dt is timezone-aware UTC
    if expiry_dt.tzinfo is None:
        logger.warning("Expiry datetime is timezone naive. Assuming UTC.")
        expiry_dt = pytz.utc.localize(expiry_dt) # Or require tz-aware input
    expiry_dt = expiry_dt.astimezone(pytz.utc) # Ensure UTC
    # Get a string representation mainly for logging and filename
    expiry_date_log_str = expiry_dt.strftime("%d%b%y").upper()
    target_date = expiry_dt.date() # Get just the date part for comparison

    logger.info(f"--- Starting Options Data Cleaning for Expiry: {expiry_date_log_str} ---")
    logger.info(f"Input File: {input_raw_file}")
    logger.info(f"Output Directory: {output_clean_dir}")

    if not input_raw_file.is_file():
        logger.error(f"Input file not found: {input_raw_file}")
        return None

    # Read the Deribit trades CSV
    try:
        df = pd.read_csv(input_raw_file)
        logger.info(f"Read {len(df)} raw rows from {input_raw_file.name}")
        if df.empty:
            logger.warning("Input file is empty.")
            output_filename = f"clean_deribit_trades_{expiry_date_log_str.lower()}.csv"
            output_path = output_clean_dir / output_filename
            ensure_dir(output_path.parent)
            pd.DataFrame(columns=CLEAN_COLS_OUTPUT).to_csv(output_path, index=False)
            logger.info(f"Created empty cleaned file: {output_path}")
            return output_path

        # Check for required input columns (flexible about 'iv')
        required_load_cols = [col for col in RAW_COLS_EXPECTED if col != 'iv']
        missing_cols = [col for col in required_load_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Input file missing required columns: {missing_cols}. Found: {df.columns.tolist()}")
            return None
        current_expected_cols = [col for col in RAW_COLS_EXPECTED if col in df.columns] # Use available cols

    except Exception as e:
        logger.error(f"Failed to read Deribit file {input_raw_file}: {e}", exc_info=True)
        return None

    # --- Cleaning Steps ---
    try:
        # 1. Filter for the specified expiry using robust date parsing
        if 'instrument_name' not in df.columns:
             logger.error("'instrument_name' column missing. Cannot filter by expiry.")
             return None

        # Handle potential NaN in instrument_name before splitting/extracting
        df.dropna(subset=['instrument_name'], inplace=True)
        if df.empty:
            logger.warning("No rows remaining after dropping NaN instrument names.")
            # Create empty file as no data to process
            output_filename = f"clean_deribit_trades_{expiry_date_log_str.lower()}.csv"
            output_path = output_clean_dir / output_filename
            ensure_dir(output_path.parent)
            pd.DataFrame(columns=CLEAN_COLS_OUTPUT).to_csv(output_path, index=False)
            logger.info(f"Created empty cleaned file: {output_path}")
            return output_path

        # --- MODIFICATION: Robust Date Filtering ---
        # Extract the date part (e.g., 7MAR25 or 14MAR25)
        currency = df['instrument_name'].iloc[0].split('-')[0] # Assume consistent currency
        # Regex to capture the date part: 1 or 2 digits, 3 letters, 2 digits
        date_extract_pattern = f'^{currency}-(\\d{{1,2}}[A-Z]{{3}}\\d{{2}})-'
        df['inst_date_str'] = df['instrument_name'].str.extract(date_extract_pattern, expand=False)

        # Parse the extracted date string - '%d' handles single/double digits
        # Coerce errors for any unexpected formats that slip through regex
        df['inst_date'] = pd.to_datetime(df['inst_date_str'], format='%d%b%y', errors='coerce')

        # Filter based on the date part matching the target expiry date
        rows_before_expiry_filter = len(df)
        # Compare just the date component, ignoring time
        df_filtered = df[df['inst_date'].dt.date == target_date].copy()
        rows_after_expiry_filter = len(df_filtered)
        logger.info(f"Filtered {rows_before_expiry_filter} rows to {rows_after_expiry_filter} trades matching expiry date {target_date.strftime('%Y-%m-%d')}")
        # --- END MODIFICATION ---

        if df_filtered.empty:
            logger.warning(f"No trades found matching expiry date {target_date.strftime('%Y-%m-%d')}.")
            # Create empty file
            output_filename = f"clean_deribit_trades_{expiry_date_log_str.lower()}.csv"
            output_path = output_clean_dir / output_filename
            ensure_dir(output_path.parent)
            pd.DataFrame(columns=CLEAN_COLS_OUTPUT).to_csv(output_path, index=False)
            logger.info(f"Created empty cleaned file: {output_path}")
            return output_path

        # 2. *** Process BOTH Calls and Puts *** (No C/P filter here)

        # 3. Select and rename relevant columns
        df_clean = df_filtered[current_expected_cols].copy() # Use columns actually present
        df_clean = df_clean.rename(columns={
            'timestamp': 'timestamp_ms', # Keep original ms timestamp
            'price': 'option_price',
            'amount': 'volume', # Deribit 'amount' usually refers to number of contracts
            'index_price': 'spot_price',
            'iv': 'implied_volatility' # Rename even if missing, will be dropped later if not needed
        })

        # 4. Convert main timestamp to datetime object (UTC)
        # Ensure timestamp_ms is numeric before conversion
        df_clean['timestamp_ms'] = pd.to_numeric(df_clean['timestamp_ms'], errors='coerce')
        df_clean.dropna(subset=['timestamp_ms'], inplace=True) # Drop rows if timestamp couldn't be read as number
        df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp_ms'], unit='ms', utc=True, errors='coerce')
        df_clean.dropna(subset=['timestamp'], inplace=True) # Drop rows if conversion failed

        # 5. Extract strike price
        # Regex extracts digits between the last two hyphens
        strike_extract = df_clean['instrument_name'].str.extract(r'-(\d+)-[CP]$')
        df_clean['strike'] = pd.to_numeric(strike_extract[0], errors='coerce')

        # 6. Derive 'type' column (C or P)
        # Extract last character - safer than assuming length if format varies slightly
        df_clean['type'] = df_clean['instrument_name'].str[-1:].str.upper()

        # 7. Drop rows where essential info extraction failed or type is not C/P
        initial_rows = len(df_clean)
        # Check for NaNs in critical columns created/used downstream
        critical_cols = ['timestamp', 'strike', 'type', 'option_price', 'volume', 'spot_price']
        # Add 'implied_volatility' to critical if it's expected downstream and present
        if 'implied_volatility' in df_clean.columns:
            critical_cols.append('implied_volatility')

        df_clean = df_clean.dropna(subset=critical_cols)
        df_clean = df_clean[df_clean['type'].isin(['C', 'P'])]
        # Ensure strike is integer after dropna (if not already NaN)
        df_clean['strike'] = df_clean['strike'].astype(int)
        rows_dropped = initial_rows - len(df_clean)
        if rows_dropped > 0:
            logger.warning(f"Dropped {rows_dropped} rows due to missing critical data (timestamp, strike, price, vol, spot etc.) or invalid type.")

        if df_clean.empty:
            logger.warning("No valid trades remain after cleaning and checking critical columns.")
            output_filename = f"clean_deribit_trades_{expiry_date_log_str.lower()}.csv"
            output_path = output_clean_dir / output_filename
            ensure_dir(output_path.parent)
            pd.DataFrame(columns=CLEAN_COLS_OUTPUT).to_csv(output_path, index=False)
            logger.info(f"Created empty cleaned file: {output_path}")
            return output_path

        # 8. Reorder and select final output columns
        # Ensure all defined output columns exist before selecting
        final_cols_present = [col for col in CLEAN_COLS_OUTPUT if col in df_clean.columns]
        df_clean = df_clean[final_cols_present]

        # 9. Sort by timestamp
        df_clean = df_clean.sort_values(by='timestamp').reset_index(drop=True)

        # --- Save Cleaned Data ---
        output_filename = f"clean_deribit_trades_{expiry_date_log_str.lower()}.csv"
        output_path = output_clean_dir / output_filename
        ensure_dir(output_path.parent) # Ensure output dir exists

        df_clean.to_csv(output_path, index=False, float_format="%.8f")
        logger.info(f"Saved {len(df_clean)} cleaned trades (Calls & Puts) to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"An error occurred during the cleaning process: {e}", exc_info=True)
        return None

# --- Standalone Execution Logic ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    # Define PROJECT_ROOT for standalone execution default path
    try:
         PROJECT_ROOT_STANDALONE = pathlib.Path(__file__).resolve().parent.parent
    except NameError:
         PROJECT_ROOT_STANDALONE = pathlib.Path.cwd()
         logger.info(f"Could not determine project root reliably, using cwd: {PROJECT_ROOT_STANDALONE}")

    DERIBIT_RAW_DIR_DEFAULT = PROJECT_ROOT_STANDALONE / 'data' / 'deribit_data' / 'raw'
    DERIBIT_CLEAN_DIR_DEFAULT = PROJECT_ROOT_STANDALONE / 'data' / 'deribit_data' / 'clean'

    parser = argparse.ArgumentParser(description="Clean raw Deribit option trades for a specific expiry.")
    parser.add_argument("input_file", help="Path to the raw Deribit trades CSV file.")
    parser.add_argument("expiry_date", help="Expiry date to filter (YYYY-MM-DD). Time defaults to 08:00 UTC.")
    parser.add_argument("-o", "--output-dir", default=str(DERIBIT_CLEAN_DIR_DEFAULT), help=f"Output directory for cleaned data (default: {DERIBIT_CLEAN_DIR_DEFAULT})")

    args = parser.parse_args()

    try:
        input_file = pathlib.Path(args.input_file)
        output_dir = pathlib.Path(args.output_dir)
        # Parse expiry date and assume standard 8:00 UTC expiry time
        # Ensure pytz is used for timezone awareness
        expiry_dt = pytz.utc.localize(datetime.strptime(args.expiry_date, "%Y-%m-%d"))
        expiry_dt = expiry_dt.replace(hour=8, minute=0, second=0, microsecond=0) # Set time

        # Call the main cleaning function
        result_path = run_clean(
            input_raw_file=input_file,
            output_clean_dir=output_dir,
            expiry_dt=expiry_dt
        )

        if result_path:
            logger.info(f"Standalone execution successful. Output: {result_path}")
            sys.exit(0)
        else:
            logger.error("Standalone execution failed.")
            sys.exit(1)

    except ValueError as e:
        logger.error(f"Error parsing expiry date. Please use YYYY-MM-DD format. Details: {e}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)