import logging
import pathlib
import sys
import re
from datetime import datetime, timedelta
import pytz
import importlib
import traceback
import pandas as pd
import time

# --- Setup Logger ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-7s] %(message)s',
    datefmt='%H:%M:%S'
)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# --- Project Structure Definition ---
try:
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = pathlib.Path.cwd()
    logger.info(f"Could not resolve __file__, using cwd as PROJECT_ROOT: {PROJECT_ROOT}")
    if not (PROJECT_ROOT / 'src').exists() or not (PROJECT_ROOT / 'data').exists():
        logger.warning("Project structure (src/, data/) not found relative to cwd. Paths might be incorrect.")
        logger.warning("Please ensure you run this script from the 'thesis2.0' directory.")

SRC_DIR = PROJECT_ROOT / 'src'
DATA_DIR = PROJECT_ROOT / 'data'
POLY_DIR = DATA_DIR / 'polymarket_data'
DERIBIT_RAW_DIR = DATA_DIR / 'deribit_data' / 'raw'
DERIBIT_CLEAN_DIR = DATA_DIR / 'deribit_data' / 'clean'
BINANCE_DIR = DATA_DIR / 'binance_data'
RND_DIR = DATA_DIR / 'rnd_data'
COMBINED_DIR = DATA_DIR / 'combined'
REPORTS_DIR = DATA_DIR / 'reports'

if not SRC_DIR.is_dir():
    logger.error(f"Source directory not found at expected location: {SRC_DIR}")
    logger.error("Please ensure main.py is in the 'thesis2.0' directory containing 'src/' and 'data/'.")
    sys.exit(1)
sys.path.insert(0, str(SRC_DIR))

# --- Configuration Dictionary ---
PIPELINE_CONFIG = {
    "deribit_currency": "BTC",
    "binance_symbol": "BTCUSDT",
    "binance_interval": "1h",
    "rnd_config": {
        'strike_threshold': 8,
        'volume_threshold': 0.1,
        'aggregation_method': 'median',
        'smoothing_sigma': 1.0,
        'risk_free_rate': 0.0,
        'grid_points': 250,
        'integral_tolerance': 0.05,
        'forward_window_pct': 0.15,
        'max_workers': None
    },
}

# --- Main Orchestration Logic ---

def run_pipeline_step(module_name, function_name, step_description, **kwargs):
    """Imports a module, runs a function, logs progress, and handles errors."""
    logger.info(f"--- Starting: {step_description} ---")
    start_time = time.time()
    try:
        if module_name in sys.modules:
            module = importlib.reload(sys.modules[module_name])
        else:
            module = importlib.import_module(module_name)

        if not hasattr(module, function_name):
            raise AttributeError(f"Module '{module_name}' does not have function '{function_name}'")

        run_function = getattr(module, function_name)
        result = run_function(**kwargs)

        if function_name == 'run_analyze_report' and result is False:
            raise Exception(f"Function '{function_name}' returned False, indicating failure.")
        elif function_name != 'run_analyze_report' and result is None:
            raise Exception(f"Function '{function_name}' returned None, indicating failure.")

        duration = time.time() - start_time
        logger.info(f"--- Finished: {step_description} (Duration: {duration:.2f}s) ---")
        return result
    except Exception as e:
        logger.error(f"--- FAILED: {step_description} ---")
        logger.error(f"Error: {e}")
        logger.debug(traceback.format_exc())
        if function_name == 'run_analyze_report':
            return False
        else:
            return None

def parse_poly_filename(filepath: pathlib.Path):
    """
    Parses date range and inferred expiry from Polymarket filename.
    Sets Deribit expiry to 08:00 UTC and Polymarket expiry to 18:00 UTC on the expiry date.

    Args:
        filepath: Path object for the Polymarket file.

    Returns:
        Tuple: (start_dt, fetch_end_dt, deribit_expiry_dt, polymarket_expiry_dt, market_id) or (None, None, None, None, None) on failure.
               Dates are timezone-aware (UTC). fetch_end_dt is exclusive.
    """
    filename = filepath.name
    match = re.search(r'(\d{2}-\d{2}-\d{4})-(\d{2}-\d{2}-\d{4})', filename)
    if not match:
        logger.error(f"Could not parse DD-MM-YYYY date range from filename: {filename}")
        logger.error("Expected format like 'polymarket-price-data-DD-MM-YYYY-DD-MM-YYYY-....csv'")
        return None, None, None, None, None

    start_dt_str = match.group(1)
    end_dt_str = match.group(2)

    try:
        start_dt = pytz.utc.localize(pd.to_datetime(start_dt_str, format='%d-%m-%Y', dayfirst=True))
        expiry_dt_date_only = pd.to_datetime(end_dt_str, format='%d-%m-%Y', dayfirst=True)

        # Set Deribit expiry to 08:00 UTC
        deribit_expiry_dt = pytz.utc.localize(expiry_dt_date_only.replace(hour=8, minute=0, second=0, microsecond=0))
        # Set Polymarket expiry to 18:00 UTC
        polymarket_expiry_dt = pytz.utc.localize(expiry_dt_date_only.replace(hour=18, minute=0, second=0, microsecond=0))

        # Fetch end date is the day *after* the Polymarket expiry date (exclusive start of day)
        fetch_end_dt = pytz.utc.localize(expiry_dt_date_only.replace(hour=0)) + pd.Timedelta(days=1)

        market_id = deribit_expiry_dt.strftime('%d%b%y').upper()

        if start_dt >= deribit_expiry_dt:
            logger.error(f"Parsed start date ({start_dt_str}) is not before Deribit expiry date ({end_dt_str} 08:00 UTC). Check filename.")
            return None, None, None, None, None

        if deribit_expiry_dt >= polymarket_expiry_dt:
            logger.error(f"Deribit expiry ({deribit_expiry_dt}) must be before Polymarket expiry ({polymarket_expiry_dt}).")
            return None, None, None, None, None

        return start_dt, fetch_end_dt, deribit_expiry_dt, polymarket_expiry_dt, market_id
    except ValueError as e:
        logger.error(f"Error parsing dates from filename parts ('{start_dt_str}', '{end_dt_str}'): {e}")
        return None, None, None, None, None

def select_polymarket_file():
    """Lists Polymarket files and asks user for selection."""
    if not POLY_DIR.is_dir():
        logger.error(f"Polymarket data directory not found: {POLY_DIR}")
        return None
    files = sorted(f for f in POLY_DIR.glob('polymarket-price-data-*-*-*-*-*.csv') if f.is_file())
    if not files:
        logger.error(f"No Polymarket CSV files found matching pattern 'polymarket-price-data-DD-MM-YYYY-DD-MM-YYYY-*.csv' in {POLY_DIR}")
        return None

    print("\nAvailable Polymarket data files:")
    for idx, fname in enumerate(files, start=1):
        print(f"  {idx}) {fname.name}")

    selected_file = None
    while selected_file is None:
        try:
            choice = input(f"Select file number to analyze (1-{len(files)}): ").strip()
            if choice.isdigit():
                selection_int = int(choice)
                if 1 <= selection_int <= len(files):
                    selected_file = files[selection_int - 1]
                else:
                    print(f"Invalid selection. Please enter a number between 1 and {len(files)}.")
            else:
                print("Invalid input. Please enter a number.")
        except (EOFError, KeyboardInterrupt):
            logger.warning("User cancelled selection.")
            return None
    return selected_file

# --- Main Pipeline Function ---
def run_full_pipeline():
    """Runs the complete analysis pipeline."""
    logger.info("=============================================")
    logger.info("=== Polymarket vs Deribit RND Analysis ===")
    logger.info("=============================================")

    # 1. Select Polymarket File & Derive Parameters
    logger.info(f"Searching for Polymarket files in: {POLY_DIR}")
    selected_poly_file = select_polymarket_file()
    if not selected_poly_file:
        sys.exit(1)

    logger.info(f"Selected Polymarket file: {selected_poly_file.name}")
    logger.info(f"Attempting to parse parameters from filename...")
    start_dt, fetch_end_dt, deribit_expiry_dt, polymarket_expiry_dt, market_id = parse_poly_filename(selected_poly_file)
    if not all([start_dt, fetch_end_dt, deribit_expiry_dt, polymarket_expiry_dt, market_id]):
        logger.error("Could not derive necessary date parameters from filename.")
        logger.error("Please ensure filename follows pattern '...DD-MM-YYYY-DD-MM-YYYY...'. Exiting.")
        sys.exit(1)

    logger.info(f"--- Processing Market: {market_id} ---")
    logger.info(f"  Time Range: {start_dt.strftime('%Y-%m-%d')} to {polymarket_expiry_dt.strftime('%Y-%m-%d')}")
    logger.info(f"  Deribit Expiry Datetime (UTC): {deribit_expiry_dt}")
    logger.info(f"  Polymarket Expiry Datetime (UTC): {polymarket_expiry_dt}")

    # --- Define Expected File Paths ---
    raw_deribit_file_expected = DERIBIT_RAW_DIR / f"deribit_trades_{market_id.lower()}.csv"
    binance_file_expected = BINANCE_DIR / f"binance_spot_{PIPELINE_CONFIG['binance_symbol']}_{PIPELINE_CONFIG['binance_interval']}_{start_dt.strftime('%Y%m%d')}_{(fetch_end_dt - timedelta(days=1)).strftime('%Y%m%d')}.csv"
    cleaned_deribit_file_expected = DERIBIT_CLEAN_DIR / f"clean_deribit_trades_{market_id.lower()}.csv"
    rnd_file_expected = RND_DIR / f"rnd_results_{market_id}.csv"
    comparison_file_expected = COMBINED_DIR / f"comparison_{market_id}.csv"
    report_output_dir = REPORTS_DIR / market_id

    # --- Execute Pipeline ---
    pipeline_status = True
    actual_paths = {}

    # Step 1: Fetch Deribit Options
    fetch1_result_path = run_pipeline_step(
        '01_deribit_options_fetcher', 'run_fetch', 'Fetching Deribit Options',
        currency=PIPELINE_CONFIG['deribit_currency'],
        start_dt=start_dt,
        end_dt=fetch_end_dt,
        expiry_dt=deribit_expiry_dt,  # Use Deribit expiry for fetching
        output_dir=DERIBIT_RAW_DIR
    )
    if fetch1_result_path is None: pipeline_status = False
    else: actual_paths['raw_deribit'] = fetch1_result_path

    # Step 2: Fetch Binance Spot
    if pipeline_status:
        fetch2_result_path = run_pipeline_step(
            '02_binance_spot_fetcher', 'run_fetch', 'Fetching Binance Spot Prices',
            symbol=PIPELINE_CONFIG['binance_symbol'],
            interval=PIPELINE_CONFIG['binance_interval'],
            start_dt=start_dt,
            end_dt=fetch_end_dt,
            output_dir=BINANCE_DIR
        )
        if fetch2_result_path is None: pipeline_status = False
        else: actual_paths['binance'] = fetch2_result_path

    # Step 3: Clean Deribit Options
    if pipeline_status:
        clean_result_path = run_pipeline_step(
            '03_options_cleaner', 'run_clean', 'Cleaning Deribit Options',
            input_raw_file=actual_paths.get('raw_deribit', raw_deribit_file_expected),
            output_clean_dir=DERIBIT_CLEAN_DIR,
            expiry_dt=deribit_expiry_dt  # Use Deribit expiry for cleaning
        )
        if clean_result_path is None: pipeline_status = False
        else: actual_paths['cleaned_deribit'] = clean_result_path

    # Step 4: Calculate RND
    if pipeline_status:
        rnd_result_path = run_pipeline_step(
            '04_rnd_calculator', 'run_calculate', 'Calculating Risk-Neutral Densities',
            input_clean_file=actual_paths.get('cleaned_deribit', cleaned_deribit_file_expected),
            output_rnd_dir=RND_DIR,
            expiry_dt=deribit_expiry_dt,  # Use Deribit expiry for RND calculation
            config=PIPELINE_CONFIG['rnd_config']
        )
        if rnd_result_path is None: pipeline_status = False
        else: actual_paths['rnd'] = rnd_result_path

    # Step 5: Combine Data Sources
    if pipeline_status:
        combine_result_path = run_pipeline_step(
            '05_combine_rnd_polymarket_spot', 'run_combine', 'Combining Data Sources',
            rnd_file=actual_paths.get('rnd', rnd_file_expected),
            poly_file=selected_poly_file,
            binance_file=actual_paths.get('binance', binance_file_expected),
            deribit_clean_file=actual_paths.get('cleaned_deribit', cleaned_deribit_file_expected),
            output_dir=COMBINED_DIR,
            deribit_expiry_dt=deribit_expiry_dt,
            polymarket_expiry_dt=polymarket_expiry_dt
        )
        if combine_result_path is None: pipeline_status = False
        else: actual_paths['comparison'] = combine_result_path

    # Step 6: Analyze and Generate Report
    if pipeline_status:
        report_output_dir.mkdir(parents=True, exist_ok=True)
        report_result = run_pipeline_step(
            '06_analyze_report',
            'run_analyze_report',
            'Generating Analysis Report',
            combined_file=actual_paths.get('comparison', comparison_file_expected),
            output_dir=report_output_dir,
            market_id=market_id,
            deribit_expiry_dt=deribit_expiry_dt,
            polymarket_expiry_dt=polymarket_expiry_dt
        )
        if report_result is False:
            pipeline_status = False

    # --- Final Summary ---
    logger.info("=============================================")
    if pipeline_status:
        logger.info(f"=== Pipeline Completed Successfully for {market_id} ===")
        logger.info(f"=== Report artifacts saved in: {report_output_dir} ===")
    else:
        logger.error(f"=== Pipeline FAILED for {market_id} ===")
        logger.error("=== Please check logs above for details on the failed step. ===")
    logger.info("=============================================")

if __name__ == "__main__":
    try:
        run_full_pipeline()
    except Exception as e:
        logger.critical(f"An unexpected critical error occurred in the main pipeline: {e}", exc_info=True)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Pipeline execution interrupted by user.")
        sys.exit(1)