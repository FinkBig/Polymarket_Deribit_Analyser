# --------------------------------------------------------------------------- #
#                            Imports & Setup                                  #
# --------------------------------------------------------------------------- #
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import trapezoid
import logging
import pathlib
import sys
from datetime import datetime
import pytz
import argparse
from concurrent.futures import ProcessPoolExecutor
import warnings
import math
from typing import Union, Optional, Dict, Any, Tuple

# --- Setup Logger ---
logger = logging.getLogger(__name__)

# Suppress RuntimeWarnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --------------------------------------------------------------------------- #
#                            Constants & Defaults                             #
# --------------------------------------------------------------------------- #
# Default configuration values (can be overridden by config dict)
DEFAULT_CONFIG = {
    'strike_threshold': 8,
    'volume_threshold': 0.1,
    'aggregation_method': 'median',  # 'vwap' or 'median'
    'smoothing_sigma': 1.0,
    'risk_free_rate': 0.0,
    'grid_points': 250,
    'integral_tolerance': 0.05,
    'forward_window_pct': 0.15,
    'max_workers': None  # Use default (CPU count)
}

# Column Names (expected from input, generated in output)
COL_TIMESTAMP = 'timestamp'
COL_STRIKE = 'strike'
COL_PRICE = 'option_price'
COL_VOLUME = 'volume'
COL_SPOT = 'spot_price'
COL_TYPE = 'type'

# Output columns
OUT_COL_RND = 'rnd'
OUT_COL_FORWARD = 'forward_price'
OUT_COL_TTE = 'time_to_expiry'
OUT_COL_RFR = 'risk_free_rate'
OUT_COL_EXPIRY = 'expiry_date'
OUT_COL_RND_MEAN = 'rnd_mean'
OUT_COL_QUALITY = 'quality_ok'

# --------------------------------------------------------------------------- #
#                      Arbitrage Checks                                       #
# --------------------------------------------------------------------------- #
def check_vertical_arbitrage(options_df):
    """Checks for basic vertical spread arbitrage violations on aggregated prices."""
    flags = {'call_monotonic': True, 'put_monotonic': True}
    options_df = options_df.sort_values(COL_STRIKE)
    calls = options_df[options_df[COL_TYPE] == 'C']
    puts = options_df[options_df[COL_TYPE] == 'P']

    if len(calls) >= 2:
        call_diffs = np.diff(calls[COL_PRICE].values)
        if np.any(call_diffs > 1e-6):
            flags['call_monotonic'] = False
    if len(puts) >= 2:
        put_diffs = np.diff(puts[COL_PRICE].values)
        if np.any(put_diffs < -1e-6):
            flags['put_monotonic'] = False
    return flags['call_monotonic'] and flags['put_monotonic']

def check_butterfly_arbitrage(options_df):
    """Checks for butterfly spread arbitrage (convexity) on aggregated prices."""
    flags = {'call_convex': True, 'put_convex': True}
    options_df = options_df.sort_values(COL_STRIKE)
    calls = options_df[options_df[COL_TYPE] == 'C']
    puts = options_df[options_df[COL_TYPE] == 'P']

    if len(calls) >= 3:
        call_strikes = calls[COL_STRIKE].values
        call_prices = calls[COL_PRICE].values
        dK = np.diff(call_strikes)
        if np.any(dK <= 1e-9):
            logger.warning("Duplicate or non-increasing call strikes found pre-butterfly check.")
            return False
        if np.all(np.isclose(dK, dK[0])):
            second_diff = np.diff(call_prices, n=2)
            if np.any(second_diff < -1e-6):
                flags['call_convex'] = False
        else:
            if np.any(dK <= 1e-9):
                return False
            dP_dK = np.diff(call_prices) / dK
            dK_mid = 0.5 * (dK[1:] + dK[:-1])
            if np.any(dK_mid <= 1e-9):
                logger.warning("Invalid mid-strike difference for call convexity check.")
                return False
            d2P_dK2_approx = np.diff(dP_dK) / dK_mid
            if np.any(d2P_dK2_approx < -1e-6):
                flags['call_convex'] = False

    if len(puts) >= 3:
        put_strikes = puts[COL_STRIKE].values
        put_prices = puts[COL_PRICE].values
        dK = np.diff(put_strikes)
        if np.any(dK <= 1e-9):
            logger.warning("Duplicate or non-increasing put strikes found pre-butterfly check.")
            return False
        if np.all(np.isclose(dK, dK[0])):
            second_diff = np.diff(put_prices, n=2)
            if np.any(second_diff < -1e-6):
                flags['put_convex'] = False
        else:
            if np.any(dK <= 1e-9):
                return False
            dP_dK = np.diff(put_prices) / dK
            dK_mid = 0.5 * (dK[1:] + dK[:-1])
            if np.any(dK_mid <= 1e-9):
                logger.warning("Invalid mid-strike difference for put convexity check.")
                return False
            d2P_dK2_approx = np.diff(dP_dK) / dK_mid
            if np.any(d2P_dK2_approx < -1e-6):
                flags['put_convex'] = False

    return flags['call_convex'] and flags['put_convex']

# --------------------------------------------------------------------------- #
#                      Forward Price Estimation (Improved)                    #
# --------------------------------------------------------------------------- #
def estimate_forward_price(options_df, S0, r, T, window_pct):
    """ Estimates forward price F using median of PCR implied forwards... """
    if T <= 1e-9:
        return S0
    discount_factor = np.exp(-r * T)
    if discount_factor > 1.1 or discount_factor < 0.5:
        return S0 * np.exp(r * T)
    if options_df is None or options_df.empty:
        return S0 * np.exp(r * T)

    lower_k = S0 * (1 - window_pct)
    upper_k = S0 * (1 + window_pct)
    puts = options_df[(options_df[COL_TYPE] == 'P') & (options_df[COL_STRIKE] >= lower_k) & (options_df[COL_STRIKE] <= upper_k)].set_index(COL_STRIKE)
    calls = options_df[(options_df[COL_TYPE] == 'C') & (options_df[COL_STRIKE] >= lower_k) & (options_df[COL_STRIKE] <= upper_k)].set_index(COL_STRIKE)
    common_strikes = calls.index.intersection(puts.index)

    if common_strikes.empty:
        return S0 * np.exp(r * T)

    implied_forwards = []
    try:
        for k in common_strikes:
            call_price = calls.loc[k, COL_PRICE]
            put_price = puts.loc[k, COL_PRICE]
            if isinstance(call_price, pd.Series):
                call_price = call_price.iloc[0]
            if isinstance(put_price, pd.Series):
                put_price = put_price.iloc[0]
            implied_f = k + (call_price - put_price) / discount_factor
            if S0 * 0.7 < implied_f < S0 * 1.3:
                implied_forwards.append(implied_f)
    except Exception:
        return S0 * np.exp(r * T)

    if not implied_forwards:
        return S0 * np.exp(r * T)

    F_median = np.median(implied_forwards)
    if not (S0 * 0.85 < F_median < S0 * 1.15):
        return S0 * np.exp(r * T)
    return F_median

# --------------------------------------------------------------------------- #
#                      Option Price Aggregation & Initial Checks              #
# --------------------------------------------------------------------------- #
def aggregate_option_prices(hour_df, volume_threshold, agg_method):
    """ Aggregates prices, performs initial arbitrage checks... """
    if COL_TYPE not in hour_df.columns:
        return pd.DataFrame(), False
    df = hour_df[hour_df[COL_VOLUME] >= volume_threshold].copy()
    if df.empty:
        return pd.DataFrame(), True

    def aggregate(group):
        if agg_method == 'vwap':
            vol_sum = group[COL_VOLUME].sum()
            return np.average(group[COL_PRICE], weights=group[COL_VOLUME]) if vol_sum > 1e-9 else group[COL_PRICE].mean()
        elif agg_method == 'median':
            return group[COL_PRICE].median()
        else:
            raise ValueError(f"Unknown aggregation method: {agg_method}")

    agg_prices = df.groupby([COL_STRIKE, COL_TYPE]).apply(
        lambda g: pd.Series({COL_PRICE: aggregate(g), 'total_volume': g[COL_VOLUME].sum()})
    ).reset_index()

    vertical_ok = check_vertical_arbitrage(agg_prices)
    butterfly_ok = check_butterfly_arbitrage(agg_prices)
    arbitrage_ok = vertical_ok and butterfly_ok
    if not arbitrage_ok:
        logger.warning(f"Aggregated price arbitrage check failed: VerticalOK={vertical_ok}, ButterflyOK={butterfly_ok}")
    return agg_prices, arbitrage_ok

# --------------------------------------------------------------------------- #
#                      RND Calculation (Core Logic)                           #
# --------------------------------------------------------------------------- #
def calculate_rnd_core(call_strikes, call_option_prices, F, T, r, grid_points, smoothing_sigma, integral_tolerance):
    """ Calculates RND using Breeden-Litzenberger, checks convexity... """
    if len(call_strikes) < 3:
        return None, None, None, False
    unique_strikes, unique_idx = np.unique(call_strikes, return_index=True)
    if len(unique_strikes) < 3:
        return None, None, None, False
    unique_prices = call_option_prices[unique_idx]
    min_strike, max_strike = unique_strikes.min(), unique_strikes.max()
    if np.isclose(min_strike, max_strike):
        return None, None, None, False

    dense_strikes = np.linspace(min_strike, max_strike, grid_points)
    discount_factor = np.exp(-r * T)

    try:
        interp_func = CubicSpline(unique_strikes, unique_prices, bc_type='natural', extrapolate=False)
        interp_prices = interp_func(dense_strikes)
        second_diff = np.diff(interp_prices, n=2)
        interp_convex = np.all(second_diff >= -1e-6)
        if not interp_convex:
            logger.warning(f"Interpolated curve non-convex (Min 2nd Diff: {second_diff.min():.4f})")
        if np.any(interp_prices < -0.001 * F):
            logger.warning("Interpolated prices have negative values.")

        d2C_dK2 = interp_func.derivative(nu=2)(dense_strikes)
        smoothed_d2C_dK2 = gaussian_filter1d(d2C_dK2, sigma=smoothing_sigma)
        rnd_raw = smoothed_d2C_dK2 / discount_factor
        rnd_non_negative = np.maximum(rnd_raw, 0)

        integral = trapezoid(rnd_non_negative, dense_strikes)
        integral_quality_ok = True
        if integral < 1e-10:
            rnd_normalized = rnd_non_negative
            integral_quality_ok = False
        else:
            rnd_normalized = rnd_non_negative / integral
            final_integral = trapezoid(rnd_normalized, dense_strikes)
            if abs(final_integral - 1.0) > integral_tolerance:
                logger.warning(f"RND integral ({final_integral:.4f}) deviates > {integral_tolerance} from 1.")
                integral_quality_ok = False

        rnd_mean = np.nan if np.all(np.isclose(rnd_normalized, 0)) else trapezoid(dense_strikes * rnd_normalized, dense_strikes)
        return dense_strikes, rnd_normalized, rnd_mean, (integral_quality_ok and interp_convex)
    except Exception as e:
        logger.error(f"Error in RND calculation core: {e}", exc_info=True)
        return None, None, None, False

# --------------------------------------------------------------------------- #
#                      Hourly Processing Function                             #
# --------------------------------------------------------------------------- #
def process_hour(task_data):
    """
    Processes option data for a single hour to calculate RND.
    Now receives config dict within task_data.
    """
    # Unpack task data including config
    hour, group_df, S0, r, T, expiry_date, config = task_data
    # Retrieve parameters from config dict
    volume_threshold = config.get('volume_threshold', DEFAULT_CONFIG['volume_threshold'])
    agg_method = config.get('aggregation_method', DEFAULT_CONFIG['aggregation_method'])
    strike_threshold = config.get('strike_threshold', DEFAULT_CONFIG['strike_threshold'])
    forward_window_pct = config.get('forward_window_pct', DEFAULT_CONFIG['forward_window_pct'])
    grid_points = config.get('grid_points', DEFAULT_CONFIG['grid_points'])
    smoothing_sigma = config.get('smoothing_sigma', DEFAULT_CONFIG['smoothing_sigma'])
    integral_tolerance = config.get('integral_tolerance', DEFAULT_CONFIG['integral_tolerance'])

    try:
        # 1. Aggregate & Check Arbitrage
        agg_prices_all, initial_arbitrage_ok = aggregate_option_prices(
            group_df, volume_threshold, agg_method
        )
        if agg_prices_all.empty:
            return pd.DataFrame()

        calls_agg = agg_prices_all[agg_prices_all[COL_TYPE] == 'C']
        if calls_agg.empty or calls_agg[COL_STRIKE].nunique() < strike_threshold:
            return pd.DataFrame()

        # 2. Estimate Forward Price
        F = estimate_forward_price(agg_prices_all, S0, r, T, forward_window_pct)
        if pd.isna(F) or F <= 0:
            logger.warning(f"Skipping hour {hour}: Invalid estimated Forward price ({F}).")
            return pd.DataFrame()

        # 3. Calculate RND
        call_strikes = calls_agg[COL_STRIKE].values
        call_prices = calls_agg[COL_PRICE].values
        dense_strikes, rnd_normalized, rnd_mean, rnd_quality_ok = calculate_rnd_core(
            call_strikes, call_prices, F, T, r, grid_points, smoothing_sigma, integral_tolerance
        )
        if dense_strikes is None:
            return pd.DataFrame()

        # 4. Final Quality & Formatting
        final_quality_ok = initial_arbitrage_ok and rnd_quality_ok
        if not final_quality_ok:
            logger.warning(f"Hour {hour}: Quality check failed (InitialArbOK={initial_arbitrage_ok}, RNDQualityOK={rnd_quality_ok}).")

        result_df = pd.DataFrame({
            COL_TIMESTAMP: hour,
            COL_STRIKE: dense_strikes,
            OUT_COL_RND: rnd_normalized,
            COL_SPOT: S0,
            OUT_COL_FORWARD: F,
            OUT_COL_TTE: T,
            OUT_COL_RFR: r,
            OUT_COL_EXPIRY: expiry_date,
            OUT_COL_RND_MEAN: rnd_mean,
            OUT_COL_QUALITY: final_quality_ok
        })
        return result_df

    except Exception as e:
        logger.error(f"Error processing hour {hour}: {e}", exc_info=True)
        return pd.DataFrame()

# --- Main Function (Callable by Orchestrator) ---
def run_calculate(input_clean_file: pathlib.Path, output_rnd_dir: pathlib.Path, expiry_dt: datetime, config: dict) -> Union[pathlib.Path, None]:
    """
    Calculates hourly RNDs from cleaned options data for a specific expiry.

    Args:
        input_clean_file: Path to the cleaned Deribit trades CSV file (from script 03).
        output_rnd_dir: Directory to save the RND results CSV file.
        expiry_dt: The exact expiry date (UTC) of the options market.
        config: Dictionary containing processing parameters (e.g., thresholds, sigma).

    Returns:
        Path to the output RND CSV file if successful, None otherwise.
    """
    # Ensure expiry is timezone aware
    if expiry_dt.tzinfo is None:
        expiry_dt = pytz.utc.localize(expiry_dt)
    expiry_dt = expiry_dt.astimezone(pytz.utc)
    expiry_str = expiry_dt.strftime("%d%b%y").upper()  # For output filename

    logger.info(f"--- Starting RND Calculation for Expiry: {expiry_str} ---")
    logger.info(f"Input File: {input_clean_file}")
    logger.info(f"Output Directory: {output_rnd_dir}")
    logger.info(f"Config: {config}")  # Log the config used

    if not input_clean_file.is_file():
        logger.error(f"Input file not found: {input_clean_file}")
        return None

    output_rnd_dir.mkdir(parents=True, exist_ok=True)  # Ensure output dir exists

    try:
        # Load data
        logger.info("Loading cleaned data...")
        df = pd.read_csv(input_clean_file)
        required_load_cols = [COL_TIMESTAMP, COL_STRIKE, COL_PRICE, COL_VOLUME, COL_SPOT, COL_TYPE]
        if not all(col in df.columns for col in required_load_cols):
            logger.error(f"Input file {input_clean_file.name} missing required columns. Need: {required_load_cols}. Found: {df.columns.tolist()}. Exiting.")
            return None

        # Updated line: Use format='ISO8601' for robust timestamp parsing
        df[COL_TIMESTAMP] = pd.to_datetime(df[COL_TIMESTAMP], format='ISO8601', utc=True, errors='coerce')
        
        # Optional: Log and handle NaT values
        if df[COL_TIMESTAMP].isnull().any():
            num_invalid = df[COL_TIMESTAMP].isnull().sum()
            logger.warning(f"{num_invalid} timestamps could not be parsed and were set to NaT.")
            df = df.dropna(subset=[COL_TIMESTAMP])  # Optional: Remove rows with NaT timestamps

        # Filter strictly before expiry
        df = df[df[COL_TIMESTAMP] < expiry_dt]
        if df.empty:
            logger.warning("No data remaining after filtering strictly before expiry.")
            output_filename = f"rnd_results_{expiry_str}.csv"
            output_path = output_rnd_dir / output_filename
            pd.DataFrame(columns=[COL_TIMESTAMP, COL_STRIKE, OUT_COL_RND, COL_SPOT, OUT_COL_FORWARD, OUT_COL_TTE, OUT_COL_RFR, OUT_COL_EXPIRY, OUT_COL_RND_MEAN]).to_csv(output_path, index=False)
            logger.info(f"Created empty RND results file: {output_path}")
            return output_path

        # --- Prepare tasks for parallel processing ---
        logger.info("Grouping data by hour...")
        if COL_TIMESTAMP in df.columns and df.index.name != COL_TIMESTAMP:
            df = df.set_index(COL_TIMESTAMP)
        elif df.index.name != COL_TIMESTAMP:
            logger.error("Timestamp column not found/set as index.")
            return None
        if df.index.tz is None:
            df = df.tz_localize('UTC')

        start_time = df.index.min().floor('h')
        end_time = df.index.max().floor('h')
        end_time = min(end_time, expiry_dt - pd.Timedelta(hours=1))
        if start_time > end_time:
            logger.warning("Start time > end time. No hours to process.")
            return None
        hourly_index = pd.date_range(start=start_time, end=end_time, freq='h', tz='UTC')

        tasks = []
        skipped_nodata, skipped_invalid = 0, 0
        for hour in hourly_index:
            group_df = df[(df.index >= hour) & (df.index < hour + pd.Timedelta(hours=1))]
            if group_df.empty:
                skipped_nodata += 1
                continue
            S0 = group_df[COL_SPOT].median()
            T = (expiry_dt - hour).total_seconds() / (365.25 * 24 * 60 * 60)
            if T > 1e-9 and pd.notna(S0) and S0 > 0:
                tasks.append((hour, group_df.reset_index(), S0, config.get('risk_free_rate', DEFAULT_CONFIG['risk_free_rate']), T, expiry_dt, config))
            else:
                skipped_invalid += 1

        logger.info(f"Prepared {len(tasks)} tasks. Skipped {skipped_nodata} hrs (no data), {skipped_invalid} hrs (invalid T/S0).")
        if not tasks:
            logger.warning("No processable hours found.")
            output_filename = f"rnd_results_{expiry_str}.csv"
            output_path = output_rnd_dir / output_filename
            pd.DataFrame(columns=[COL_TIMESTAMP, COL_STRIKE, OUT_COL_RND, COL_SPOT, OUT_COL_FORWARD, OUT_COL_TTE, OUT_COL_RFR, OUT_COL_EXPIRY, OUT_COL_RND_MEAN]).to_csv(output_path, index=False)
            logger.info(f"Created empty RND results file: {output_path}")
            return output_path

        # --- Execute tasks ---
        results = []
        max_workers = config.get('max_workers', DEFAULT_CONFIG['max_workers'])
        logger.info(f"Starting parallel processing with {max_workers or 'default'} workers...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            try:
                from tqdm.auto import tqdm
                results = list(tqdm(executor.map(process_hour, tasks), total=len(tasks), desc="Calculating RND"))
            except ImportError:
                logger.info("tqdm not installed, progress bar disabled for RND calculation.")
                results = list(executor.map(process_hour, tasks))

        logger.info("Parallel processing finished.")

        # --- Combine and save results ---
        final_df = pd.concat([r for r in results if r is not None and not r.empty], ignore_index=True)
        output_filename = f"rnd_results_{expiry_str}.csv"
        output_path = output_rnd_dir / output_filename

        if not final_df.empty:
            num_hours = len(final_df[COL_TIMESTAMP].unique())
            num_quality_issues = 0
            if OUT_COL_QUALITY in final_df.columns:
                final_df[OUT_COL_QUALITY] = final_df[OUT_COL_QUALITY].astype(bool)
                num_quality_issues = (~final_df.groupby(COL_TIMESTAMP)[OUT_COL_QUALITY].first()).sum()
                final_df.drop(columns=[OUT_COL_QUALITY], inplace=True, errors='ignore')

            logger.info(f"Processing complete. {num_hours} hours generated RNDs.")
            if num_quality_issues > 0:
                logger.warning(f"Quality issues flagged in {num_quality_issues} hours.")

            final_df.to_csv(output_path, index=False, float_format='%.8f')
            logger.info(f"Saving results for {expiry_str} to {output_path} ({len(final_df)} rows)")
            return output_path
        else:
            logger.warning(f"No RND results generated for expiry {expiry_str}.")
            pd.DataFrame(columns=[COL_TIMESTAMP, COL_STRIKE, OUT_COL_RND, COL_SPOT, OUT_COL_FORWARD, OUT_COL_TTE, OUT_COL_RFR, OUT_COL_EXPIRY, OUT_COL_RND_MEAN]).to_csv(output_path, index=False)
            logger.info(f"Created empty RND results file: {output_path}")
            return output_path

    except FileNotFoundError:
        logger.error(f"Input file not found: {input_clean_file}")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"Input file is empty: {input_clean_file}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during RND calculation: {e}", exc_info=True)
        return None

# --- Standalone Execution Logic ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

    # Define PROJECT_ROOT for standalone
    try:
        PROJECT_ROOT_STANDALONE = pathlib.Path(__file__).resolve().parent.parent
    except NameError:
        PROJECT_ROOT_STANDALONE = pathlib.Path.cwd()
        logger.info(f"Could not determine project root reliably, using cwd: {PROJECT_ROOT_STANDALONE}")

    # Define default I/O dirs based on project root
    DEFAULT_INPUT_DIR_STANDALONE = PROJECT_ROOT_STANDALONE / 'data' / 'deribit_data' / 'clean'
    DEFAULT_OUTPUT_DIR_STANDALONE = PROJECT_ROOT_STANDALONE / 'data' / 'rnd_data'

    parser = argparse.ArgumentParser(description="Calculate RND from cleaned options data.")
    parser.add_argument("input_file", help="Path to the cleaned Deribit trades CSV file.")
    parser.add_argument("expiry_date", help="Expiry date (YYYY-MM-DD). Time defaults to 08:00 UTC.")
    parser.add_argument("-o", "--output-dir", default=str(DEFAULT_OUTPUT_DIR_STANDALONE), help="Output directory.")
    parser.add_argument('--strike-threshold', type=int)
    parser.add_argument('--volume-threshold', type=float)
    parser.add_argument('--aggregation-method', type=str, choices=['vwap', 'median'])
    parser.add_argument('--smoothing-sigma', type=float)
    parser.add_argument('--risk-free-rate', type=float)
    parser.add_argument('--grid-points', type=int)
    parser.add_argument('--max-workers', type=int)
    parser.add_argument('--forward-window-pct', type=float)
    parser.add_argument('--integral-tolerance', type=float)

    args = parser.parse_args()

    try:
        input_file = pathlib.Path(args.input_file)
        output_dir = pathlib.Path(args.output_dir)
        expiry_dt = pytz.utc.localize(datetime.strptime(args.expiry_date, "%Y-%m-%d"))
        expiry_dt = expiry_dt.replace(hour=8, minute=0, second=0, microsecond=0)

        # Build config dictionary from args, using defaults if arg is None
        config = DEFAULT_CONFIG.copy()
        for key, value in vars(args).items():
            if key in config and value is not None:
                config[key] = value

        # Call the main calculation function
        result_path = run_calculate(
            input_clean_file=input_file,
            output_rnd_dir=output_dir,
            expiry_dt=expiry_dt,
            config=config
        )

        if result_path:
            logger.info(f"Standalone execution successful. Output: {result_path}")
            sys.exit(0)
        else:
            logger.error("Standalone execution failed.")
            sys.exit(1)

    except ValueError as e:
        logger.error(f"Error parsing date or number. Please check formats. Details: {e}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)