import pandas as pd
import numpy as np
from scipy.integrate import trapezoid
import logging
import pathlib
import re
import pytz
import sys
import argparse
from datetime import datetime, timedelta
from typing import Union, Optional, Tuple

# --- Setup Logger ---
logger = logging.getLogger(__name__)

# --- Constants ---
LARGE_FINITE_NUMBER = 1_000_000  # Value to replace np.inf for strike ranges

# --- Helper Functions ---

def ensure_dir(dir_path: pathlib.Path):
    """Ensure the directory exists."""
    dir_path.mkdir(parents=True, exist_ok=True)

def parse_polymarket_header_range(header_str: str) -> Optional[Tuple[float, float]]:
    """
    Parses Polymarket column header string into a strike range tuple (min_k, max_k).
    Handles formats like: ">87k", "85-87k", "<77k". Uses LARGE_FINITE_NUMBER instead of np.inf.
    Multiplies 'k' by 1000.
    """
    if not isinstance(header_str, str): return None
    header_str = header_str.strip().lower().replace('k', '')
    # Match > (e.g., >93) -> (93000, LARGE_FINITE_NUMBER)
    match = re.match(r'>(\d+)', header_str)
    if match: return (float(match.group(1)) * 1000, LARGE_FINITE_NUMBER)
    # Match < (e.g., <77) -> (0, 77000)
    match = re.match(r'<(\d+)', header_str)
    if match: return (0.0, float(match.group(1)) * 1000)  # Assume lower bound 0
    # Match range (e.g., 85-87) -> (85000, 87000)
    match = re.match(r'(\d+)-(\d+)', header_str)
    if match:
        # Handle cases like 87-85k just in case
        k1 = float(match.group(1)) * 1000
        k2 = float(match.group(2)) * 1000
        return (min(k1, k2), max(k1, k2))
    logger.warning(f"Could not parse Polymarket header string: '{header_str}'")
    return None

def integrate_rnd_for_range(rnd_df_hour: pd.DataFrame, min_k: float, max_k: float) -> float:
    """Integrates RND probability within a given strike range [min_k, max_k)."""
    if rnd_df_hour.empty or len(rnd_df_hour) < 2: return 0.0  # Cannot integrate without >= 2 points

    # Ensure data is sorted by strike for trapezoid integration
    rnd_df_hour = rnd_df_hour.sort_values('strike')
    strikes = rnd_df_hour['strike'].values
    rnd = rnd_df_hour['rnd'].values  # Assuming 'rnd' column contains the density values

    # Define the mask for the integration range
    # We integrate *up to* max_k (exclusive), but include min_k (inclusive)
    mask = (strikes >= min_k) & (strikes < max_k)

    # Handle edge cases where the range might be outside the RND grid or only contain one point
    strikes_in_range = strikes[mask]
    rnd_in_range = rnd[mask]

    if len(strikes_in_range) < 2:
        # If only one point is within the strict range, or none, check boundaries
        # Need to interpolate density at the boundaries (min_k, max_k) if they fall within the grid range
        min_grid, max_grid = strikes.min(), strikes.max()
        if max_k <= min_grid or min_k >= max_grid:  # Range is entirely outside grid
            return 0.0

        # Add boundary points if they are within the grid and not already included
        points_k = []
        points_rnd = []

        if min_k > min_grid and min_k < max_grid:
            # Interpolate density at min_k if needed
            interp_rnd_min = np.interp(min_k, strikes, rnd)
            points_k.append(min_k)
            points_rnd.append(interp_rnd_min)

        # Add existing points strictly within the range
        inner_mask = (strikes > min_k) & (strikes < max_k)
        points_k.extend(strikes[inner_mask].tolist())
        points_rnd.extend(rnd[inner_mask].tolist())

        if max_k > min_grid and max_k < max_grid:
            # Interpolate density at max_k if needed
            interp_rnd_max = np.interp(max_k, strikes, rnd)
            points_k.append(max_k)
            points_rnd.append(interp_rnd_max)

        if len(points_k) < 2:
            # Still not enough points, likely range too narrow or outside data
            return 0.0

        # Sort the combined points before integration
        sort_idx = np.argsort(points_k)
        points_k = np.array(points_k)[sort_idx]
        points_rnd = np.array(points_rnd)[sort_idx]
        strikes_in_range = points_k
        rnd_in_range = points_rnd

    # Perform integration using the trapezoidal rule
    try:
        probability = trapezoid(rnd_in_range, strikes_in_range)
        # Clamp probability between 0 and 1 to handle potential numerical inaccuracies
        return max(0.0, min(probability, 1.0))
    except Exception as e:
        logger.error(f"Error during RND integration for range [{min_k}, {max_k}): {e}")
        return np.nan  # Return NaN on error

def get_deribit_volume_proxy(deribit_df_hour: pd.DataFrame, min_k: float, max_k: float) -> float:
    """Calculates a proxy for Deribit volume related to a strike range [min_k, max_k)."""
    if deribit_df_hour.empty: return 0.0
    # Ensure 'strike' and 'volume' columns exist
    if 'strike' not in deribit_df_hour.columns or 'volume' not in deribit_df_hour.columns:
        logger.warning("Missing 'strike' or 'volume' in Deribit data for volume proxy calculation.")
        return 0.0

    # Filter trades within the strike range [min_k, max_k)
    mask = (deribit_df_hour['strike'] >= min_k) & (deribit_df_hour['strike'] < max_k)

    # Sum the volume for the filtered trades
    return deribit_df_hour.loc[mask, 'volume'].sum()

def normalize_probabilities(df: pd.DataFrame, prob_col_raw: str, prob_col_pct: str) -> pd.DataFrame:
    """
    Normalizes probabilities within each timestamp group to sum to 1 (or 100%).

    Args:
        df: DataFrame containing the probabilities.
        prob_col_raw: Name of the column with raw probabilities (0-1 range).
        prob_col_pct: Name of the output column for normalized percentages (0-100 range).

    Returns:
        DataFrame with the added normalized percentage column.
    """
    if prob_col_raw not in df.columns:
        logger.error(f"Raw probability column '{prob_col_raw}' not found for normalization.")
        df[prob_col_pct] = np.nan
        return df

    logger.info(f"Normalizing probabilities in column '{prob_col_raw}'...")
    # Calculate sum per timestamp, handling potential NaNs in individual probabilities
    prob_sum = df.groupby('timestamp')[prob_col_raw].transform('sum')

    # Calculate normalized probability (0-1 range)
    # Avoid division by zero or NaN sums
    normalized_prob = df[prob_col_raw].copy()  # Start with original values
    valid_sum_mask = (prob_sum.notna()) & (prob_sum > 1e-9)  # Mask where sum is valid and non-zero
    normalized_prob.loc[valid_sum_mask] = df.loc[valid_sum_mask, prob_col_raw] / prob_sum.loc[valid_sum_mask]

    # Handle cases where sum was zero or NaN - set normalized prob to NaN or keep original if only one bucket?
    # For simplicity, if sum is invalid, result is NaN. If only one bucket, it's already 1.
    normalized_prob.loc[~valid_sum_mask] = np.nan

    # Convert to percentage and store in the output column
    df[prob_col_pct] = normalized_prob * 100

    # Log summary of normalization
    invalid_sums = len(df[~valid_sum_mask]['timestamp'].unique())
    if invalid_sums > 0:
        logger.warning(f"Could not normalize probabilities for {invalid_sums} timestamps due to invalid sums (zero or NaN).")

    logger.info(f"Finished normalizing '{prob_col_raw}'. Results in '{prob_col_pct}'.")
    return df

def cap_outliers_iqr(series: pd.Series, factor: float = 3.0) -> pd.Series:
    """
    Caps outliers in a pandas Series using the IQR method.

    Args:
        series: The pandas Series to cap.
        factor: The IQR factor to determine outlier bounds (default: 3.0).

    Returns:
        The pandas Series with outliers capped.
    """
    if not pd.api.types.is_numeric_dtype(series):
        logger.warning(f"Series '{series.name}' is not numeric. Skipping outlier capping.")
        return series
    if series.isnull().all():
        logger.warning(f"Series '{series.name}' contains only NaNs. Skipping outlier capping.")
        return series

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr

    original_min = series.min()
    original_max = series.max()

    capped_series = series.clip(lower=lower_bound, upper=upper_bound)

    num_lower_capped = (series < lower_bound).sum()
    num_upper_capped = (series > upper_bound).sum()

    if num_lower_capped > 0 or num_upper_capped > 0:
        logger.info(f"Capped {num_lower_capped} lower and {num_upper_capped} upper outliers in '{series.name}'.")
        logger.debug(f"Original Range: [{original_min}, {original_max}], IQR Bounds: [{lower_bound}, {upper_bound}]")

    return capped_series

# --- Main Combining Function (Callable by Orchestrator) ---

def run_combine(rnd_file: pathlib.Path,
                poly_file: pathlib.Path,
                binance_file: pathlib.Path,
                deribit_clean_file: pathlib.Path,
                output_dir: pathlib.Path,
                deribit_expiry_dt: datetime,
                polymarket_expiry_dt: datetime) -> Union[pathlib.Path, None]:
    """
    Combines RND, Polymarket, Binance Spot, and Deribit volume data.
    Includes Polymarket probability normalization and outlier capping.

    Args:
        rnd_file: Path to the RND results CSV file.
        poly_file: Path to the Polymarket data CSV file.
        binance_file: Path to the Binance spot price CSV file.
        deribit_clean_file: Path to the cleaned Deribit trades CSV file (for volume proxy).
        output_dir: Directory to save the combined output CSV file.
        deribit_expiry_dt: The exact expiry date (UTC) of the Deribit options.
        polymarket_expiry_dt: The exact expiry date (UTC) of the Polymarket prediction markets.

    Returns:
        Path to the combined output CSV file if successful, None otherwise.
    """
    if deribit_expiry_dt.tzinfo is None: deribit_expiry_dt = pytz.utc.localize(deribit_expiry_dt)
    deribit_expiry_dt = deribit_expiry_dt.astimezone(pytz.utc)
    if polymarket_expiry_dt.tzinfo is None: polymarket_expiry_dt = pytz.utc.localize(polymarket_expiry_dt)
    polymarket_expiry_dt = polymarket_expiry_dt.astimezone(pytz.utc)
    market_id = deribit_expiry_dt.strftime("%d%b%y").upper()

    logger.info(f"--- Starting Data Combination for Market: {market_id} ---")
    logger.info(f"RND Input: {rnd_file.name}")
    logger.info(f"Polymarket Input: {poly_file.name}")
    logger.info(f"Binance Spot Input: {binance_file.name}")
    logger.info(f"Deribit Clean Input: {deribit_clean_file.name}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Deribit Expiry Datetime: {deribit_expiry_dt}")
    logger.info(f"Polymarket Expiry Datetime: {polymarket_expiry_dt}")

    # --- Load Data ---
    try:
        logger.info("Loading Binance spot data...")
        binance_cols = ['timestamp', 'close']
        df_binance = pd.read_csv(binance_file, usecols=binance_cols)
        df_binance['timestamp'] = pd.to_datetime(df_binance['timestamp'], format='ISO8601', utc=True, errors='coerce')
        if df_binance['timestamp'].isnull().any():
            logger.warning(f"Coerced {df_binance['timestamp'].isnull().sum()} invalid timestamps in Binance file. Dropping.")
            df_binance = df_binance.dropna(subset=['timestamp'])
        df_binance.rename(columns={'close': 'spot_price'}, inplace=True)

        logger.info("Loading RND data...")
        # Ensure essential RND columns are parsed correctly
        rnd_cols_dtypes = {'strike': float, 'rnd': float}  # Add other cols if needed
        df_rnd = pd.read_csv(rnd_file, parse_dates=['timestamp', 'expiry_date'], dtype=rnd_cols_dtypes)
        # Robust check for timestamp column and timezone localization
        if 'timestamp' in df_rnd.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_rnd['timestamp']):
                logger.warning("RND 'timestamp' column not parsed as datetime. Attempting conversion.")
                df_rnd['timestamp'] = pd.to_datetime(df_rnd['timestamp'], errors='coerce')
                df_rnd.dropna(subset=['timestamp'], inplace=True)  # Drop if conversion failed

            if pd.api.types.is_datetime64_any_dtype(df_rnd['timestamp']) and df_rnd['timestamp'].dt.tz is None and not df_rnd.empty:
                 df_rnd['timestamp'] = df_rnd['timestamp'].dt.tz_localize('UTC')
        else:
             logger.warning("RND file missing 'timestamp' column.")

        logger.info("Loading Polymarket data...")
        df_poly_raw = pd.read_csv(poly_file)
        ts_col = 'Timestamp (UTC)'
        if ts_col not in df_poly_raw.columns: raise ValueError(f"'{ts_col}' not направие found in Polymarket file.")
        df_poly_raw.rename(columns={ts_col: 'timestamp'}, inplace=True)
        df_poly_raw['timestamp'] = pd.to_datetime(df_poly_raw['timestamp'], unit='s', utc=True)
        bucket_cols = [col for col in df_poly_raw.columns if 'k' in col.lower() or '<' in col or '>' in col]
        if not bucket_cols: raise ValueError("Could not identify Polymarket bucket columns.")
        logger.info(f"Identified Polymarket bucket columns: {bucket_cols}")
        cols_to_keep = ['timestamp'] + bucket_cols
        df_poly = df_poly_raw[cols_to_keep].copy()
        # Convert bucket columns to numeric, coercing errors
        for col in bucket_cols:
            df_poly[col] = pd.to_numeric(df_poly[col], errors='coerce')

        logger.info("Loading Deribit cleaned trades data...")
        deribit_cols = ['timestamp', 'strike', 'volume']
        df_deribit_clean = pd.read_csv(deribit_clean_file, usecols=deribit_cols, dtype={'strike':float, 'volume':float})
        df_deribit_clean['timestamp'] = pd.to_datetime(df_deribit_clean['timestamp'], format='ISO8601', utc=True, errors='coerce')
        if df_deribit_clean['timestamp'].isnull().any():
            num_invalid = df_deribit_clean['timestamp'].isnull().sum()
            logger.warning(f"Could not parse {num_invalid} timestamps in Deribit clean file (set to NaT). Removing corresponding rows.")
            df_deribit_clean = df_deribit_clean.dropna(subset=['timestamp'])

    except FileNotFoundError as e:
        logger.error(f"Input file not found: {e}. Cannot combine data.")
        return None
    except ValueError as e:
        logger.error(f"Failed to load or parse input data (ValueError): {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Failed to load or parse input data (Other Exception): {e}", exc_info=True)
        return None

    # --- Prepare Data (Resampling and Indexing) ---
    logger.info("Preparing and aligning data...")
    try:
        # 1. Unpivot Polymarket
        df_poly_long = pd.melt(df_poly, id_vars=['timestamp'], value_vars=bucket_cols,
                               var_name='polymarket_strike_range_str', value_name='polymarket_prob_raw')  # Store raw prob
        df_poly_long['strike_range_tuple'] = df_poly_long['polymarket_strike_range_str'].apply(parse_polymarket_header_range)
        df_poly_long = df_poly_long.dropna(subset=['strike_range_tuple', 'polymarket_prob_raw'])  # Drop if range invalid or prob missing

        # 2. Resample & Index
        df_binance = df_binance.set_index('timestamp')
        # Use ffill() to carry forward last known spot price within the hour if needed
        df_spot_hourly = df_binance['spot_price'].resample('h').ffill().reset_index()

        df_poly_long = df_poly_long.set_index('timestamp')
        # Resample Polymarket - keep the last observation within the hour
        df_poly_hourly = df_poly_long.groupby([pd.Grouper(freq='h'), 'polymarket_strike_range_str', 'strike_range_tuple'])\
                                    ['polymarket_prob_raw'].last().reset_index()

        # Index RND and Deribit data
        if not df_rnd.empty and 'timestamp' in df_rnd.columns and pd.api.types.is_datetime64_any_dtype(df_rnd['timestamp']):
             df_rnd = df_rnd.set_index('timestamp')
        # else: df_rnd is empty or timestamp unusable

        if not df_deribit_clean.empty:
            df_deribit_clean = df_deribit_clean.set_index('timestamp')
        # else: df_deribit_clean is empty

    except Exception as e:
        logger.error(f"Failed during data preparation/resampling: {e}", exc_info=True)
        return None

    # --- Combine Data ---
    logger.info("Integrating RND and combining hourly data...")
    combined_results = []
    # Use Polymarket expiry as the latest timestamp to include all relevant data
    latest_expiry = max(deribit_expiry_dt, polymarket_expiry_dt)
    hourly_timestamps = sorted([ts for ts in df_poly_hourly['timestamp'].unique() if ts <= latest_expiry])
    if not hourly_timestamps:
        logger.warning("No hourly timestamps found after processing Polymarket data.")
        # Handle empty output case
        output_filename = f"comparison_{market_id}.csv"
        output_path = output_dir / output_filename
        ensure_dir(output_path.parent)
        try:
            cols = ['timestamp', 'polymarket_strike_range', 'polymarket_prob_pct',
                    'deribit_rnd_prob_pct', 'time_to_expiration_days', 'spot_price',
                    'deribit_volume_proxy', 'min_strike', 'max_strike']
            pd.DataFrame(columns=cols).to_csv(output_path, index=False)
            logger.info(f"Created empty combined file due to no hourly timestamps: {output_path}")
            return output_path
        except Exception as e_save:
            logger.error(f"Failed to save empty output file: {e_save}")
            return None

    try: from tqdm.auto import tqdm; iterator = tqdm(hourly_timestamps, desc="Processing Hours")
    except ImportError: iterator = hourly_timestamps; logger.info("Processing hours...")

    for hour in iterator:
        try:
            poly_hour_buckets = df_poly_hourly[df_poly_hourly['timestamp'] == hour]
            spot_hour_val = df_spot_hourly.loc[df_spot_hourly['timestamp'] == hour, 'spot_price'].iloc[0] if not df_spot_hourly[df_spot_hourly['timestamp'] == hour].empty else np.nan

            # Get RND data for this hour (only if before Deribit expiry)
            rnd_hour_full = pd.DataFrame()  # Default to empty
            if hour <= deribit_expiry_dt and isinstance(df_rnd.index, pd.DatetimeIndex) and not df_rnd.empty:
                rnd_hour_full = df_rnd.loc[df_rnd.index == hour].copy()  # Ensure it's a copy

            # Get Deribit trade data for this hour (can continue past Deribit expiry if desired)
            deribit_trades_hour = pd.DataFrame()  # Default to empty
            if isinstance(df_deribit_clean.index, pd.DatetimeIndex) and not df_deribit_clean.empty:
                 deribit_trades_hour = df_deribit_clean.loc[(df_deribit_clean.index >= hour) & (df_deribit_clean.index < hour + pd.Timedelta(hours=1))]

            # Skip hour if spot price is missing (critical for context)
            if pd.isna(spot_hour_val):
                 continue

            # Calculate time to expiration based on the relevant expiry
            # For timestamps after Deribit expiry but before Polymarket expiry, we still want Polymarket data
            if hour <= deribit_expiry_dt:
                # Use Deribit expiry for time to expiration when RND data is relevant
                time_to_exp_days = (deribit_expiry_dt - hour).total_seconds() / (60 * 60 * 24)
            else:
                # After Deribit expiry, use Polymarket expiry for time to expiration
                time_to_exp_days = (polymarket_expiry_dt - hour).total_seconds() / (60 * 60 * 24)

            # Process each Polymarket bucket for this hour
            for _, poly_row in poly_hour_buckets.iterrows():
                strike_range_str = poly_row['polymarket_strike_range_str']
                min_k, max_k = poly_row['strike_range_tuple']
                poly_prob_raw = poly_row['polymarket_prob_raw']  # Raw probability (0-1)

                # Calculate RND probability for the bucket (only if before Deribit expiry)
                rnd_prob_raw = np.nan
                if hour <= deribit_expiry_dt and not rnd_hour_full.empty:
                    rnd_prob_raw = integrate_rnd_for_range(rnd_hour_full, min_k, max_k)

                # Calculate Deribit volume proxy for the bucket
                deribit_vol_proxy = get_deribit_volume_proxy(deribit_trades_hour, min_k, max_k)

                combined_results.append({
                    'timestamp': hour,
                    'polymarket_strike_range': strike_range_str,
                    'polymarket_prob_raw': poly_prob_raw if pd.notna(poly_prob_raw) else np.nan,
                    'deribit_rnd_prob_raw': rnd_prob_raw if pd.notna(rnd_prob_raw) else np.nan,  # Store raw RND prob
                    'time_to_expiration_days': time_to_exp_days,
                    'spot_price': spot_hour_val,
                    'deribit_volume_proxy': deribit_vol_proxy,
                    'min_strike': min_k,
                    'max_strike': max_k
                })
        except Exception as e:
            logger.error(f"Error processing hour {hour}: {e}", exc_info=True)
            continue

    # --- Create Final DataFrame ---
    if not combined_results:
         logger.warning("No combined results generated after hourly processing.")
         # Handle empty output case again
         output_filename = f"comparison_{market_id}.csv"
         output_path = output_dir / output_filename
         ensure_dir(output_path.parent)
         try:
              cols = ['timestamp', 'polymarket_strike_range', 'polymarket_prob_pct',
                      'deribit_rnd_prob_pct', 'time_to_expiration_days', 'spot_price',
                      'deribit_volume_proxy', 'min_strike', 'max_strike']
              pd.DataFrame(columns=cols).to_csv(output_path, index=False)
              logger.info(f"Created empty combined file: {output_path}")
              return output_path
         except Exception as e_save:
              logger.error(f"Failed to save empty output file: {e_save}")
              return None

    final_df = pd.DataFrame(combined_results)
    logger.info(f"Generated initial combined DataFrame with {len(final_df)} rows.")

    # --- Apply Data Quality Improvements ---
    # 1. Normalize Polymarket Probabilities
    final_df = normalize_probabilities(final_df, 'polymarket_prob_raw', 'polymarket_prob_pct')

    # 2. Optionally Normalize RND Probabilities (usually not needed, but for consistency if desired)
    # final_df = normalize_probabilities(final_df, 'deribit_rnd_prob_raw', 'deribit_rnd_prob_pct')
    # If not normalizing RND, just convert raw to percentage
    if 'deribit_rnd_prob_raw' in final_df.columns:
         final_df['deribit_rnd_prob_pct'] = final_df['deribit_rnd_prob_raw'] * 100
    else:
         final_df['deribit_rnd_prob_pct'] = np.nan

    # 3. Cap Outliers
    logger.info("Applying outlier capping (IQR method)...")
    if 'spot_price' in final_df.columns:
        final_df['spot_price'] = cap_outliers_iqr(final_df['spot_price'], factor=3.0)
    if 'deribit_volume_proxy' in final_df.columns:
         # Be cautious capping volume - could be genuinely spikey. Factor might need adjustment.
         final_df['deribit_volume_proxy'] = cap_outliers_iqr(final_df['deribit_volume_proxy'], factor=5.0)  # Using a larger factor for volume

    # 4. Final Column Selection and Sorting
    # Define final columns explicitly
    final_cols = ['timestamp', 'polymarket_strike_range', 'polymarket_prob_pct',
                  'deribit_rnd_prob_pct', 'time_to_expiration_days', 'spot_price',
                  'deribit_volume_proxy', 'min_strike', 'max_strike']
    # Select only columns that actually exist in the dataframe
    final_df = final_df[[col for col in final_cols if col in final_df.columns]]

    final_df = final_df.sort_values(by=['timestamp', 'min_strike']).reset_index(drop=True)
    logger.info(f"Final combined DataFrame size after processing: {len(final_df)} rows.")

    # --- Save Final DataFrame ---
    output_filename = f"comparison_{market_id}.csv"
    output_path = output_dir / output_filename
    logger.info(f"Saving final combined data to: {output_path}")
    ensure_dir(output_path.parent)
    try:
        final_df.to_csv(output_path, index=False, float_format='%.5f')
        logger.info("Successfully saved final combined data.")
        return output_path
    except Exception as e:
        logger.error(f"Failed to save final output file: {e}")
        return None

# --- Standalone Execution Logic ---
if __name__ == "__main__":
    # Setup basic logging if run standalone
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%%(levelname)s] %(message)s', datefmt='%H:%M:%S')

    # Define PROJECT_ROOT for standalone execution default path
    try:
         PROJECT_ROOT_STANDALONE = pathlib.Path(__file__).resolve().parent.parent
    except NameError:
         PROJECT_ROOT_STANDALONE = pathlib.Path.cwd()
         logger.info(f"Could not determine project root reliably in standalone mode, using cwd: {PROJECT_ROOT_STANDALONE}")

    # Default paths for standalone based on assumed project structure
    RND_DIR_DEFAULT = PROJECT_ROOT_STANDALONE / 'data' / 'rnd_data'
    POLY_DIR_DEFAULT = PROJECT_ROOT_STANDALONE / 'data' / 'polymarket_data'
    BINANCE_DIR_DEFAULT = PROJECT_ROOT_STANDALONE / 'data' / 'binance_data'
    DERIBIT_CLEAN_DIR_DEFAULT = PROJECT_ROOT_STANDALONE / 'data' / 'deribit_data' / 'clean'
    OUTPUT_DIR_DEFAULT = PROJECT_ROOT_STANDALONE / 'data' / 'combined'

    parser = argparse.ArgumentParser(description="Combine RND, Polymarket, Spot, and Volume data with data quality improvements.")
    parser.add_argument("rnd_file", help="Path to RND results CSV file.")
    parser.add_argument("poly_file", help="Path to Polymarket data CSV file.")
    parser.add_argument("binance_file", help="Path to Binance spot price CSV file.")
    parser.add_argument("deribit_clean_file", help="Path to cleaned Deribit trades CSV file.")
    parser.add_argument("deribit_expiry_date", help="Deribit expiry date and time (YYYY-MM-DD HH:MM UTC).")
    parser.add_argument("polymarket_expiry_date", help="Polymarket expiry date and time (YYYY-MM-DD HH:MM UTC).")
    parser.add_argument("-o", "--output-dir", default=str(OUTPUT_DIR_DEFAULT), help=f"Output directory (default: {OUTPUT_DIR_DEFAULT})")

    args = parser.parse_args()

    try:
        # Parse expiry date strings to datetime objects, ensure UTC
        deribit_expiry_dt = pytz.utc.localize(datetime.strptime(args.deribit_expiry_date, "%Y-%m-%d %H:%M"))
        polymarket_expiry_dt = pytz.utc.localize(datetime.strptime(args.polymarket_expiry_date, "%Y-%m-%d %H:%M"))

        # Call the main combining function using Path objects
        result_path = run_combine(
            rnd_file=pathlib.Path(args.rnd_file),
            poly_file=pathlib.Path(args.poly_file),
            binance_file=pathlib.Path(args.binance_file),
            deribit_clean_file=pathlib.Path(args.deribit_clean_file),
            output_dir=pathlib.Path(args.output_dir),
            deribit_expiry_dt=deribit_expiry_dt,
            polymarket_expiry_dt=polymarket_expiry_dt
        )

        # Check result and exit
        if result_path:
            logger.info(f"Standalone execution successful. Output: {result_path}")
            sys.exit(0)
        else:
            logger.error("Standalone execution failed.")
            sys.exit(1)

    except ValueError as e:
        logger.error(f"Error parsing date or path. Please check formats (e.g., YYYY-MM-DD HH:MM). Details: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"Input file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)