import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns
import logging
import pathlib
import sys
import argparse
from datetime import datetime
import pytz
import shutil  # For file operations

# --- Setup Logger ---
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def ensure_dir(dir_path: pathlib.Path):
    """Ensure the directory exists."""
    dir_path.mkdir(parents=True, exist_ok=True)

# --- Plotting Functions ---

def plot_probability_timeseries(df, output_path, deribit_expiry_dt, polymarket_expiry_dt):
    """
    Plots time series of probabilities for each strike range from both Polymarket and Deribit RND.
    Deribit data stops at deribit_expiry_dt, Polymarket data stops at polymarket_expiry_dt.
    Includes a vertical red line at Deribit options expiry.
    """
    # Melt the dataframe to long format for seaborn plotting
    df_melted = pd.melt(df, id_vars=['timestamp', 'polymarket_strike_range'],
                        value_vars=['polymarket_prob_pct', 'deribit_rnd_prob_pct'],
                        var_name='source', value_name='probability')

    # Drop rows where probability is NaN to prevent plotting issues
    df_melted = df_melted.dropna(subset=['probability'])

    # Determine the full timestamp range for the x-axis
    min_timestamp = df['timestamp'].min()
    max_timestamp = df['timestamp'].max()
    logger.info(f"Full timestamp range: {min_timestamp} to {max_timestamp}")

    def custom_plot(data, deribit_expiry_dt, polymarket_expiry_dt, **kwargs):
        ax = plt.gca()
        # Plot Deribit (stops at Deribit expiry)
        deribit_data = data[(data['source'] == 'deribit_rnd_prob_pct') & (data['timestamp'] <= deribit_expiry_dt)]
        if not deribit_data.empty:
            sns.lineplot(x='timestamp', y='probability', data=deribit_data, label='Deribit RND', ax=ax)
            logger.info(f"Deribit data for {data['polymarket_strike_range'].iloc[0]}: {deribit_data['timestamp'].min()} to {deribit_data['timestamp'].max()}")

        # Plot Polymarket (solid line until its own expiry)
        poly_data = data[(data['source'] == 'polymarket_prob_pct') & (data['timestamp'] <= polymarket_expiry_dt)]
        if not poly_data.empty:
            sns.lineplot(x='timestamp', y='probability', data=poly_data, label='Polymarket', linestyle='-', ax=ax)
            logger.info(f"Polymarket data for {data['polymarket_strike_range'].iloc[0]}: {poly_data['timestamp'].min()} to {poly_data['timestamp'].max()}")
        else:
            logger.warning(f"No Polymarket data for {data['polymarket_strike_range'].iloc[0]}")

        # Add vertical expiry line for Deribit
        ax.axvline(deribit_expiry_dt, color='red', linestyle='--', label='Deribit Expiry')

        # Set x-axis limits to the full range of timestamps
        ax.set_xlim(min_timestamp, max_timestamp)

    g = sns.FacetGrid(df_melted, col='polymarket_strike_range', col_wrap=3, height=4, aspect=2, sharey=False)
    g.map_dataframe(custom_plot, deribit_expiry_dt=deribit_expiry_dt, polymarket_expiry_dt=polymarket_expiry_dt)
    g.add_legend()
    g.set_titles('{col_name}')
    g.set_axis_labels('Time', 'Probability (%)')
    for ax in g.axes.flat:
        ax.xaxis.set_major_formatter(DateFormatter('%d/%m'))
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    g.savefig(output_path)
    plt.close()

def plot_probability_with_spot(df, output_path, deribit_expiry_dt, polymarket_expiry_dt):
    """
    Plots time series of probabilities with spot price on a secondary axis.
    Deribit data stops at deribit_expiry_dt, Polymarket data stops at polymarket_expiry_dt.
    Includes a vertical red line at Deribit options expiry.
    """
    # Melt the dataframe for probabilities
    df_melted = pd.melt(df, id_vars=['timestamp', 'polymarket_strike_range'],
                        value_vars=['polymarket_prob_pct', 'deribit_rnd_prob_pct'],
                        var_name='source', value_name='probability')

    # Drop rows where probability is NaN to prevent plotting issues
    df_melted = df_melted.dropna(subset=['probability'])

    # Determine the full timestamp range for the x-axis
    min_timestamp = df['timestamp'].min()
    max_timestamp = df['timestamp'].max()
    logger.info(f"Full timestamp range (with spot): {min_timestamp} to {max_timestamp}")

    def custom_plot(data, deribit_expiry_dt, polymarket_expiry_dt, **kwargs):
        ax1 = plt.gca()
        # Plot Deribit (stops at Deribit expiry)
        deribit_data = data[(data['source'] == 'deribit_rnd_prob_pct') & (data['timestamp'] <= deribit_expiry_dt)]
        if not deribit_data.empty:
            sns.lineplot(x='timestamp', y='probability', data=deribit_data, label='Deribit RND', ax=ax1)
            logger.info(f"Deribit data (with spot) for {data['polymarket_strike_range'].iloc[0]}: {deribit_data['timestamp'].min()} to {deribit_data['timestamp'].max()}")

        # Plot Polymarket (solid line until its own expiry)
        poly_data = data[(data['source'] == 'polymarket_prob_pct') & (data['timestamp'] <= polymarket_expiry_dt)]
        if not poly_data.empty:
            sns.lineplot(x='timestamp', y='probability', data=poly_data, label='Polymarket', linestyle='-', ax=ax1)
            logger.info(f"Polymarket data (with spot) for {data['polymarket_strike_range'].iloc[0]}: {poly_data['timestamp'].min()} to {poly_data['timestamp'].max()}")
        else:
            logger.warning(f"No Polymarket data (with spot) for {data['polymarket_strike_range'].iloc[0]}")

        # Create a secondary axis for spot price
        ax2 = ax1.twinx()
        # Plot spot price (available for all timestamps)
        spot_data = df[df['polymarket_strike_range'] == data['polymarket_strike_range'].iloc[0]]
        if not spot_data.empty:
            ax2.plot(spot_data['timestamp'], spot_data['spot_price'], color='green', label='Spot Price', alpha=0.5)
            ax2.set_ylabel('Spot Price (USD)', color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            logger.info(f"Spot price data for {data['polymarket_strike_range'].iloc[0]}: {spot_data['timestamp'].min()} to {spot_data['timestamp'].max()}")

        # Add vertical expiry line for Deribit
        ax1.axvline(deribit_expiry_dt, color='red', linestyle='--', label='Deribit Expiry')

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

        # Set x-axis limits to the full range of timestamps
        ax1.set_xlim(min_timestamp, max_timestamp)

    g = sns.FacetGrid(df_melted, col='polymarket_strike_range', col_wrap=3, height=4, aspect=2, sharey=False)
    g.map_dataframe(custom_plot, deribit_expiry_dt=deribit_expiry_dt, polymarket_expiry_dt=polymarket_expiry_dt)
    g.set_titles('{col_name}')
    g.set_axis_labels('Time', 'Probability (%)')
    for ax in g.axes.flat:
        ax.xaxis.set_major_formatter(DateFormatter('%d/%m'))
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    g.savefig(output_path)
    plt.close()

def plot_difference_timeseries(df, output_path, deribit_expiry_dt):
    """
    Plots time series of differences between Polymarket and Deribit RND probabilities for each strike range.
    Includes a vertical red line at Deribit expiry.
    """
    # Determine the full timestamp range for the x-axis
    min_timestamp = df['timestamp'].min()
    max_timestamp = df['timestamp'].max()

    g = sns.FacetGrid(df, col='polymarket_strike_range', col_wrap=3, height=4, aspect=2, sharey=False)
    g.map(sns.lineplot, 'timestamp', 'difference_pct')
    g.set_titles('{col_name}')
    g.set_axis_labels('Time', 'Difference (%)')
    for ax in g.axes.flat:
        ax.axvline(deribit_expiry_dt, color='red', linestyle='--', label='Deribit Expiry')
        ax.set_xlim(min_timestamp, max_timestamp)
        ax.xaxis.set_major_formatter(DateFormatter('%d/%m'))
        ax.tick_params(axis='x', rotation=45)
    g.add_legend()
    plt.tight_layout()
    g.savefig(output_path)
    plt.close()

def plot_difference_distribution(df, output_path):
    """
    Plots the distribution of probability differences.
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(df['difference_pct'].dropna(), kde=True)
    plt.title('Distribution of Probability Differences')
    plt.xlabel('Difference (%)')
    plt.ylabel('Frequency')
    plt.savefig(output_path)
    plt.close()

def plot_difference_vs_moneyness(df, output_path):
    """
    Plots differences vs. moneyness (using min_strike as a proxy).
    """
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='min_strike', y='difference_pct', alpha=0.5)
    plt.title('Difference vs. Min Strike (Proxy for Moneyness)')
    plt.xlabel('Min Strike')
    plt.ylabel('Difference (%)')
    plt.savefig(output_path)
    plt.close()

def plot_convergence(df, output_path):
    """
    Plots average absolute difference vs. time to expiration.
    """
    if 'time_to_expiration_days' not in df.columns or not pd.api.types.is_numeric_dtype(df['time_to_expiration_days']):
        logger.warning("Column 'time_to_expiration_days' not found or not numeric. Skipping convergence plot.")
        return
    if df['time_to_expiration_days'].isnull().all():
        logger.warning("Column 'time_to_expiration_days' contains only NaNs. Skipping convergence plot.")
        return

    df_grouped = df.groupby('time_to_expiration_days')['difference_pct'].apply(lambda x: np.mean(np.abs(x.dropna()))).reset_index()

    if df_grouped.empty:
        logger.warning("No data available after grouping for convergence plot. Skipping.")
        return

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_grouped, x='time_to_expiration_days', y='difference_pct')
    plt.title('Average Absolute Difference vs. Time to Expiration')
    plt.xlabel('Time to Expiration (days)')
    plt.ylabel('Average Absolute Difference (%)')
    if not df_grouped['time_to_expiration_days'].empty:
        plt.gca().invert_xaxis()  # Time to expiration decreases
    plt.savefig(output_path)
    plt.close()

# --- Summary Statistics Function ---

def calculate_summary_stats(df):
    """
    Calculates summary statistics for the differences between Polymarket and Deribit RND probabilities,
    including DTW distances and correlation with spot price changes.
    """
    stats = []
    diff_col = df['difference_pct'].dropna()  # Use dropna() for calculations

    if diff_col.empty:
        stats.append("No valid difference data available for summary statistics.")
        return "\n".join(stats)

    stats.append(f"Overall Mean Difference: {diff_col.mean():.2f}%")
    stats.append(f"Overall Median Difference: {diff_col.median():.2f}%")
    stats.append(f"Overall Std Dev of Difference: {diff_col.std():.2f}%")
    stats.append(f"Overall Count: {len(diff_col)}")

    # By strike range
    if 'polymarket_strike_range' in df.columns:
        strike_range_stats = df.dropna(subset=['difference_pct']).groupby('polymarket_strike_range')['difference_pct'].agg(['mean', 'median', 'std', 'count'])
        if not strike_range_stats.empty:
            stats.append("\nBy Strike Range:")
            for range_str, row in strike_range_stats.iterrows():
                stats.append(f"  {range_str}: Mean={row['mean']:.2f}%, Median={row['median']:.2f}%, Std={row['std']:.2f}%, Count={row['count']}")
        else:
            stats.append("\nNo data available for strike range statistics.")
    else:
        stats.append("\n'polymarket_strike_range' column not found for strike range statistics.")

    # By weeks to expiration
    if 'time_to_expiration_days' in df.columns and pd.api.types.is_numeric_dtype(df['time_to_expiration_days']):
        df_temp = df.dropna(subset=['difference_pct', 'time_to_expiration_days']).copy()
        if not df_temp.empty:
            df_temp['weeks_to_expiration'] = (df_temp['time_to_expiration_days'] / 7).astype(int)
            time_stats = df_temp.groupby('weeks_to_expiration')['difference_pct'].agg(['mean', 'median', 'std', 'count'])
            if not time_stats.empty:
                stats.append("\nBy Weeks to Expiration:")
                time_stats = time_stats.sort_index()
                for weeks, row in time_stats.iterrows():
                    stats.append(f"  {weeks} weeks: Mean={row['mean']:.2f}%, Median={row['median']:.2f}%, Std={row['std']:.2f}%, Count={row['count']}")
            else:
                stats.append("\nNo data available for weeks to expiration statistics.")
        else:
            stats.append("\nNo valid data for weeks to expiration statistics after dropping NaNs.")
    else:
        stats.append("\n'time_to_expiration_days' column not found or not numeric for weeks statistics.")

    # DTW Distance Calculation with Enhanced Checks
    try:
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean

        stats.append("\nDTW Distances by Strike Range (Polymarket vs Deribit):")
        for range_str, group in df.groupby('polymarket_strike_range'):
            group = group.dropna(subset=['polymarket_prob_pct', 'deribit_rnd_prob_pct'])
            if len(group) <= 1:
                stats.append(f"  {range_str}: Insufficient data for DTW calculation (n={len(group)}).")
                continue

            poly_values = group['polymarket_prob_pct'].values.ravel()
            deribit_values = group['deribit_rnd_prob_pct'].values.ravel()

            # Ensure arrays are 1-D and numeric
            if not isinstance(poly_values, np.ndarray) or poly_values.ndim != 1:
                logger.error(f"  {range_str}: poly_values is not a 1-D NumPy array: type={type(poly_values)}, shape={getattr(poly_values, 'shape', 'N/A')}")
                stats.append(f"  {range_str}: Error - Polymarket data is not a 1-D array.")
                continue
            if not isinstance(deribit_values, np.ndarray) or deribit_values.ndim != 1:
                logger.error(f"  {range_str}: deribit_values is not a 1-D NumPy array: type={type(deribit_values)}, shape={getattr(deribit_values, 'shape', 'N/A')}")
                stats.append(f"  {range_str}: Error - Deribit data is not a 1-D array.")
                continue

            # Check for numeric data
            if not np.issubdtype(poly_values.dtype, np.number):
                logger.error(f"  {range_str}: poly_values contains non-numeric data: dtype={poly_values.dtype}")
                stats.append(f"  {range_str}: Error - Polymarket data is non-numeric.")
                continue
            if not np.issubdtype(deribit_values.dtype, np.number):
                logger.error(f"  {range_str}: deribit_values contains non-numeric data: dtype={deribit_values.dtype}")
                stats.append(f"  {range_str}: Error - Deribit data is non-numeric.")
                continue

            # Check for finite values
            if not np.all(np.isfinite(poly_values)):
                logger.error(f"  {range_str}: poly_values contains non-finite values: {poly_values}")
                stats.append(f"  {range_str}: Error - Polymarket data has non-finite values.")
                continue
            if not np.all(np.isfinite(deribit_values)):
                logger.error(f"  {range_str}: deribit_values contains non-finite values: {deribit_values}")
                stats.append(f"  {range_str}: Error - Deribit data has non-finite values.")
                continue

            # Ensure matching lengths
            if len(poly_values) != len(deribit_values):
                logger.error(f"  {range_str}: Array length mismatch: poly_values={len(poly_values)}, deribit_values={len(deribit_values)}")
                stats.append(f"  {range_str}: Error - Array lengths do not match.")
                continue

            # Perform DTW calculation
            try:
                distance, _ = fastdtw(poly_values, deribit_values, dist=euclidean)
                stats.append(f"  {range_str}: DTW Distance = {distance:.2f}")
            except Exception as dtw_e:
                logger.error(f"  {range_str}: DTW calculation failed: {dtw_e}")
                stats.append(f"  {range_str}: Error during DTW calculation.")

    except ImportError:
        stats.append("\nDTW calculation skipped: 'fastdtw' library not installed.")

    # Correlation with Spot Price Changes
    stats.append("\nCorrelation with Spot Price Changes by Strike Range:")
    for range_str, group in df.groupby('polymarket_strike_range'):
        group = group.sort_values('timestamp')
        # Calculate percentage changes
        spot_changes = group['spot_price'].pct_change(fill_method=None).dropna()
        poly_changes = group['polymarket_prob_pct'].pct_change(fill_method=None).dropna()
        deribit_changes = group['deribit_rnd_prob_pct'].pct_change(fill_method=None).dropna()

        # Align indices for correlation calculation
        aligned_data = pd.DataFrame({
            'spot_changes': spot_changes,
            'poly_changes': poly_changes,
            'deribit_changes': deribit_changes
        }).dropna()

        if len(aligned_data) > 1:
            poly_corr = aligned_data['spot_changes'].corr(aligned_data['poly_changes'])
            deribit_corr = aligned_data['spot_changes'].corr(aligned_data['deribit_changes'])
            stats.append(f"  {range_str}:")
            stats.append(f"    Polymarket Probability Change: {poly_corr:.4f}")
            stats.append(f"    Deribit Probability Change: {deribit_corr:.4f}")
        else:
            stats.append(f"  {range_str}: Insufficient data for correlation calculation.")

    return "\n".join(stats)

# --- Main Analysis and Reporting Function ---

def run_analyze_report(combined_file: pathlib.Path, output_dir: pathlib.Path, market_id: str, deribit_expiry_dt: datetime, polymarket_expiry_dt: datetime) -> bool:
    """
    Analyzes the combined data, generates plots and summary statistics,
    and copies the combined data CSV to the output directory.

    Args:
        combined_file: Path to the combined CSV file.
        output_dir: Directory to save the plots, summary statistics, and combined CSV.
        market_id: Market identifier (e.g., '21MAR25').
        deribit_expiry_dt: Datetime of Deribit options expiry (e.g., 08:00 UTC).
        polymarket_expiry_dt: Datetime of Polymarket expiry (e.g., 18:00 UTC).

    Returns:
        True if successful, False otherwise.
    """
    logger.info(f"--- Starting Analysis and Reporting for Market: {market_id} ---")
    logger.info(f"Combined Input: {combined_file.name}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Deribit Expiry Datetime: {deribit_expiry_dt}")
    logger.info(f"Polymarket Expiry Datetime: {polymarket_expiry_dt}")

    try:
        # Ensure output directory exists
        ensure_dir(output_dir)

        if not combined_file.is_file():
            logger.error(f"Combined input file not found: {combined_file}")
            return False

        df = pd.read_csv(combined_file, parse_dates=['timestamp'])
        if df.empty:
            logger.warning("Combined file is empty. Skipping analysis, but copying file.")
            try:
                destination_csv_path = output_dir / combined_file.name
                shutil.copy2(combined_file, destination_csv_path)
                logger.info(f"Copied empty combined data CSV to: {destination_csv_path}")
                with open(output_dir / f'summary_stats_{market_id}.txt', 'w') as f:
                    f.write("Input combined data file was empty. No analysis performed.")
                return True
            except Exception as copy_e:
                logger.error(f"Failed to copy empty combined file: {copy_e}", exc_info=True)
                return False

        # Ensure timestamp and expiry dates are timezone-aware UTC
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        if deribit_expiry_dt.tzinfo is None:
            deribit_expiry_dt = pytz.utc.localize(deribit_expiry_dt)
        deribit_expiry_dt = deribit_expiry_dt.astimezone(pytz.utc)
        if polymarket_expiry_dt.tzinfo is None:
            polymarket_expiry_dt = pytz.utc.localize(polymarket_expiry_dt)
        polymarket_expiry_dt = polymarket_expiry_dt.astimezone(pytz.utc)

        # Ensure numeric columns for probability data
        df['polymarket_prob_pct'] = pd.to_numeric(df['polymarket_prob_pct'], errors='coerce')
        df['deribit_rnd_prob_pct'] = pd.to_numeric(df['deribit_rnd_prob_pct'], errors='coerce')
        df['spot_price'] = pd.to_numeric(df['spot_price'], errors='coerce')

        # Calculate difference
        df['difference_pct'] = df['polymarket_prob_pct'] - df['deribit_rnd_prob_pct']

        # Generate plots
        logger.info("Generating plots...")
        plot_probability_timeseries(df, output_dir / f'plot1_prob_timeseries_{market_id}.png', deribit_expiry_dt, polymarket_expiry_dt)
        plot_probability_with_spot(df, output_dir / f'plot1_prob_timeseries_with_spot_{market_id}.png', deribit_expiry_dt, polymarket_expiry_dt)
        plot_difference_timeseries(df, output_dir / f'plot2_diff_timeseries_{market_id}.png', deribit_expiry_dt)
        plot_difference_distribution(df, output_dir / f'plot3_diff_distribution_{market_id}.png')
        plot_difference_vs_moneyness(df, output_dir / f'plot4_diff_vs_moneyness_{market_id}.png')
        plot_convergence(df, output_dir / f'plot5_convergence_{market_id}.png')

        # Calculate and save summary statistics
        logger.info("Calculating summary statistics...")
        stats = calculate_summary_stats(df)
        stats_file_path = output_dir / f'summary_stats_{market_id}.txt'
        with open(stats_file_path, 'w') as f:
            f.write(stats)
        logger.info(f"Saved summary statistics to: {stats_file_path}")

        # Copy Combined CSV File
        try:
            destination_csv_path = output_dir / combined_file.name
            shutil.copy2(combined_file, destination_csv_path)
            logger.info(f"Copied combined data CSV to: {destination_csv_path}")
        except Exception as copy_e:
            logger.error(f"Failed to copy combined data CSV: {copy_e}", exc_info=True)

        logger.info("Analysis and reporting completed successfully.")
        return True

    except Exception as e:
        logger.error(f"Error during analysis and reporting: {e}", exc_info=True)
        return False

# --- Standalone Execution Logic ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

    # Define PROJECT_ROOT for standalone execution default path
    try:
        PROJECT_ROOT_STANDALONE = pathlib.Path(__file__).resolve().parent.parent
    except NameError:
        PROJECT_ROOT_STANDALONE = pathlib.Path.cwd()
        logger.info(f"Could not determine project root reliably, using cwd: {PROJECT_ROOT_STANDALONE}")

    # Default output directory for standalone
    OUTPUT_DIR_DEFAULT = PROJECT_ROOT_STANDALONE / 'data' / 'reports'

    parser = argparse.ArgumentParser(description="Analyze and report on combined data.")
    parser.add_argument("combined_file", help="Path to the combined CSV file.")
    parser.add_argument("market_id", help="Market ID (e.g., 21MAR25).")
    parser.add_argument("deribit_expiry_date", help="Deribit expiry date and time (YYYY-MM-DD HH:MM UTC).")
    parser.add_argument("polymarket_expiry_date", help="Polymarket expiry date and time (YYYY-MM-DD HH:MM UTC).")
    parser.add_argument("-o", "--output-dir", default=str(OUTPUT_DIR_DEFAULT), help=f"Output directory (default: {OUTPUT_DIR_DEFAULT})")

    args = parser.parse_args()

    # Parse expiry dates to datetime
    deribit_expiry_dt = pytz.utc.localize(datetime.strptime(args.deribit_expiry_date, "%Y-%m-%d %H:%M"))
    polymarket_expiry_dt = pytz.utc.localize(datetime.strptime(args.polymarket_expiry_date, "%Y-%m-%d %H:%M"))

    # Construct the specific output directory for this market_id
    output_dir_market = pathlib.Path(args.output_dir) / args.market_id
    ensure_dir(output_dir_market)  # Ensure this specific directory exists

    success = run_analyze_report(
        combined_file=pathlib.Path(args.combined_file),
        output_dir=output_dir_market,
        market_id=args.market_id,
        deribit_expiry_dt=deribit_expiry_dt,
        polymarket_expiry_dt=polymarket_expiry_dt
    )

    if success:
        sys.exit(0)
    else:
        sys.exit(1)