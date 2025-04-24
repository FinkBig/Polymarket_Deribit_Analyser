# Polymarket vs Deribit RND Analysis (Thesis Project)

This project analyzes and compares implied probability distributions derived from Deribit Bitcoin options (using Risk-Neutral Density - RND) with probabilities from Polymarket prediction markets for corresponding price ranges. The goal is to understand how expectations reflected in these two distinct market types align or differ.

## Project Structure
```
polymarket_deribit_analyser/
├── data/                     # Data files
│   ├── binance_data/         # Raw Binance spot price data
│   ├── combined/             # Combined comparison data (output of script 05)
│   ├── deribit_data/         # Deribit options data
│   │   ├── raw/              # Raw fetched Deribit trades (output of script 01)
│   │   └── clean/            # Cleaned Deribit trades (output of script 03)
│   ├── polymarket_data/      # Raw Polymarket data (INPUT - required)
│   ├── reports/              # Analysis reports and plots (output of script 06)
│   │   └── [market_id]/      # Subfolder for each analyzed market
│   └── rnd_data/             # Calculated RND results (output of script 04)
├── src/                      # Source code Python scripts
│   ├── 01_deribit_options_fetcher.py
│   ├── 02_binance_spot_fetcher.py
│   ├── 03_options_cleaner.py
│   ├── 04_rnd_calculator.py
│   ├── 05_combine_rnd_polymarket_spot.py
│   └── 06_analyze_report.py
├── main.py                   # Main script to run the pipeline
├── requirements.txt          # Python package dependencies
├── README.md                 # This file
└── .gitignore                # Git ignore configuration (recommended)
```
## Pipeline Details

### Data Sources

The analysis relies on three main data sources:

1.  **Polymarket Prediction Market Data:**
    * **Source:** User-provided CSV files downloaded from Polymarket (placed in `data/polymarket_data/`).
    * **Content:** Represents the market's collective belief about the probability of Bitcoin's price falling within specific ranges at a future expiry date and time (typically 18:00 UTC on the expiry day). Each column (e.g., ">87k", "85-87k", "<77k") corresponds to a price bucket, and the values represent the price (probability) of the contract for that bucket.
    * **Filename:** The filename **must** contain the start and end dates of the data period in `DD-MM-YYYY` format (e.g., `polymarket-price-data-07-03-2025-14-03-2025-someid.csv`). These dates are crucial as they define the analysis window and are used to infer the expiry dates for both Polymarket (18:00 UTC on end date) and Deribit (08:00 UTC on end date).

2.  **Deribit Options Market Data:**
    * **Source:** Fetched programmatically from the Deribit Historical Data API (`src/01_deribit_options_fetcher.py`)[cite: 1].
    * **Content:** Raw historical trade data for Bitcoin options contracts expiring at the inferred Deribit expiry time (08:00 UTC). This includes trade timestamp, price, volume (amount), underlying index price, and potentially implied volatility for various strike prices (both Calls and Puts)[cite: 1].

3.  **Binance Spot Market Data:**
    * **Source:** Fetched programmatically from the Binance API (`src/02_binance_spot_fetcher.py`)[cite: 2].
    * **Content:** Historical hourly OHLCV (Open, High, Low, Close, Volume) data for the BTCUSDT trading pair. The 'close' price is used as the hourly spot price reference throughout the analysis[cite: 2].

### Data Processing Workflow

The pipeline processes the data in several stages:

1.  **Fetch Raw Data:** Retrieves Deribit options trades and Binance spot prices for the time window defined by the selected Polymarket file (`src/01`, `src/02`)[cite: 1, 2].
2.  **Clean Deribit Data:** Filters the raw Deribit trades for the specific target expiry date, selects essential columns, standardizes names (e.g., 'price' -> 'option_price'), extracts strike price and option type (Call/Put) from the instrument name (`src/03`).
3.  **Calculate RND:** Computes the hourly Risk-Neutral Density from the cleaned Deribit options data (details below) (`src/04`).
4.  **Combine Data:** Merges the hourly data streams. It parses Polymarket's price bucket headers into strike ranges, integrates the hourly RND curve over these ranges to get comparable probabilities, aligns these with Polymarket probabilities and Binance spot prices, calculates a Deribit volume proxy for each bucket, normalizes Polymarket probabilities per hour, and applies outlier capping (`src/05`).
5.  **Analyze & Report:** Takes the final combined dataset, calculates differences between the two probability sources, and generates plots and summary statistics to visualize and quantify the comparison (`src/06`).

### RND Calculation Methodology

The Risk-Neutral Density (RND) represents the probability distribution of the future price of the underlying asset (Bitcoin) implied by the current prices of options contracts. It assumes a risk-neutral world where investors are indifferent to risk. The pipeline calculates hourly RNDs using the Breeden-Litzenberger (1978) approach implemented in `src/04_rnd_calculator.py`:

1.  **Hourly Aggregation:** Cleaned option trade data is grouped by hour. Within each hour, prices for the same strike and type (Call/Put) are aggregated (using volume-weighted average or median, based on config) after filtering for minimum trade volume.
2.  **Arbitrage Checks:** Basic checks are performed on the aggregated hourly prices to detect potential vertical spread (monotonicity) and butterfly spread (convexity) arbitrage opportunities. Hours failing these checks are flagged.
3.  **Forward Price Estimation:** The implied forward price for the expiry is estimated using Put-Call Parity on options near the current spot price.
4.  **Spline Interpolation:** A cubic spline function is fitted to the aggregated Call option prices versus their strike prices for the hour. This creates a smooth representation of the call price function $C(K)$.
5.  **Second Derivative:** The core of the Breeden-Litzenberger formula relates the RND to the second derivative of the call price function with respect to the strike price ($K$):
    $RND(K) \approx e^{rT} \frac{\partial^2 C}{\partial K^2}$
    where $r$ is the risk-free rate and $T$ is the time to expiration in years. The second derivative is calculated analytically from the fitted cubic spline.
6.  **Smoothing:** Gaussian smoothing is applied to the calculated second derivative to reduce noise.
7.  **Adjustment & Normalization:** The smoothed second derivative is adjusted by the discount factor ($e^{rT}$) and negative values are floored at zero. The resulting density curve is numerically integrated (using the trapezoidal rule), and the entire curve is scaled so that the total probability integrates to 1 (within a defined tolerance).
8.  **Quality Check:** The final RND is assessed based on the initial arbitrage checks, the convexity of the interpolated curve, and whether the final integral is close to 1.

### Results and Interpretation

**The primary final results for each analyzed market can be found within the `data/reports/[market_id]/` directory.** This includes the combined dataset, summary statistics, and visualization plots described below:

* **Combined Data CSV (`comparison_[market_id].csv`):** An hourly dataset containing:
    * `timestamp`: The specific hour (UTC).
    * `polymarket_strike_range`: The price bucket definition from Polymarket (e.g., "85-87k").
    * `polymarket_prob_pct`: The normalized probability (%) from Polymarket for that bucket at that hour.
    * `deribit_rnd_prob_pct`: The probability (%) for the *same* price bucket, derived by integrating the calculated Deribit RND curve over that range.
    * `time_to_expiration_days`: Time remaining until the Deribit expiry.
    * `spot_price`: The hourly BTC spot price from Binance.
    * `deribit_volume_proxy`: Sum of Deribit contract volume traded within the bucket's strike range during that hour.
    * `min_strike`, `max_strike`: The numeric boundaries of the bucket.
* **Plots (`plot*.png`):** Visualizations comparing the Polymarket and Deribit RND probabilities over time, showing their difference, the distribution of this difference, and how the difference evolves as expiry approaches. Plots also show the relationship with the spot price.
* **Summary Statistics (`summary_stats_[market_id].txt`):** Text file containing quantitative measures of the comparison:
    * Overall mean, median, and standard deviation of the difference between Polymarket and RND probabilities.
    * Breakdown of these statistics by strike range and time to expiration.
    * (Optional) Dynamic Time Warping (DTW) distance: A measure of similarity between the time series of the two probability sources for each bucket (requires `fastdtw` library).
    * Correlation between changes in spot price and changes in probabilities from both sources.

**Overall Goal:** The results aim to shed light on how these two different markets price the probability of future Bitcoin price movements. Discrepancies could arise from different participant bases, market microstructures, risk premia, or information efficiency. The analysis explores the magnitude, persistence, and potential drivers (like time to expiry or moneyness) of these differences.

## Setup

1.  **Clone the repository (if applicable).**
2.  **Python Environment:** It's recommended to use a virtual environment (e.g., `venv`). Requires Python 3.8+.
    ```bash
    python -m venv .venv

    # On Mac OS or Linux
    source .venv/bin/activate  
    
    # On Windows
    .venv\Scripts\activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Input Data:** Place your downloaded Polymarket CSV file(s) into the `data/polymarket_data/` directory.
    * **CRITICAL:** The filename **must** follow the pattern `*-DD-MM-YYYY-DD-MM-YYYY-*.csv` (e.g., `polymarket-price-data-07-03-2025-14-03-2025-someid.csv`).
    * The pipeline parses the start date (`DD-MM-YYYY`) and end date (`DD-MM-YYYY`) directly from this filename to define the data fetching period.
    * It infers the Deribit options expiry as 08:00 UTC on the *end date* parsed from the filename.
    * It infers the Polymarket market expiry as 18:00 UTC on the *end date* parsed from the filename.

## Running the Pipeline

1.  **Activate Virtual Environment:**

    ```bash
    source .venv/bin/activate # Or equivalent
    ```
2.  **Run the Main Script:** Execute `main.py` from project root.

    ```bash
    python main.py
    ```
3.  **Select File:** The script will list the available Polymarket files found in `data/polymarket_data/`. Enter the number corresponding to the file you want to analyze.
4.  **Execution:** The pipeline will execute the steps outlined above sequentially, logging progress and errors.
5.  **Output:** Intermediate data files are saved in the relevant `data/` subdirectories (see Project Structure). **The final data results and analysis output** for the selected market (identified by `market_id`, e.g., `14MAR25`), including the combined comparison CSV, summary statistics, and plots, **are located in the `data/reports/[market_id]/` directory.**

## Configuration

Some key parameters are set in `main.py` within the `PIPELINE_CONFIG` dictionary:
* `deribit_currency`: e.g., "BTC"
* `binance_symbol`: e.g., "BTCUSDT"
* `binance_interval`: e.g., "1h"
* `rnd_config`: Contains parameters for the RND calculation (script 04), such as volume/strike thresholds, aggregation method, smoothing sigma, risk-free rate, etc.

## Script Descriptions

* **`src/01_deribit_options_fetcher.py`**: Fetches raw historical option trades from Deribit API for a specific currency, date range, and expiry time derived from the Polymarket filename[cite: 1].
* **`src/02_binance_spot_fetcher.py`**: Fetches historical hourly Kline (OHLCV) data from Binance API for a specific symbol and date range derived from the Polymarket filename[cite: 2].
* **`src/03_options_cleaner.py`**: Cleans raw Deribit trades, filters by expiry date, selects/renames columns, derives strike and type (Call/Put).
* **`src/04_rnd_calculator.py`**: Calculates hourly Risk-Neutral Density from cleaned option data using the Breeden-Litzenberger approach with spline interpolation and smoothing. Includes arbitrage checks and uses parallel processing.
* **`src/05_combine_rnd_polymarket_spot.py`**: Combines hourly RNDs, Polymarket probabilities (parsed from wide format headers), Binance spot prices, and Deribit volume proxies. Integrates RND over Polymarket buckets, normalizes probabilities, and applies outlier capping.
* **`src/06_analyze_report.py`**: Analyzes the combined data, generates summary statistics (including differences, optional DTW, correlations), and creates visualizations comparing Polymarket and RND probabilities over time.
* **`main.py`**: Orchestrates the execution of the pipeline steps based on user selection of a Polymarket input file and derived date parameters. Handles configuration and logging.
