from IPython.display import display
from data_acquisition_coverage_validation import (
    data_acquisition,
    get_crsp_monthly_panel,
)
from minimum_variance_optimization import (
    returns_to_excess,
    backtest_minvar,
    summarize_performance,
    compute_turnover,
    compute_weight_stability,
    make_all_charts,
    show_summary_tables,
    build_topn_indexes,
    build_summary_plus,
)
import numpy as np
import pandas as pd


def main():

    print("Starting data acquisition...")
    # Acquire data
    (
        raw_returns_wide,
        tickers,
        permono_list,
        start_date,
        look_back_period,
        max_weight,
        min_weight,
        risk_free_rate_series,
        df_full,
    ) = data_acquisition()

    # Build top-n indexes for equal-weighted, market-cap weighted, and price-weighted portfolios
    topn_weights_df, topn_raw_returns_df, topn_levels_df = build_topn_indexes(
        df_full, look_back_period
    )

    # Convert topn returns to excess returns
    topn_excess_returns_df = returns_to_excess(
        topn_raw_returns_df, risk_free_rate_series
    )
    for col in topn_excess_returns_df.columns:
        topn_excess_returns_df = topn_excess_returns_df.rename(
            columns={col: col.split("_")[0] + "_excess_return"}
        )

    # Sort the columns alphabetically for easier comparison
    raw_returns_wide = raw_returns_wide.reindex(
        sorted(raw_returns_wide.columns), axis=1
    )

    # Subtract risk-free rate from raw returns to get excess returns
    excess_returns_wide = returns_to_excess(raw_returns_wide, risk_free_rate_series)

    # Backtest minimum variance portfolio
    perf_df, weights_df = backtest_minvar(
        raw_returns_wide, excess_returns_wide, look_back_period, min_weight, max_weight
    )
    perf_df = perf_df.set_index("date")

    # Combine performance dataframes
    perf_df = pd.concat(
        [perf_df, topn_raw_returns_df, topn_excess_returns_df, topn_levels_df], axis=1
    )
    perf_df = perf_df.reset_index()

    # Combine weights dataframes
    weights_df = pd.concat([weights_df, topn_weights_df], axis=0)

    # Calculate average risk-free rate for Sharpe ratio calculation
    monthly_rf = risk_free_rate_series.mean()

    # Summary statistics
    summary_df = summarize_performance(perf_df)
    print("\nSummary statistics for Minimum Variance Portfolio:")
    display(summary_df)

    # Extended summary statistics
    summary_plus_df = build_summary_plus(perf_df, weights_df)
    print("\nExtended Portfolio Summary Statistics")
    display(summary_plus_df)

    # Compute turnover and weight stability
    turnover_df = compute_turnover(weights_df)
    weights_stability_df = compute_weight_stability(weights_df)

    # Plot charts
    make_all_charts(perf_df, turnover_df, weights_stability_df, raw_returns_wide)


if __name__ == "__main__":
    main()
