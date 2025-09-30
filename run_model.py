# get the full data set.
# from start to end date, get the length of the training data
# then use that length to get the rolling window of returns

# how to get full data set, and how to get the end of the data set
# idea: use min end date to get the full data set
# use max begin date to get beginning of the full data set


from data_acquisition_coverage_validation import *
from minimum_variance_optimization import *


def main():

    df, tickers, start_date, end_date, first_delisting_date, \
        max_weight, min_weight, window_size, risk_free_rate_annual = data_acquisition()
    # print("Data acquisition complete.")
    # print(tickers)
    # print(df)

    # Sort the columns alphabetically for easier comparison
    df = df.reindex(sorted(df.columns), axis=1)

    mvo_sample_covmat(df, start_date, end_date, max_weight, min_weight)
    print("\n")
    mvo_lwo_covmat(df, start_date, end_date, max_weight, min_weight)
    print("\n")

    mvo_sample_weights, mvo_sample_returns, mvo_lwo_weights, mvo_lwo_returns \
        = rolling_window_optimization(df, window_size, start_date, end_date,
                                      first_delisting_date, max_weight, min_weight)

    # print(mvo_sample_weights)
    # print(mvo_sample_returns)
    # print(mvo_lwo_weights)
    # print(mvo_lwo_returns)

    # Summary statistics
    print("Summary statistics for MVO with Sample Covariance Matrix:")
    cumulative_returns_sample, annualized_return_sample, annualized_volatility_sample, sharpe_ratio_sample, \
        turnover_sample, weight_stability_sample = \
        summary_statistics(mvo_sample_weights,
                           mvo_sample_returns, risk_free_rate_annual)
    # print('cumulative_return:', cumulative_returns_sample)
    print('annualized_return:', np.round(annualized_return_sample, 4))
    print('annualized_volatility:', np.round(annualized_volatility_sample, 4))
    print('sharpe_ratio:', np.round(sharpe_ratio_sample, 4))
    print('turnover (per month):', np.round(np.mean(turnover_sample), 4))
    # print('weight_stability:', weight_stability_sample, '\n\n')
    print('\n')

    print("Summary statistics for MVO with Ledoit-Wolf Covariance Matrix:")
    cumulative_returns_lwo, annualized_return_lwo, annualized_volatility_lwo, sharpe_ratio_lwo, \
        turnover_lwo, weight_stability_lwo = \
        summary_statistics(mvo_lwo_weights, mvo_lwo_returns, risk_free_rate_annual)
    # print('cumulative_return:', cumulative_returns_lwo)
    print('annualized_return:', np.round(annualized_return_lwo, 4))
    print('annualized_volatility:', np.round(annualized_volatility_lwo, 4))
    print('sharpe_ratio:', np.round(sharpe_ratio_lwo, 4))
    print('turnover (per month):', np.round(np.mean(turnover_lwo), 4))
    # print('weight_stability:', weight_stability_lwo)

    cumulative_returns_plot(cumulative_returns_sample, cumulative_returns_lwo)

    turnover_plot(turnover_sample, turnover_lwo)


if __name__ == "__main__":
    main()
