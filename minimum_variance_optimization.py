import numpy as np
import pandas as pd
import cvxpy as cp
import os
import sys
import math
from copy import deepcopy
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm


# from IPython.display import display

# 3. Minimum Variance Portfolio Optimization
# Calculate the sample covariance matrix from the estimation period return data.
# Implement Ledoit-Wolf covariance shrinkage estimator using the sklearn package.
# Solve the minimum variance portfolio by minimizing portfolio variance subject to:
# Weights summing to 1 (fully invested)
# User-specified long/short position limits
# Use cvxpy to perform the constrained quadratic programming optimization efficiently.

# 4. Rolling Window Out-of-Sample Backtesting
# Implement a rolling window scheme to:
# Re-estimate covariance matrices monthly with a fixed-length look-back window.
# Solve for portfolio weights using both sample and Ledoit-Wolf covariance estimates.
# Compute out-of-sample portfolio returns for the month following each estimation window.

# 5. Performance Metrics and Diagnostics
# Track and report portfolio performance metrics across the backtest period:
# Out-of-sample cumulative returns, annualized return, volatility, Sharpe ratio.
# Portfolio turnover based on weight changes month-to-month.
# Stability of portfolio weights as measured by cross-sectional standard deviation.
# Create and interpret visualization plots including:
# Cumulative return curves.
# Turnover distributions and time series plots.

def returns_to_excess(returns: pd.DataFrame, rf_df: pd.DataFrame | pd.Series) -> pd.DataFrame:
    # If rf_df is a 1-col DataFrame, squeeze to a Series
    out = deepcopy(returns)
    rf = rf_df.squeeze() if isinstance(rf_df, pd.DataFrame) else rf_df
    # Align by index (in case one has extra/missing dates)
    rf = rf.reindex(out.index)
    # Vectorized row-wise subtraction across all columns
    out = out.sub(rf, axis=0)
    return out


def sample_covariance(ret_window):
    return ret_window.cov().values


def ledoit_wolf_covariance(ret_window):
    lw = LedoitWolf().fit(ret_window.values)
    return lw.covariance_


def gmv_weights(cov_matrix, lower_bound=None, upper_bound=None):
    n = cov_matrix.shape[0]
    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, cov_matrix))
    constraints = [cp.sum(w) == 1]
    # Add maximum weight constraints if specified
    if upper_bound is not None:
        constraints.append(w <= upper_bound)
    # Add minimum weight constraints if specified
    if lower_bound is not None:
        constraints.append(w >= lower_bound)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, verbose=False)
    return np.array(w.value).ravel()


def backtest_minvar(raw_returns_wide, excess_returns_wide, window_months=36,
                    lower_bound=None, upper_bound=None):
    dates = excess_returns_wide.index
    tickers = list(excess_returns_wide.columns)
    perf_rows, weight_rows = [], []

    for t in range(window_months, len(dates) - 1):
        est_window = excess_returns_wide.iloc[t - window_months: t]
        # print(est_window.shape)
        next_raw_ret = raw_returns_wide.iloc[t + 1].values
        next_excess_ret = excess_returns_wide.iloc[t + 1].values

        s_cov = sample_covariance(est_window)
        lw_cov = ledoit_wolf_covariance(est_window)

        w_sample = gmv_weights(s_cov, lower_bound, upper_bound)
        w_lw = gmv_weights(lw_cov, lower_bound, upper_bound)

        perf_rows.append({
            "date": dates[t + 1],
            "sample_return": np.dot(w_sample, next_raw_ret),
            "lw_return": np.dot(w_lw, next_raw_ret),
            "sample_excess_return": np.dot(w_sample, next_excess_ret),
            "lw_excess_return": np.dot(w_lw, next_excess_ret)
        })

        for i, tic in enumerate(tickers):
            weight_rows.append(
                {"date": dates[t], "method": "sample", "ticker": tic, "weight": w_sample[i]})
            weight_rows.append(
                {"date": dates[t], "method": "ledoit_wolf", "ticker": tic, "weight": w_lw[i]})

    perf_df = pd.DataFrame(perf_rows).sort_values(
        "date").reset_index(drop=True)
    perf_df["sample_cum"] = (1 + perf_df["sample_return"]).cumprod()
    perf_df["lw_cum"] = (1 + perf_df["lw_return"]).cumprod()
    weights_df = pd.DataFrame(weight_rows)
    return perf_df, weights_df


def summarize_performance(perf_df):
    # summarize performance
    summary = []
    for label, col1, col2 in [("sample", "sample_return", "sample_excess_return"),
                              ("ledoit_wolf", "lw_return", "lw_excess_return"),
                              ("equal_weighted", "ew_return", "ew_excess_return"),
                              ("value_weighted", "vw_return", "vw_excess_return"),
                              ("price_weighted", "pw_return", "pw_excess_return")]:
        ann_mean, ann_std = annualize_mean_std(perf_df[col1])
        sr = sharpe_ratio(perf_df[col2])
        summary.append({"strategy": label, "annual_return": ann_mean,
                        "annual_volatility": ann_std, "sharpe_ratio": sr})
    summary_df = pd.DataFrame(summary)
    return summary_df


def annualize_mean_std(monthly_returns):
    mean_monthly = monthly_returns.mean()
    std_monthly = monthly_returns.std(ddof=1)
    ann_mean = (1 + mean_monthly) ** 12 - 1
    ann_std = std_monthly * math.sqrt(12)
    return ann_mean, ann_std


def sharpe_ratio(monthly_excess_returns):
    # monthly_rf = (1 + annual_rf) ** (1/12) - 1
    # excess = monthly_returns - monthly_rf
    return (monthly_excess_returns.mean()) / (monthly_excess_returns.std(ddof=1) if monthly_excess_returns.std(ddof=1) != 0 else np.nan)


# ========= output helpers (all functions, clean + commented) =========
def turnover_series(weight_list: pd.DataFrame):
    # --- Turnover (per month) ---
    # Turnover ≈ 0.5 * Σ_i |w_t,i − w_{t−1,i}|
    if isinstance(weight_list, pd.DataFrame) and not weight_list.empty:
        W = weight_list.sort_index()
        dW = W.diff().abs()
        turnover = 0.5 * dW.sum(axis=1)
        turnover = turnover.iloc[1:]  # drop first (NaN) period
        turnover.name = "turnover"
    else:
        turnover = pd.Series(dtype=float, name="turnover")

    return turnover


def compute_turnover(weights_df):
    """
    Turnover ≈ 0.5 * Σ_i |w_t,i − w_{t−1,i}|
    weights_df : long-form DataFrame with ['date','method','ticker','weight']
    returns     : DataFrame with ['date','method','turnover']
    """
    out = []
    for method in weights_df['method'].unique():
        w = (weights_df[weights_df['method'] == method]
             .pivot(index='date', columns='ticker', values='weight')
             .sort_index())
        for i in range(1, len(w)):
            prev, curr = w.iloc[i-1].values, w.iloc[i].values
            out.append({
                'date': w.index[i],
                'method': method,
                'turnover': float(0.5 * np.sum(np.abs(curr - prev)))
            })
    return pd.DataFrame(out)


def compute_weight_stability(weights_df: pd.DataFrame):
    out = []
    for method in weights_df['method'].unique():
        w = (weights_df[weights_df['method'] == method]
             .pivot(index='date', columns='ticker', values='weight')
             .sort_index())
        for i in range(len(w)):
            curr = w.iloc[i].values
            out.append({
                'date': w.index[i],
                'method': method,
                'weight_dispersion': float(np.std(curr))
            })
    return pd.DataFrame(out)


def drawdown_series(cum_series):
    """
    compute drawdown series and max drawdown
    cum_series : pd.Series of cumulative returns (growth of $1)
    returns    : (drawdown_series, max_drawdown_as_positive_float)
    """
    peak = cum_series.cummax()
    dd = (cum_series / peak) - 1.0
    mdd = -dd.min()
    return dd, mdd


def rolling_sharpe_series(excess_monthly_returns, window=12):
    """
    rolling (annualized) sharpe using excess returns (subtract monthly rf)
    """
    import math
    # monthly_rf = (1.0 + monthly_rf) ** (1.0 / 12.0) - 1.0
    # excess = monthly_returns - monthly_rf
    rs = (excess_monthly_returns.rolling(window).mean() /
          excess_monthly_returns.rolling(window).std())
    return rs


def plot_cumulative(perf_df):
    """
    line chart: cumulative returns (growth of $1) for sample vs ledoit-wolf
    expects perf_df with columns: date, sample_cum, lw_cum
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9, 5))
    plt.plot(perf_df["date"], perf_df["sample_cum"], label="gmv (sample)")
    plt.plot(perf_df["date"], perf_df["lw_cum"],     label="gmv (ledoit-wolf)")
    # the bench marks should be ploted with dotted lines
    plt.plot(perf_df["date"], perf_df["ew_cum"],     label="equal weighted", linestyle="--")
    plt.plot(perf_df["date"], perf_df["vw_cum"],     label="value weighted", linestyle="--")
    plt.plot(perf_df["date"], perf_df["pw_cum"],     label="price weighted", linestyle="--")
    plt.title("cumulative return (growth of $1)")
    plt.xlabel("date")
    plt.ylabel("cumulative")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_drawdowns(perf_df):
    """
    line chart: drawdowns for sample vs ledoit-wolf
    expects perf_df with columns: date, sample_cum, lw_cum
    """
    import matplotlib.pyplot as plt
    dd_s, mdd_s = drawdown_series(perf_df["sample_cum"])
    dd_l, mdd_l = drawdown_series(perf_df["lw_cum"])
    dd_ew, mdd_ew = drawdown_series(perf_df["ew_cum"])
    dd_vw, mdd_vw = drawdown_series(perf_df["vw_cum"])
    dd_pw, mdd_pw = drawdown_series(perf_df["pw_cum"])

    plt.figure(figsize=(9, 4))
    plt.plot(perf_df["date"], dd_s, label=f"sample (mdd {mdd_s:.2%})")
    plt.plot(perf_df["date"], dd_l, label=f"ledoit-wolf (mdd {mdd_l:.2%})")
    plt.plot(perf_df["date"], dd_ew, label=f"equal weighted (mdd {mdd_ew:.2%})", linestyle="--")
    plt.plot(perf_df["date"], dd_vw, label=f"value weighted (mdd {mdd_vw:.2%})", linestyle="--")
    plt.plot(perf_df["date"], dd_pw, label=f"price weighted (mdd {mdd_pw:.2%})", linestyle="--")
    plt.title("drawdown")
    plt.xlabel("date")
    plt.ylabel("drawdown")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_rolling_sharpe(perf_df, window=12):
    """
    line chart: rolling sharpe for sample vs ledoit-wolf
    expects perf_df with columns: date, sample_return, lw_return
    """
    import matplotlib.pyplot as plt
    rs_s = rolling_sharpe_series(
        perf_df["sample_excess_return"], window=window)
    rs_lw = rolling_sharpe_series(
        perf_df["lw_excess_return"], window=window)
    rs_ew = rolling_sharpe_series(
        perf_df["ew_excess_return"], window=window)
    rs_vw = rolling_sharpe_series(
        perf_df["vw_excess_return"], window=window)
    rs_pw = rolling_sharpe_series(
        perf_df["pw_excess_return"], window=window)

    plt.figure(figsize=(9, 4))
    plt.plot(perf_df["date"], rs_s,  label="sample")
    plt.plot(perf_df["date"], rs_lw, label="ledoit-wolf")
    plt.plot(perf_df["date"], rs_ew, label="equal weighted")
    plt.plot(perf_df["date"], rs_vw, label="value weighted")
    plt.plot(perf_df["date"], rs_pw, label="price weighted")
    plt.title(f"rolling {window}-month sharpe")
    plt.xlabel("date")
    plt.ylabel("sharpe")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_turnover_timeseries(turnover_df):
    """
    line charts: turnover by method over time
    expects turnover_df with columns: date, method, turnover
    """
    import matplotlib.pyplot as plt
    # for method, grp in turnover_df.groupby("method"):
    #     plt.figure(figsize = (9, 3))
    #     plt.plot(grp["date"], grp["turnover"], label = method)
    #     plt.title(f"turnover — {method}")
    #     plt.xlabel("date")
    #     plt.ylabel("0.5 * sum |Δw|")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()
    df = turnover_df.sort_values("date").copy()

    fig, ax = plt.subplots(figsize=(10, 4))
    for method, grp in df.groupby("method"):
        ax.plot(grp["date"], grp["turnover"], label=method)

    ax.set_title("turnover by method over time")
    ax.set_xlabel("date")
    ax.set_ylabel("0.5 * sum |Δw|")
    ax.legend(title="method", ncol=2, loc='upper left')
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def plot_weights_stability_timeseries(weights_stability_df):
    import matplotlib.pyplot as plt

    df = weights_stability_df.sort_values("date").copy()

    fig, ax = plt.subplots(figsize=(10, 4))
    for method, grp in df.groupby("method"):
        ax.plot(grp["date"], grp["weight_dispersion"], label=method)

    ax.set_title("weight dispersion by method over time")
    ax.set_xlabel("date")
    ax.set_ylabel("weight dispersion (std of portfolio weights)")
    ax.legend(title="method", ncol=2, loc='upper left')
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

    # for method, grp in weights_stability_df.groupby("method"):
    #     plt.figure(figsize = (9, 3))
    #     plt.plot(grp["date"], grp["weight_dispersion"], label = method)
    #     plt.title(f"weight_dispersion — {method}")
    #     plt.xlabel("date")
    #     plt.ylabel("weight_dispersion (std of portfolio weights)")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()


def plot_turnover_distribution(turnover_df, bins=30):
    """
    histogram: turnover distribution by method
    expects turnover_df with columns: method, turnover
    """
    import matplotlib.pyplot as plt
    for method, grp in turnover_df.groupby("method"):
        plt.figure(figsize=(7, 4))
        plt.hist(grp["turnover"], bins=bins, alpha=0.85)
        plt.title(f"turnover distribution — {method}")
        plt.xlabel("0.5 * sum |Δw| per month")
        plt.ylabel("frequency")
        plt.tight_layout()
        plt.show()


def plot_corr_heatmap(returns_wide):
    """
    heatmap: asset correlation matrix for the universe used
    expects returns_wide wide DataFrame (assets in columns)
    """
    import matplotlib.pyplot as plt
    corr = returns_wide.corr()
    plt.figure(figsize=(7, 6))
    im = plt.imshow(corr.values, interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("asset correlation (in-sample universe)")
    plt.tight_layout()
    plt.show()


def show_summary_tables(summary_df, perf_df, turnover_df, tail_n=5):
    """
    display key tables inline for grading
    """
    from IPython.display import display
    display(summary_df)
    display(perf_df.tail(tail_n))
    display(turnover_df.tail(tail_n))


def save_csvs(perf_df, weights_df, turnover_df, stability_df, summary_df):
    """
    save results to csv for reporting (kept modular so you can call or skip)
    """
    perf_df.to_csv("oos_performance.csv", index=False)
    weights_df.to_csv("weights_history.csv", index=False)
    turnover_df.to_csv("turnover.csv", index=False)
    stability_df.to_csv("stability.csv", index=False)
    summary_df.to_csv("summary_metrics.csv", index=False)


def make_all_charts(perf_df, turnover_df, weights_stability_df, returns_wide=None):
    """
    one-click plot routine that produces all required visuals
    """
    plot_cumulative(perf_df)
    plot_drawdowns(perf_df)
    plot_rolling_sharpe(perf_df, window=12)
    plot_turnover_timeseries(turnover_df)
    plot_turnover_distribution(turnover_df)
    plot_weights_stability_timeseries(weights_stability_df)
    if returns_wide is not None:
        plot_corr_heatmap(returns_wide)


def build_topn_indexes(panel: pd.DataFrame, look_back_period: int) -> pd.DataFrame:
    """
    Construct Equal-, Value-, and Price-Weighted index levels.

    Methodology:
    - At month t, use constituents & weights formed from month t-1 information.
    - Grow the index from t-1 to t using 'ret' (should include delistings if available).

    Returns
    -------
    (weights_df, returns_df)
      weights_df: columns ['EW Weights','VW Weights','PW Weights'], indexed by month t (from the 2nd month on);
                  each cell is a pd.Series of permno->weight for that month.
      returns_df: columns ['EW Returns','VW Returns','PW Returns'], indexed by month t (from the 2nd month on).
    """

    import pandas as pd
    from tqdm import tqdm

    # Sort & prep
    panel = panel.sort_values(['date', 'permno'])
    # Unique months (respect lookback but keep at least 2 months for t-1/t pairing)
    months_all = pd.Index(panel['date'].drop_duplicates().sort_values())
    months = months_all[look_back_period:]
    if len(months) < 2:
        raise ValueError("Not enough months after look_back_period to compute indices.")

    # Fast month lookup
    by_month = {d: g for d, g in panel.groupby('date')}

    # Levels start at 1.0 at the first available month in our slice
    ew_level = [1.0]
    vw_level = [1.0]
    pw_level = [1.0]

    # Per-month weights and returns (start at t = months[1], since we need t-1)
    ew_w_list, vw_w_list, pw_w_list = [], [], []
    ew_ret_list, vw_ret_list, pw_ret_list = [], [], []

    # Loop over t (needs t-1)
    for i in tqdm(range(1, len(months)), desc="Index Calculation Progress"):
        t_1, t = months[i-1], months[i]
        g_t1 = by_month.get(t_1, pd.DataFrame()).dropna(subset=['mktcap', 'prc'], how='any')
        g_t  = by_month.get(t,   pd.DataFrame())

        if g_t1.empty or g_t.empty:
            # carry forward last level; returns = 0; weights empty
            ew_level.append(ew_level[-1])
            vw_level.append(vw_level[-1])
            pw_level.append(pw_level[-1])

            ew_w_list.append(pd.Series(dtype=float))
            vw_w_list.append(pd.Series(dtype=float))
            pw_w_list.append(pd.Series(dtype=float))

            ew_ret_list.append(0.0)
            vw_ret_list.append(0.0)
            pw_ret_list.append(0.0)
            continue

        # Choose universe by mktcap at t-1 (Top-N if you want: .head(N))
        chosen = g_t1.sort_values('mktcap', ascending=False).set_index('permno')

        # Effective returns at t for chosen names (ensure alignment & numeric dtype)
        g_t = g_t.set_index('permno')
        common = chosen.index.intersection(g_t.index)
        if len(common) == 0:
            ew_level.append(ew_level[-1])
            vw_level.append(vw_level[-1])
            pw_level.append(pw_level[-1])

            ew_w_list.append(pd.Series(dtype=float))
            vw_w_list.append(pd.Series(dtype=float))
            pw_w_list.append(pd.Series(dtype=float))

            ew_ret_list.append(0.0)
            vw_ret_list.append(0.0)
            pw_ret_list.append(0.0)
            continue

        r_t = pd.to_numeric(g_t.loc[common, 'ret'], errors='coerce')

        # ----- Equal-Weighted -----
        r_ew = r_t.dropna()
        if r_ew.empty:
            ew_gross = 1.0
            ew_weights = pd.Series(dtype=float)
        else:
            ew_weights = pd.Series(1.0 / len(r_ew), index=r_ew.index)
            ew_gross = (1.0 + r_ew).mean()

        # ----- Value-Weighted (mktcap at t-1) -----
        w_vw_raw = pd.to_numeric(chosen.loc[common, 'mktcap'], errors='coerce')
        mask_vw = r_t.notna() & w_vw_raw.notna() & (w_vw_raw > 0)
        r_vw = r_t.loc[mask_vw]
        if r_vw.empty:
            vw_gross = 1.0
            vw_weights = pd.Series(dtype=float)
        else:
            w_vw = w_vw_raw.loc[mask_vw]
            w_vw = w_vw / w_vw.sum()
            vw_weights = w_vw.copy()
            vw_gross = ((1.0 + r_vw) * w_vw).sum()

        # ----- Price-Weighted (price at t-1) -----
        w_pw_raw = pd.to_numeric(chosen.loc[common, 'prc'], errors='coerce').abs()
        mask_pw = r_t.notna() & w_pw_raw.notna() & (w_pw_raw > 0)
        r_pw = r_t.loc[mask_pw]
        if r_pw.empty:
            pw_gross = 1.0
            pw_weights = pd.Series(dtype=float)
        else:
            w_pw = w_pw_raw.loc[mask_pw]
            w_pw = w_pw / w_pw.sum()
            pw_weights = w_pw.copy()
            pw_gross = ((1.0 + r_pw) * w_pw).sum()

        # Update levels
        ew_level.append(ew_level[-1] * float(ew_gross))
        vw_level.append(vw_level[-1] * float(vw_gross))
        pw_level.append(pw_level[-1] * float(pw_gross))

        # Store weights (at month t) and returns (gross-1)
        ew_w_list.append(ew_weights)
        vw_w_list.append(vw_weights)
        pw_w_list.append(pw_weights)

        ew_ret_list.append(float(ew_gross - 1.0))
        vw_ret_list.append(float(vw_gross - 1.0))
        pw_ret_list.append(float(pw_gross - 1.0))

    # -------- Assemble outputs --------
    idx_months = pd.to_datetime(months[:-1])           # all months incl. the first (level=1.0)
    idx_rt     = pd.to_datetime(months[1:])       # returns/weights start at second month

    # Weights: store the per-month Series in an object-dtype Series
    ew_weights_s = pd.Series(ew_w_list, index=idx_months, name='EW Weights')
    vw_weights_s = pd.Series(vw_w_list, index=idx_months, name='VW Weights')
    pw_weights_s = pd.Series(pw_w_list, index=idx_months, name='PW Weights')
    weights_df   = pd.concat([ew_weights_s, vw_weights_s, pw_weights_s], axis=1)

    # Returns
    ew_returns_s = pd.Series(ew_ret_list, index=idx_rt, name='ew_return')
    vw_returns_s = pd.Series(vw_ret_list, index=idx_rt, name='vw_return')
    pw_returns_s = pd.Series(pw_ret_list, index=idx_rt, name='pw_return')
    returns_df   = pd.concat([ew_returns_s, vw_returns_s, pw_returns_s], axis=1)

    weights_df = weights_to_long(weights_df, panel, map_from_t_minus_1=True)
    # # (Optional) Levels, if you want them:
    levels_df = pd.DataFrame({
        'ew_cum': ew_level[1:],
        'vw_cum': vw_level[1:],
        'pw_cum': pw_level[1:],
    }, index=idx_rt)

    # print(levels_df.tail())

    return weights_df, returns_df, levels_df  # or (levels_df, weights_df, returns_df)


def weights_to_long(weights_df: pd.DataFrame, panel: pd.DataFrame,
                    map_from_t_minus_1: bool = True) -> pd.DataFrame:
    """
    Convert the 'weights_df' returned by build_topn_indexes into long/tidy format:

        date       method   ticker   weight

    Assumptions
    -----------
    - weights_df index = month-end Timestamp for t (starts at months[1:])
    - Each cell in weights_df is a pd.Series indexed by PERMNO with weight values
    - 'panel' has at least ['date','permno','ticker'] (one row per stock per month)
    - If tickers change over time, mapping can be taken from t-1 (default) or t
    """
    import pandas as pd
    from pandas.tseries.offsets import MonthEnd

    # Keep only what's needed and make a month→(permno→ticker) map
    if not {'date','permno','ticker'}.issubset(panel.columns):
        raise ValueError("`panel` must contain columns: 'date', 'permno', 'ticker'.")

    p = panel[['date', 'permno', 'ticker']].drop_duplicates()
    month_map = {d: g.set_index('permno')['ticker'] for d, g in p.groupby('date')}

    rows = []
    for dt, row in weights_df.iterrows():
        # Choose which month to use for the permno→ticker mapping
        key = (dt - MonthEnd(1)) if map_from_t_minus_1 else dt
        tickmap = month_map.get(key, month_map.get(dt, pd.Series(dtype=object)))

        for method_name, s in row.items():
            if s is None or len(s) == 0:
                continue
            # Ensure it's a Series with permno index
            s = pd.Series(s, copy=False)
            df = s.rename('weight').reset_index().rename(columns={'index': 'permno'})
            # Map ticker; if missing, fall back to permno as string
            df['ticker'] = df['permno'].map(tickmap).fillna(df['permno'].astype(str))
            df['date'] = pd.to_datetime(dt)
            # Clean method label (e.g., "EW Weights" -> "ew")
            df['method'] = (str(method_name)
                            .replace(' Weights', '')
                            .replace('_weights', '')
                            .strip()
                            .lower())
            rows.append(df[['date', 'method', 'ticker', 'weight']])

    out = (pd.concat(rows, ignore_index=True)
             .sort_values(['date', 'method', 'ticker'])
             .reset_index(drop=True))
    return out


# def build_topn_indexes(panel: pd.DataFrame, look_back_period: int) -> pd.DataFrame:
#     """
#     Construct Equal-, Value-, and Price-Weighted index levels.

#     Methodology reminder:
#     - At month t, choose constituents and weights from t-1 information.
#     - Grow index from t-1 to t using ret_eff (includes delistings).

#     Returns
#     -------
#     DataFrame with three columns (EW, VW, PW) indexed by month t.
#     """

#     # Organize by month for quick lookup
#     panel = panel.sort_values(['date', 'permno'])
#     months = sorted(panel['date'].unique())
#     months = months[look_back_period:] 
#     num_stocks = panel['permno'].nunique()

#     by_month = {d: g for d, g in panel.groupby('date')}

#     # Initialize index levels
#     ew_level = [1.0]
#     vw_level = [1.0]
#     pw_level = [1.0]

#     ew_weights = [1/num_stocks] * num_stocks  # fixed equal weights
#     vw_weights = []
#     pw_weights = []

#     ew_returns = []
#     vw_returns = []
#     pw_returns = []

#     # Loop from 2nd available month; need t-1 and t
#     for i in tqdm(range(1, len(months)), desc="Index Calculation Progress"):
#         t_1, t = months[i-1], months[i]
#         g_t1 = by_month[t_1].dropna(subset=['mktcap', 'prc'])
#         g_t = by_month[t]

#         if g_t1.empty or g_t.empty:
#             # carry forward if missing data
#             ew_level.append(ew_level[-1])
#             vw_level.append(vw_level[-1])
#             pw_level.append(pw_level[-1])
#             continue

#         # Top-N by market cap at t-1
#         chosen = g_t1.sort_values(
#             'mktcap', ascending=False).set_index('permno')

#         # Ensure we have returns at t for the chosen PERMNOs
#         g_t = g_t.set_index('permno')
#         # Find intersection of chosen stocks and those with returns at t
#         common = chosen.index.intersection(g_t.index)
#         # If no common stocks, carry forward previous index levels
#         if len(common) == 0:
#             ew_level.append(ew_level[-1])
#             vw_level.append(vw_level[-1])
#             pw_level.append(pw_level[-1])
#             continue

#         # Effective returns at t for the chosen stocks
#         r_t = g_t.loc[common, 'ret'].astype(float)

#         # Equal-Weighted: average of gross returns
#         r_t_ew = r_t.dropna()
#         ew_gross = (1.0 + r_t_ew).mean() if not r_t_ew.empty else 1.0

#         # Value-Weighted: weights from t-1 market cap
#         w_vw = chosen.loc[common, 'mktcap'].astype(float)
#         mask_vw = r_t.notna()
#         # Restrict to stocks with non-missing returns at t
#         w_vw = w_vw.loc[mask_vw]
#         r_vw = r_t.loc[mask_vw]
#         if r_vw.empty or w_vw.sum() <= 0:
#             vw_gross = 1.0
#         else:
#             w_vw = w_vw / w_vw.sum()
#             vw_gross = ((1.0 + r_vw) * w_vw).sum()
#         vw_weights.append(w_vw)

#         # Price-Weighted: weights from t-1 price
#         w_pw = chosen.loc[common, 'prc'].astype(float)
#         mask_pw = r_t.notna()
#         # Restrict to stocks with non-missing returns at t
#         w_pw = w_pw.loc[mask_pw]
#         r_pw = r_t.loc[mask_pw]
#         if r_pw.empty or w_pw.sum() <= 0:
#             pw_gross = 1.0
#         else:
#             w_pw = w_pw / w_pw.sum()
#             pw_gross = ((1.0 + r_pw) * w_pw).sum()
#         pw_weights.append(w_pw)

#         # if (i == len(months) - 1) :
#             # sort descending, keep the index
#             # print(sorted(w_vw.items(), key=lambda x: x[1], reverse=True))
#             # print(sorted(w_pw.items(), key=lambda x: x[1], reverse=True))

#         # Compound index levels
#         ew_level.append(ew_level[-1] * float(ew_gross))
#         vw_level.append(vw_level[-1] * float(vw_gross))
#         pw_level.append(pw_level[-1] * float(pw_gross))

#         ew_returns.append(float(ew_gross))
#         vw_returns.append(float(vw_gross))
#         pw_returns.append(float(pw_gross))

#     # Assemble output DataFrame
#     # Convert to tidy DataFrame indexed by month t
#     idx_months = pd.to_datetime(months[:])
#     # ew = pd.Series(ew_level[:], index=idx_months, name='Equal-Weighted')
#     # vw = pd.Series(vw_level[:], index=idx_months, name='Value-Weighted')
#     # pw = pd.Series(pw_level[:], index=idx_months, name='Price-Weighted')

#     ew_weights = pd.Series(ew_weights, index=idx_months[1:], name='EW Weights')
#     vw_weights = pd.Series(vw_weights, index=idx_months[1:], name='VW Weights')
#     pw_weights = pd.Series(pw_weights, index=idx_months[1:], name='PW Weights')
#     weights_df = pd.concat([ew_weights, vw_weights, pw_weights], axis=1)

#     ew_returns = pd.Series(ew_returns, index=idx_months[:], name='EW Returns')
#     vw_returns = pd.Series(vw_returns, index=idx_months[:], name='VW Returns')
#     pw_returns = pd.Series(pw_returns, index=idx_months[:], name='PW Returns')
#     returns_df = pd.concat([ew_returns, vw_returns, pw_returns], axis=1)

#     # Combine all index levels into a single DataFrame
#     # return pd.concat([ew, vw, pw], axis=1)

#     return weights_df, returns_df


# def summary_statistics(weight_list: pd.DataFrame,
#                        return_list: pd.DataFrame,
#                        rf_yearly: float):
#     """
#     Compute portfolio performance and diagnostics.

#     Parameters
#     ----------
#     weight_list : pd.DataFrame
#         Rows = dates (ascending), columns = assets, values = weights at each rebalance.
#     return_list : pd.DataFrame or pd.Series
#         Out-of-sample portfolio returns by month. If DataFrame, must contain a column
#         named 'portfolio_return'.
#     rf : float or pd.Series, optional
#         Monthly risk-free rate (scalar or Series indexed by the same dates as returns).
#     """
#     rf_monthly = rf_yearly / 12.0
#     r = return_list.squeeze().astype(float).copy()

#     # --- Cumulative returns ---
#     cumulative_returns = (1.0 + r).cumprod()

#     # --- Annualized metrics (assume monthly frequency) ---
#     periods_per_year = 12.0
#     n = len(r)
#     if n == 0:
#         annualized_return = np.nan
#         annualized_volatility = np.nan
#         sharpe_ratio = np.nan
#     else:
#         # Geometric annualized return
#         total_return = (1.0 + r).prod()
#         annualized_return = total_return ** (periods_per_year / n) - 1.0

#         # Annualized volatility from monthly std
#         monthly_std = r.std(ddof=1)
#         annualized_volatility = monthly_std * np.sqrt(periods_per_year)

#         # Sharpe ratio from monthly excess returns
#         excess = (r - rf_monthly).dropna()
#         if len(excess) < 2 or excess.std(ddof=1) == 0:
#             sharpe_ratio = np.nan
#         else:
#             sharpe_ratio = (excess.mean()) / (excess.std(ddof=1))


#     return (cumulative_returns,
#             float(annualized_return) if pd.notna(
#                 annualized_return) else np.nan,
#             float(annualized_volatility) if pd.notna(
#                 annualized_volatility) else np.nan,
#             float(sharpe_ratio) if pd.notna(sharpe_ratio) else np.nan,
#             turnover,
#             weight_stability)


# def sample_covariance_matrix(returns, start_date, end_date):
#     """
#     Calculate the sample covariance matrix of asset returns.

#     Parameters:
#     returns (pd.DataFrame): DataFrame where each column represents an asset's returns over time.
#     start_date (pd.Timestamp): Start date for the analysis period.
#     end_date (pd.Timestamp): End date for the analysis period.

#     Returns:
#     pd.DataFrame: Sample covariance matrix of the asset returns.
#     """
#     # Filter returns for the specified date range
#     filtered_returns = returns.loc[start_date:end_date]
#     mean = filtered_returns.mean()
#     # print("Mean returns:")
#     # print(mean)

#     # Subtract the mean return from each asset's returns
#     filtered_returns = filtered_returns - mean
#     # print("Centered returns:")
#     # print(filtered_returns)

#     return filtered_returns.cov()


# def ledoit_wolf_shrinkage(returns, start_date, end_date):
#     """
#     Apply Ledoit-Wolf shrinkage to estimate a more robust covariance matrix.

#     Parameters:
#     returns (pd.DataFrame): DataFrame where each column represents an asset's returns over time.

#     Returns:
#     pd.DataFrame: Shrunk covariance matrix of the asset returns.
#     """
#     filtered_returns = returns.loc[start_date:end_date]
#     mean = filtered_returns.mean()

#     filtered_returns = filtered_returns - mean

#     lw = LedoitWolf()
#     lw.fit(filtered_returns)
#     shrunk_cov = lw.covariance_

#     return pd.DataFrame(shrunk_cov, index=returns.columns, columns=returns.columns)


# def minimum_variance_optimization(cov_matrix, max_weight, min_weight):
#     """
#     Perform minimum variance portfolio optimization with optional weight constraints.

#     Parameters:
#     cov_matrix (pd.DataFrame): Covariance matrix of asset returns.
#     max_weight (float, optional): Maximum weight constraint for each asset. Defaults to None.
#     min_weight (float, optional): Minimum weight constraint for each asset. Defaults to None.

#     Returns:
#     np.ndarray: Optimal weights for the minimum variance portfolio.
#     """
#     n = cov_matrix.shape[0]
#     w = cp.Variable(n)

#     # Objective: Minimize portfolio variance
#     objective = cp.Minimize(cp.quad_form(w, cov_matrix.values))

#     # Constraints: Weights sum to 1
#     constraints = [cp.sum(w) == 1]

#     # Add maximum weight constraints if specified
#     if max_weight is not None:
#         constraints.append(w <= max_weight)

#     # Add minimum weight constraints if specified
#     if min_weight is not None:
#         constraints.append(w >= min_weight)

#     # Define and solve the problem
#     prob = cp.Problem(objective, constraints)
#     prob.solve()

#     return w.value


# def mvo_sample_covmat(df, start_date, end_date, max_weight, min_weight):

#     covmat = sample_covariance_matrix(df, start_date, end_date)
#     # print("Sample covariance matrix:")
#     # print(covmat)

#     mvo_sample = minimum_variance_optimization(covmat, max_weight, min_weight)
#     mvo_sample = pd.Series(mvo_sample, index=df.columns)
#     # Order the tickers by alphabetical order for easier comparison
#     print("Minimum variance optimization (sample covariance):")
#     print(mvo_sample)
#     print("Sum of weights:", np.round(mvo_sample.sum(), 2))

#     print("Next month returns:")
#     next_month_returns = returns_calculation(df, mvo_sample, end_date)
#     print(next_month_returns)


# def mvo_lwo_covmat(df, start_date, end_date, max_weight, min_weight):
#     lw_covmat = ledoit_wolf_shrinkage(df, start_date, end_date)
#     # print("Ledoit-Wolf shrunk covariance matrix:")
#     # print(lw_covmat)

#     mvo_lw = minimum_variance_optimization(lw_covmat, max_weight, min_weight)
#     mvo_lw = pd.Series(mvo_lw, index=df.columns)
#     # Order the tickers by alphabetical order for easier comparison
#     print("Minimum variance optimization (Ledoit-Wolf covariance):")
#     print(mvo_lw)
#     print("Sum of weights:", np.round(mvo_lw.sum(), 2))

#     print("Next month returns:")
#     next_month_returns = returns_calculation(df, mvo_lw, end_date)
#     print(next_month_returns)


# def rolling_window_optimization(returns, window_size, start_date, end_date,
#                                 first_delisting_date, max_weight=None, min_weight=None):
#     """
#     Perform rolling-window minimum variance optimization using both
#     sample covariance and Ledoit-Wolf covariance.

#     Returns
#     -------
#     (mvo_sample_weights, mvo_sample_returns, mvo_lvo_weights, mvo_lvo_returns)
#         Each weights DataFrame has index = window end date, columns = tickers.
#         Each returns DataFrame has a single column 'portfolio_return'
#         and index = the *next* month after the window end.
#     """
#     if not isinstance(returns.index, pd.DatetimeIndex):
#         raise TypeError("returns must have a DatetimeIndex.")

#     effective_end = returns.index[-1]
#     effective_start = returns.index[0]

#     # Restrict sample
#     sample = returns.loc[effective_start:effective_end].copy()
#     if sample.shape[0] < window_size + 1:
#         print(sample.shape, window_size)
#         # need at least window_size rows plus 1 future row for realized next-month return
#         print("Not enough data for the rolling window optimization.")
#         empty_w = pd.DataFrame(columns=returns.columns)
#         empty_r = pd.DataFrame(columns=["portfolio_return"])
#         return empty_w.copy(), empty_r.copy(), empty_w.copy(), empty_r.copy()

#     idx = sample.index
#     nrows = len(idx)

#     # Collectors
#     mvo_sample_weights = pd.DataFrame(columns=returns.columns)
#     mvo_sample_returns = pd.DataFrame(columns=["portfolio_return"])
#     mvo_lvo_weights = pd.DataFrame(columns=returns.columns)
#     mvo_lvo_returns = pd.DataFrame(columns=["portfolio_return"])

#     # We will end each window at position e (inclusive); need a next row at e+1 for realized return.
#     # e runs from window_size-1 to nrows-2 (so that e+1 exists).
#     for e in range(window_size - 1, nrows - 1):
#         s = e - window_size + 1
#         start_date_window = idx[s]
#         end_date_window = idx[e]

#         # --- Sample covariance path ---
#         covmat = sample_covariance_matrix(
#             returns, start_date_window, end_date_window)
#         mvo_sample = minimum_variance_optimization(
#             covmat, max_weight, min_weight)
#         mvo_sample = pd.Series(
#             mvo_sample, index=returns.columns, name=end_date_window)

#         # Save weights (row = window end)
#         mvo_sample_weights.loc[end_date_window] = mvo_sample

#         # Realized next-month return
#         next_month_return = returns_calculation(
#             returns, mvo_sample, end_date_window)  # 1-element Series
#         # ensure the target exists with the right dtype (do this once before the loop)
#         if "portfolio_return" not in mvo_sample_returns.columns:
#             mvo_sample_returns["portfolio_return"] = pd.Series(dtype=float)

#         # next_month_return is a 1-element Series indexed by the next date
#         for dt, val in next_month_return.items():
#             mvo_sample_returns.loc[pd.Timestamp(
#                 dt), "portfolio_return"] = float(val)

#         # --- Ledoit–Wolf path ---
#         lw_covmat = ledoit_wolf_shrinkage(
#             returns, start_date_window, end_date_window)
#         mvo_lw = minimum_variance_optimization(
#             lw_covmat, max_weight, min_weight)
#         mvo_lw = pd.Series(mvo_lw, index=returns.columns, name=end_date_window)

#         mvo_lvo_weights.loc[end_date_window] = mvo_lw

#         next_month_return_lw = returns_calculation(
#             returns, mvo_lw, end_date_window)
#         # ensure the target exists with the right dtype (do this once before the loop)
#         if "portfolio_return" not in mvo_lvo_returns.columns:
#             mvo_lvo_returns["portfolio_return"] = pd.Series(dtype=float)

#         # next_month_return is a 1-element Series indexed by the next date
#         for dt, val in next_month_return_lw.items():
#             mvo_lvo_returns.loc[pd.Timestamp(
#                 dt), "portfolio_return"] = float(val)

#     # Ensure proper dtypes/index names
#     mvo_sample_weights.index.name = "date"
#     mvo_lvo_weights.index.name = "date"
#     mvo_sample_returns.index.name = "date"
#     mvo_lvo_returns.index.name = "date"

#     return mvo_sample_weights, mvo_sample_returns, mvo_lvo_weights, mvo_lvo_returns


# def returns_calculation(df: pd.DataFrame, weights: pd.Series, end_date: str | pd.Timestamp) -> pd.Series:
#     """
#     Compute the portfolio return for the month immediately AFTER end_date.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Monthly returns. Index = DatetimeIndex (ascending). Columns = assets.
#     weights : pd.Series
#         Portfolio weights indexed by the same asset names as df.columns.
#         Missing assets are treated as 0.
#     end_date : str or Timestamp
#         'YYYY-MM-DD' (or Timestamp). We select the last date <= end_date, then use the next row.

#     Returns
#     -------
#     pd.Series
#         One-element series: index is the next month’s date, value is the portfolio return.
#     """
#     if not isinstance(df.index, pd.DatetimeIndex):
#         raise TypeError("df index must be a DatetimeIndex.")
#     if df.empty:
#         raise ValueError("df is empty.")

#     # Ensure ascending order
#     if not df.index.is_monotonic_increasing:
#         df = df.sort_index()

#     end_ts = pd.to_datetime(end_date)

#     # Find the position of the last index value <= end_ts
#     pos = df.index.get_indexer([end_ts], method="pad")[0]
#     if pos == -1:
#         raise ValueError("end_date is before the first date in df.")

#     next_pos = pos + 1
#     if next_pos >= len(df):
#         raise ValueError(
#             "End date is too close to the end to get the next month's return.")

#     next_date = df.index[next_pos]
#     # Series of per-asset returns at next month
#     r_next = df.iloc[next_pos]

#     # Align weights to columns; treat missing weights as 0
#     w = weights.reindex(df.columns).fillna(0.0)

#     # Dot product = portfolio return
#     port_ret = float(r_next.dot(w))

#     return pd.Series([port_ret], index=[next_date], name="portfolio_return")


# def summary_statistics(weight_list: pd.DataFrame,
#                        return_list: pd.DataFrame,
#                        rf_yearly: float):
#     """
#     Compute portfolio performance and diagnostics.

#     Parameters
#     ----------
#     weight_list : pd.DataFrame
#         Rows = dates (ascending), columns = assets, values = weights at each rebalance.
#     return_list : pd.DataFrame or pd.Series
#         Out-of-sample portfolio returns by month. If DataFrame, must contain a column
#         named 'portfolio_return'.
#     rf : float or pd.Series, optional
#         Monthly risk-free rate (scalar or Series indexed by the same dates as returns).
#     """
#     rf_monthly = rf_yearly / 12.0
#     r = return_list.squeeze().astype(float).copy()

#     # --- Cumulative returns ---
#     cumulative_returns = (1.0 + r).cumprod()

#     # --- Annualized metrics (assume monthly frequency) ---
#     periods_per_year = 12.0
#     n = len(r)
#     if n == 0:
#         annualized_return = np.nan
#         annualized_volatility = np.nan
#         sharpe_ratio = np.nan
#     else:
#         # Geometric annualized return
#         total_return = (1.0 + r).prod()
#         annualized_return = total_return ** (periods_per_year / n) - 1.0

#         # Annualized volatility from monthly std
#         monthly_std = r.std(ddof=1)
#         annualized_volatility = monthly_std * np.sqrt(periods_per_year)

#         # Sharpe ratio from monthly excess returns
#         excess = (r - rf_monthly).dropna()
#         if len(excess) < 2 or excess.std(ddof=1) == 0:
#             sharpe_ratio = np.nan
#         else:
#             sharpe_ratio = (excess.mean()) / (excess.std(ddof=1))

#     # --- Turnover (per month) ---
#     # Turnover ≈ 0.5 * Σ_i |w_t,i − w_{t−1,i}|
#     if isinstance(weight_list, pd.DataFrame) and not weight_list.empty:
#         W = weight_list.sort_index()
#         dW = W.diff().abs()
#         turnover = 0.5 * dW.sum(axis=1)
#         turnover = turnover.iloc[1:]  # drop first (NaN) period
#         turnover.name = "turnover"
#     else:
#         turnover = pd.Series(dtype=float, name="turnover")

#     # --- Weight stability (cross-sectional std each month) ---
#     if isinstance(weight_list, pd.DataFrame) and not weight_list.empty:
#         weight_stability = weight_list.std(axis=1)
#     else:
#         weight_stability = pd.Series(dtype=float, name="weight_stability")

#     return (cumulative_returns,
#             float(annualized_return) if pd.notna(
#                 annualized_return) else np.nan,
#             float(annualized_volatility) if pd.notna(
#                 annualized_volatility) else np.nan,
#             float(sharpe_ratio) if pd.notna(sharpe_ratio) else np.nan,
#             turnover,
#             weight_stability)


# def cumulative_returns_plot(cumulative_returns_sample, cumulative_returns_lwo):
#     # ensure datetime-like x
#     x1 = getattr(cumulative_returns_sample.index, "to_timestamp",
#                  lambda: cumulative_returns_sample.index)()
#     x2 = getattr(cumulative_returns_lwo.index, "to_timestamp",
#                  lambda: cumulative_returns_lwo.index)()

#     fig, ax = plt.subplots(figsize=(10, 6))

#     ax.plot(x1, cumulative_returns_sample.values,
#             label='Cumulative Returns (Sample)')
#     ax.plot(x2, cumulative_returns_lwo.values,
#             label='Cumulative Returns (LWO)')

#     ax.set_xlabel('Month')
#     ax.set_ylabel('Cumulative Returns')
#     ax.set_title('Cumulative Returns Over Time')
#     ax.legend()
#     ax.grid(True)

#     locator = mdates.AutoDateLocator()
#     formatter = mdates.ConciseDateFormatter(locator)
#     ax.xaxis.set_major_locator(locator)
#     ax.xaxis.set_major_formatter(formatter)
#     fig.autofmt_xdate()  # tilt labels if crowded
#     plt.show()


# def turnover_plot(turnover_sample, turnover_lwo):
#     # ensure datetime index (works for DatetimeIndex or PeriodIndex)
#     x1 = getattr(turnover_sample.index, "to_timestamp",
#                  lambda: turnover_sample.index)()
#     x2 = getattr(turnover_lwo.index, "to_timestamp",
#                  lambda: turnover_lwo.index)()

#     fig, ax = plt.subplots(figsize=(10, 6))

#     # plot using index dates directly
#     ax.plot(x1, turnover_sample.values, label='Portfolio Turnover (Sample)')
#     ax.plot(x2, turnover_lwo.values, label='Portfolio Turnover (LWO)')

#     ax.set_xlabel('Date')
#     ax.set_ylabel('Turnover')
#     ax.set_title('Portfolio Turnover Over Time')
#     ax.legend()
#     ax.grid(True)

#     # show only dates on x-axis, nicely formatted
#     locator = mdates.AutoDateLocator()
#     formatter = mdates.ConciseDateFormatter(locator)
#     ax.xaxis.set_major_locator(locator)
#     ax.xaxis.set_major_formatter(formatter)
#     fig.autofmt_xdate()  # tilt labels if crowded

#     plt.show()
