import pandas as pd
import numpy as np
from datetime import datetime as dt
import wrds
from pandas.tseries.offsets import MonthEnd

# Data Acquisition and Coverage Validation
# Connect to the WRDS database to retrieve monthly return data for a user-defined set of stock
# tickers.
# Implement a ticker data coverage validation routine that ensures stocks have sufficient
# consecutive return observations in an estimation window, allowing for limited missing months.
# Design a user-interaction loop to replace insufficient tickers with alternatives until all tickers
# pass the coverage test.


def data_acquisition():

    # Connect to WRDS database
    db = db_connect()

    # Get user inputs
    tickers = ticker_input()
    start_date_eval, months = date_input()
    max_missing = missing_months_input()
    all_data = pd.DataFrame()

    shrcd_list = [10, 11]  # Common shares

    # Since the WRDS CRSP database is updated at the end of each year, 
    # set the end of df to 12/31 of last year from today's date
    # today = pd.Timestamp.today()
    # end_date_df = pd.Timestamp(year=today.year - 1, month=12, day=31)

    # Get valid PERMNOs for the tickers
    permno_list, valid_tickers = get_active_permnos(db, tickers, start_date_eval, max_missing, shrcd_list)

    # Get returns data
    all_data = get_returns(db, permno_list, start_date_eval.strftime('%Y-%m-%d'))
    # Get delisting returns data
    # delisting_data = get_delistings(db, permno_list, start_date_eval.strftime('%Y-%m-%d'),
    #                                 end_date_eval.strftime('%Y-%m-%d'))
    # Add effective returns
    # all_data = add_effective_returns(all_data, delisting_data)

    permno_to_ticker = dict(zip(permno_list, valid_tickers))
    all_data = all_data.rename(columns=permno_to_ticker)

    # Ask the user if we should set weight constraints
    max_weight, min_weight = weight_constraint_input()

    # Ask the user for the rolling window size (in months)
    window_size = window_size_input()

    # Ask the user for the annual risk-free rate (as a decimal)
    # risk_free_rate = risk_free_rate_input()
    end_date_eval = all_data.index.max()
    # print("End date for evaluation (last available data):", end_date_eval)
    risk_free_rate_series = get_risk_free_rate_series(db, start_date_eval, end_date_eval)

    # print(all_data)
    # print(valid_tickers)
    # print(start_date_eval, months)
    # print(max_weight, min_weight)
    # print(window_size)
    # print(risk_free_rate_series)

    return all_data, valid_tickers, start_date_eval, months, \
        max_weight, min_weight, window_size, risk_free_rate_series


def db_connect():
    # Connect to WRDS database
    try:
        db = wrds.Connection(wrds_username='rio_yoko',
                             wrds_password='yokoyama0928')
        print("Connected to WRDS database.")
        return db
    except Exception as e:
        print("Failed to connect to WRDS database:", e)
        raise


def ticker_input():
    # Ask the user for a list of tickers (comma separated) with error handling
    while True:
        tickers = input(
            "Enter a list of stock tickers (comma separated): ").split(',')
        tickers = [ticker.strip().upper() for ticker in tickers]
        if not tickers or tickers == ['']:
            print("Invalid input. Please enter at least one ticker.")
        else:
            break
    print(f"Tickers: {tickers}")
    return tickers


def date_input():
    """
    Prompt for a start date (YYYY-MM-DD) and number of months (positive int),
    then return (start_date, number of months).
    """
    while True:
        try:
            start_str = input("Enter the start date (YYYY-MM-DD): ").strip()
            months_str = input("Enter the number of months for the analysis: ").strip()

            # strict date parsing (won't accept "0")
            start_date = dt.strptime(start_str, "%Y-%m-%d")
            num_months = int(months_str)

            if num_months <= 0:
                print("Number of months must be a positive integer. Please try again.")
                continue

            end_date = pd.Timestamp(start_date) + pd.DateOffset(months=num_months-1)
            print(
                f"Date range: {pd.Timestamp(start_date).date()} to {end_date.date()}")
            return pd.Timestamp(start_date), months_str

        except ValueError:
            print("Invalid input. Use YYYY-MM-DD for the date and a positive integer for months.")


def missing_months_input():
    # Ask the user for the maximum number of allowed missing months with error handling
    while True:
        try:
            max_missing = int(
                input("Enter the maximum number of allowed missing months: "))
            if max_missing < 0:
                print("Please enter a non-negative integer.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a non-negative integer.")
    print(f"Maximum allowed missing months: {max_missing}")
    return max_missing


def get_active_permnos(db, tickers: list, start_date: pd.Timestamp, max_missing: int, shrcd_list: list):
    """
    Get active PERMNOs for the given tickers and date range.
    """
    permno_list = []
    valid_tickers = []
    # date_of_delisting = []

    for ticker in tickers:
        # If there is no valid PERMNO found, ask the user for a replacement ticker
        permno = pick_permno_for_ticker(db, ticker, start_date, max_missing, shrcd_list)
        while permno is None:
            print(f"No valid PERMNO found for ticker {ticker}.")
            ticker = input(
                "Please enter a replacement ticker: ").strip().upper()
            permno = pick_permno_for_ticker(db, ticker, start_date, max_missing, shrcd_list)
        permno_list.append(permno)
        valid_tickers.append(ticker)
        # date_of_delisting.append(delisting_date)

    print(f"Final tickers used: {valid_tickers}")
    print(f"Corresponding PERMNOs: {permno_list}")
    # print(f"Collecting data until first delisting date (if any): {max(date_of_delisting)}")

    return permno_list, valid_tickers


def pick_permno_for_ticker(db, ticker: str, start_date: pd.Timestamp, max_missing: int, \
                           shrcd_list: list) -> int | None:
    """
    Given a ticker and a date, pick the most appropriate PERMNO.

    Criteria (in order):
    1) Active on the start_date
    2) If multiple, pick the one with the namedt earliest

    Returns
    -------
    Selected PERMNO or None if no valid PERMNO found. Delisting dates.
    """
    q = f"""
    SELECT
        n.permno,
        n.namedt,
        n.nameendt,
        n.exchcd,
        n.shrcd
    FROM crsp.msenames AS n
    WHERE n.ticker = '{ticker}'
    ORDER BY n.namedt;
    """
    df = db.raw_sql(q)

    df['namedt'] = pd.to_datetime(df['namedt'])
    # df['nameendt'] = pd.to_datetime(df['nameendt'])
    # df['nameendt'] = df['nameendt'].fillna(pd.Timestamp.max)

    if df.empty:
        return None

    df = df[df['shrcd'].isin(shrcd_list)]

    if df.empty:
        # Nothing in the allowed share codes
        print(f"Ticker {ticker} has no active PERMNOs in common share classes as of {start_date.date()}.")
        return None
    
    df = collapse_name_ranges(df)

    # end_date = pd.Timestamp.today()
    # 1) Exact coverage: [namedt, nameendt] fully covers [start_date, end_date]
    # mask_full = (df['namedt'] <= start_date) & (df['nameendt'] >= end_date)
    active = df[df['namedt'] <= start_date]
    if active.empty:
        print(f"Ticker {ticker} has no active PERMNOs as of {start_date.date()}.")
        return None

    # iterate rows so we can grab shrcd for the same permno
    for _, row in active.iterrows():
        permno = int(row['permno'])

        df1 = get_delistings(db, [permno], start_date.strftime('%Y-%m-%d'))
        df2 = get_returns(db, [permno], start_date.strftime('%Y-%m-%d'), max_missing)

        if df1.empty and (df2 is not None):
            return permno
        elif not df1.empty:
            print('Delisting data found for PERMNO:', permno)
            print(df1)
        
    return None


    # if mask_full.any():
    #     if len(df[mask_full]) == 1:
    #         return int(df.loc[mask_full, 'permno'].iloc[0]), df.loc[mask_full, 'nameendt'].iloc[0]
    #     else:
    #         df = df[mask_full]
    #         # most current first
    #         df = df.sort_values(by='nameendt', ascending=False)
    #         # If still multiple, pick the one which is most current, i.e. with the latest nameendt
    #         print('Multiple candidates found. Picking the most current one.')
    #         print('Picked PERMNO:', df.iloc[0]['permno'])
    #         print('Candidates were:')
    #         print(df)
    #         return int(df.iloc[0]['permno']), df.iloc[0]['nameendt']

    # # 2) Missing months before the window (namedt > start_date)
    # after_start = df['namedt'] > start_date
    # months_before = np.where(
    #     after_start,
    #     (df['namedt'].dt.year - start_date.year) * 12
    #     + (df['namedt'].dt.month - start_date.month)
    #     # ceil if partial month
    #     + (df['namedt'].dt.day > start_date.day).astype(int),
    #     0
    # )

    # # 3) Missing months after the window (nameendt < end_date)
    # before_end = df['nameendt'] < end_date
    # months_after = np.where(
    #     before_end,
    #     (end_date.year - df['nameendt'].dt.year) * 12
    #     + (end_date.month - df['nameendt'].dt.month)
    #     # ceil if partial month
    #     + (end_date.day > df['nameendt'].dt.day).astype(int),
    #     0
    # )

    # total_missing_months = months_before + months_after

    # mask_tol = total_missing_months <= int(max_missing)
    # if mask_tol.any():
    #     if len(df[mask_tol]) == 1:
    #         return int(df.loc[mask_tol, 'permno'].iloc[0])
    #     else:
    #         # choose the row with the smallest total missing months (best match)
    #         idx = pd.Series(total_missing_months, index=df.index)[
    #             mask_tol].idxmin()
    #         print(
    #             'Multiple candidates found. Picking PERMNO with the smallest total missing months.')
    #         print('Picked PERMNO:', df.loc[idx, 'permno'])
    #         print('Candidates were:')
    #         print(df[mask_tol])
    #         return int(df.loc[idx, 'permno']), df.loc[idx, 'nameendt']



def collapse_name_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge rows per permno when the next segment starts the day after (or earlier than)
    the previous segment ends (i.e., overlapping or touching intervals).

    Keeps only: permno, namedt, nameendt
    """
    out = df.copy()
    out['namedt'] = pd.to_datetime(out['namedt'])
    out['nameendt'] = pd.to_datetime(
        out['nameendt']).fillna(pd.Timestamp('9999-12-31'))

    # sort within permno by start then end
    out = out.sort_values(['permno', 'namedt', 'nameendt'])

    # previous end per permno
    prev_end = out.groupby('permno')['nameendt'].shift()

    # start a new group when there is a gap (> 1 day). If you want to merge *only*
    # exactly-adjacent segments (next_start == prev_end + 1 day) and NOT overlaps,
    # see the note below.
    is_new_group = (prev_end.isna()) | (
        out['namedt'] > (prev_end + pd.Timedelta(days=1)))

    # running group id per permno
    out['grp'] = is_new_group.groupby(out['permno']).cumsum()

    # aggregate each group to a single interval
    merged = (
        out.groupby(['permno', 'grp'])
        .agg(namedt=('namedt', 'min'), nameendt=('nameendt', 'max'))
        .reset_index()
    )
    # print(merged)

    return merged[['permno', 'namedt', 'nameendt']]


def get_returns(df_conn, permno_list: list, start: str, max_missing=None) -> pd.DataFrame:
    """
    Wide monthly returns: index=date (Timestamp), columns=permno (int), values=ret (float).
    Missing/non-numeric RET are filled with 0.0 (policy choice).
    """
    if not permno_list:
        return pd.DataFrame()

    permno_list = [int(p) for p in permno_list]
    q = f"""
        SELECT permno, date, ret
        FROM crsp.msf
        WHERE permno IN ({','.join(map(str, permno_list))})
          AND date >= '{start}'
        ORDER BY permno, date;
    """
    d = df_conn.raw_sql(q)
    if d.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], name='date'))

    d['date'] = pd.to_datetime(d['date']) + MonthEnd(0)  # ensure month-end
    d['ret'] = pd.to_numeric(d['ret'], errors='coerce')

    n_missing = d['ret'].isna().sum()
    if n_missing:
        print(
            f"Warning: {n_missing} missing returns coerced to 0.0 across {d['permno'].nunique()} PERMNO(s).")
        d['ret'] = d['ret'].fillna(0.0)
    
    if max_missing is not None and n_missing > max_missing:
        print(f"Warning: Number of missing returns ({n_missing}) exceeds the allowed maximum ({max_missing}).")
        return None

    wide = (
        d.pivot(index='date', columns='permno', values='ret')
        .sort_index()
        .rename_axis(index='date', columns='permno')
        .astype(float)
    )
    return wide


def get_delistings(db, permno_list: list, start: str) -> pd.DataFrame:
    """
    Delisting returns aligned to month-end for merging with monthly CRSP returns.
    Returns long DataFrame with columns: permno (int), date (Timestamp month-end), dlret (float).
    """
    if not permno_list:
        return pd.DataFrame(columns=['permno', 'date', 'dlret'])

    permno_list = [int(p) for p in permno_list]
    q = f"""
        SELECT permno, dlstdt, dlret
        FROM crsp.msedelist
        WHERE permno IN ({','.join(map(str, permno_list))})
          AND dlstdt >= '{start}'
          AND dlret IS NOT NULL
        ORDER BY permno, dlstdt;
    """
    df = db.raw_sql(q)
    if df.empty:
        return pd.DataFrame(columns=['permno', 'date', 'dlret'])

    df['date'] = pd.to_datetime(df['dlstdt']) + MonthEnd(0)  # align to month-end
    df['dlret'] = pd.to_numeric(df['dlret'], errors='coerce')
    df = df[['permno', 'date', 'dlret']].dropna(subset=['dlret'])
    df['permno'] = df['permno'].astype(int)
    return df


def add_effective_returns(panel_wide: pd.DataFrame, delist_long: pd.DataFrame) -> pd.DataFrame:
    """
    Merge delisting returns into a wide monthly returns panel and compute effective returns:
        ret_eff = (1 + ret) * (1 + dlret) - 1
    Returns a wide DataFrame with the same shape (date x permno), values=ret_eff.
    - If a month has both ret and dlret, they compound.
    - If only one is present, the other is treated as 0 (no effect).
    """
    if panel_wide.empty:
        return panel_wide
    if delist_long.empty:
        return panel_wide

    panel_wide = panel_wide.rename_axis(index="date", columns="permno")

    # Wide -> long for merge
    long = (
        panel_wide.stack()
        .rename('ret')
        .reset_index()
        # not needed if columns already named 'permno'
        .rename(columns={'level_1': 'permno'})
    )
    # Ensure correct dtypes
    long['permno'] = long['permno'].astype(int)
    long['date'] = pd.to_datetime(long['date'])

    # Merge and compute effective returns
    merged = long.merge(delist_long, on=['permno', 'date'], how='left')
    merged['ret'] = pd.to_numeric(merged['ret'], errors='coerce').fillna(0.0)
    merged['dlret'] = pd.to_numeric(
        merged['dlret'], errors='coerce').fillna(0.0)
    merged['ret_eff'] = (1.0 + merged['ret']) * (1.0 + merged['dlret']) - 1.0

    # Long -> wide
    out = (
        merged.pivot(index='date', columns='permno', values='ret_eff')
        .sort_index()
        .rename_axis(index='date', columns='permno')
        .astype(float)
    )

    # Preserve original index/columns union (optional)
    out = out.reindex(index=panel_wide.index.union(out.index)).reindex(
        columns=panel_wide.columns, fill_value=out.reindex(columns=panel_wide.columns))
    # align to original dates/permnos
    out = out.reindex(index=panel_wide.index, columns=panel_wide.columns)
    return out


def weight_constraint_input():
    def ask_yn(prompt: str) -> str:
        while True:
            ans = input(prompt).strip().lower()
            if ans in {"y", "n"}:
                return ans
            print("Please answer 'y' or 'n'.")

    # ---- max (long) constraint ----
    set_max = ask_yn("Do you want to set maximum weight constraints? (y/n): ")
    max_weight = None
    if set_max == "y":
        while True:
            try:
                max_weight = float(
                    input("Enter the maximum long position (e.g., 0.1 for 10%): ").strip())
                if max_weight <= 0:
                    print("Please enter a positive number.")
                    continue
                break
            except ValueError:
                print("Invalid input. Please enter a positive number.")
        print(f"Maximum long position: {max_weight}")
    else:
        print("No maximum weight constraints set.")

    # ---- min (short) constraint ----
    set_min = ask_yn(
        "Do you want to set maximum short position constraints? (y/n): ")
    min_weight = None
    if set_min == "y":
        while True:
            try:
                min_weight = float(
                    input("Enter the maximum short position (e.g., -0.1 for -10%): ").strip())
                if min_weight > 0:
                    print("Please enter a negative number (or zero for no shorting).")
                    continue
                break
            except ValueError:
                print("Invalid input. Please enter a negative number.")
        print(f"Maximum short position: {min_weight}")
    else:
        print("No minimum weight constraints set.")

    return max_weight, min_weight


def window_size_input():
    # Ask the user for the rolling window size (in months) with error handling
    while True:
        try:
            window_size = int(
                input("Enter the rolling window size (in months) for out-of-sample testing: "))
            if window_size <= 0:
                print("Please enter a positive integer.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a positive integer.")
    print(f"Rolling window size: {window_size} months")
    return window_size

def risk_free_rate_input():
    # Ask the user for the annual risk-free rate (as a decimal) with error handling
    while True:
        try:
            rf_rate = float(
                input("Enter the annual risk-free rate (as a decimal, e.g., 0.03 for 3%): "))
            if rf_rate < 0:
                print("Please enter a non-negative number.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a non-negative number.")
    print(f"Annual risk-free rate: {rf_rate}")
    return rf_rate


def get_risk_free_rate_series(db, start, end) -> pd.DataFrame:
    rf = db.get_table(library='ff', table='factors_monthly')[['date', 'rf']]
    rf['date'] = pd.to_datetime(rf['date']) + MonthEnd(0)   # ensure month-end
    rf = rf.rename(columns={'rf': 'risk_free_rate'})
    rf = rf[(rf['date'] >= start) & (rf['date'] <= end)]
    rf.set_index('date', inplace=True)
    return rf


if __name__ == "__main__":
    data_acquisition()
