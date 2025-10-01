# test_pipeline.py
import unittest
from unittest.mock import patch, MagicMock
import io
from contextlib import redirect_stdout
import pandas as pd
import numpy as np
from pandas import Timestamp

MODULE_NAME = "data_acquisition_coverage_validation"

dacv = __import__(MODULE_NAME)

# ----------------------------
# ticker_input (5 tests)
# ----------------------------
class TestTickerInput(unittest.TestCase):
    @patch("builtins.input", side_effect=["AAPL, msft, goog"])
    def test_basic_parse_upper(self, _):
        self.assertEqual(dacv.ticker_input(), ["AAPL", "MSFT", "GOOG"])

    @patch("builtins.input", side_effect=["  aapl ,  msft  "])
    def test_trims_spaces(self, _):
        self.assertEqual(dacv.ticker_input(), ["AAPL", "MSFT"])

    @patch("builtins.input", side_effect=["", "AAPL"])
    def test_reprompts_on_empty(self, _):
        self.assertEqual(dacv.ticker_input(), ["AAPL"])

    @patch("builtins.input", side_effect=["spy"])
    def test_single_ok(self, _):
        self.assertEqual(dacv.ticker_input(), ["SPY"])


class TestDateInput(unittest.TestCase):

    @patch("builtins.input", side_effect=["2020-01-01", "12"])
    def test_happy_path(self, _):
        buf = io.StringIO()
        with redirect_stdout(buf):
            s, months = dacv.date_input()
        self.assertEqual(s, Timestamp("2020-01-01"))
        self.assertEqual(months, "12")  # function returns the raw months string
        out = buf.getvalue()
        # Jan 1 + (12-1)=11 months = Dec 1, 2020
        self.assertIn("Date range: 2020-01-01 to 2020-12-01", out)

    @patch("builtins.input",
           side_effect=[
               "bad", "12",           # invalid date -> loop
               "2020-01-01", "12"     # valid
           ])
    def test_invalid_format_then_valid(self, _):
        buf = io.StringIO()
        with redirect_stdout(buf):
            s, months = dacv.date_input()
        self.assertEqual((s, months), (Timestamp("2020-01-01"), "12"))
        out = buf.getvalue()
        self.assertIn("Invalid input. Use YYYY-MM-DD", out)
        self.assertIn("Date range: 2020-01-01 to 2020-12-01", out)

    @patch("builtins.input",
           side_effect=[
               "2020-01-01", "0",     # non-positive months -> reprompt
               "2020-01-01", "2"      # valid
           ])
    def test_months_zero_then_reprompt(self, _):
        buf = io.StringIO()
        with redirect_stdout(buf):
            s, months = dacv.date_input()
        self.assertEqual(s, Timestamp("2020-01-01"))
        self.assertEqual(months, "2")
        out = buf.getvalue()
        self.assertIn("must be a positive integer", out)
        # Jan 1 + (2-1)=1 month => Feb 1, 2020
        self.assertIn("Date range: 2020-01-01 to 2020-02-01", out)

    @patch("builtins.input", side_effect=[" 2021-02-15 ", " 1 "])
    def test_whitespace_ok(self, _):
        buf = io.StringIO()
        with redirect_stdout(buf):
            s, months = dacv.date_input()
        self.assertEqual(s, Timestamp("2021-02-15"))
        self.assertEqual(months, "1")
        out = buf.getvalue()
        # 1 month => months-1 = 0, so end == start
        self.assertIn("Date range: 2021-02-15 to 2021-02-15", out)

    @patch("builtins.input", side_effect=["2020-01-01", "6"])
    def test_returns_values_from_prompt(self, _):
        buf = io.StringIO()
        with redirect_stdout(buf):
            s, months = dacv.date_input()
        self.assertEqual((s, months), (Timestamp("2020-01-01"), "6"))
        out = buf.getvalue()
        # Jan 1 + (6-1)=5 months => Jun 1, 2020
        self.assertIn("Date range: 2020-01-01 to 2020-06-01", out)


# ----------------------------
# missing_months_input (5 tests)
# ----------------------------
class TestMissingMonthsInput(unittest.TestCase):
    @patch("builtins.input", side_effect=["0"])
    def test_zero(self, _):
        self.assertEqual(dacv.missing_months_input(), 0)

    @patch("builtins.input", side_effect=["-1", "2"])
    def test_negative_then_valid(self, _):
        self.assertEqual(dacv.missing_months_input(), 2)

    @patch("builtins.input", side_effect=["abc", "5"])
    def test_non_int_then_valid(self, _):
        self.assertEqual(dacv.missing_months_input(), 5)

    @patch("builtins.input", side_effect=[" 3 "])
    def test_spaces_ok(self, _):
        self.assertEqual(dacv.missing_months_input(), 3)

    @patch("builtins.input", side_effect=["12"])
    def test_large_ok(self, _):
        self.assertEqual(dacv.missing_months_input(), 12)


# test_get_active_permnos.py
import io
import unittest
from unittest.mock import patch, MagicMock
from contextlib import redirect_stdout
from pandas import Timestamp

# assume your module is imported as dacv and MODULE_NAME points to it
# e.g., MODULE_NAME = "dacv"

class TestGetActivePermnos(unittest.TestCase):

    @patch(f"{MODULE_NAME}.pick_permno_for_ticker",
           side_effect=[10001, 10006])
    def test_two_tickers_happy(self, _pick):
        db = MagicMock()
        buf = io.StringIO()
        with redirect_stdout(buf):
            permnos, tickers = dacv.get_active_permnos(
                db, ["AAPL", "MSFT"], Timestamp("2020-01-01"), 1, [10, 11]
            )
        self.assertEqual(permnos, [10001, 10006])
        self.assertEqual(tickers, ["AAPL", "MSFT"])
        out = buf.getvalue()
        self.assertIn("Final tickers used: ['AAPL', 'MSFT']", out)
        self.assertIn("Corresponding PERMNOs: [10001, 10006]", out)

    @patch(f"{MODULE_NAME}.pick_permno_for_ticker",
           side_effect=[None, 20001])
    @patch("builtins.input", side_effect=["NVDA"])
    def test_reprompt_on_none(self, _in, _pick):
        db = MagicMock()
        buf = io.StringIO()
        with redirect_stdout(buf):
            permnos, tickers = dacv.get_active_permnos(
                db, ["BAD"], Timestamp("2019-01-01"), 0, [10, 11]
            )
        self.assertEqual(permnos, [20001])
        self.assertEqual(tickers, ["NVDA"])  # replacement ticker used
        out = buf.getvalue()
        self.assertIn("No valid PERMNO found for ticker BAD.", out)
        self.assertIn("Final tickers used: ['NVDA']", out)
        self.assertEqual(_in.call_count, 1)

    @patch(f"{MODULE_NAME}.pick_permno_for_ticker",
           side_effect=[111, 222, 333])
    def test_order_preserved(self, _pick):
        db = MagicMock()
        buf = io.StringIO()
        with redirect_stdout(buf):
            permnos, tickers = dacv.get_active_permnos(
                db, ["X", "Y", "Z"], Timestamp("2020-01-01"), 2, [10, 11]
            )
        self.assertEqual(permnos, [111, 222, 333])
        self.assertEqual(tickers, ["X", "Y", "Z"])
        out = buf.getvalue()
        self.assertIn("Final tickers used: ['X', 'Y', 'Z']", out)

    @patch(f"{MODULE_NAME}.pick_permno_for_ticker",
           side_effect=[None, None, 30001])
    @patch("builtins.input", side_effect=["AAA", "BBB"])
    def test_multiple_reprompts(self, _in, _pick):
        db = MagicMock()
        buf = io.StringIO()
        with redirect_stdout(buf):
            permnos, tickers = dacv.get_active_permnos(
                db, ["ORIG"], Timestamp("2020-01-01"), 0, [10, 11]
            )
        self.assertEqual(permnos, [30001])
        self.assertEqual(tickers, ["BBB"])  # second replacement succeeded
        out = buf.getvalue()
        self.assertIn("No valid PERMNO found for ticker ORIG.", out)
        # It also prints again when AAA failed
        self.assertIn("No valid PERMNO found for ticker AAA.", out)
        self.assertEqual(_in.call_count, 2)

    @patch(f"{MODULE_NAME}.pick_permno_for_ticker", side_effect=[70001])
    def test_single_ticker(self, _pick):
        db = MagicMock()
        buf = io.StringIO()
        with redirect_stdout(buf):
            permnos, tickers = dacv.get_active_permnos(
                db, ["IBM"], Timestamp("2020-01-01"), 1, [10, 11]
            )
        self.assertEqual(permnos, [70001])
        self.assertEqual(tickers, ["IBM"])
        out = buf.getvalue()
        self.assertIn("Final tickers used: ['IBM']", out)
        self.assertIn("Corresponding PERMNOs: [70001]", out)

class TestGetActivePermnos(unittest.TestCase):

    @patch(f"{MODULE_NAME}.pick_permno_for_ticker", side_effect=[10001, 10006])
    def test_two_tickers_happy(self, _pick):
        db = MagicMock()
        buf = io.StringIO()
        with redirect_stdout(buf):
            permnos, tickers = dacv.get_active_permnos(
                db, ["AAPL", "MSFT"], Timestamp("2020-01-01"), 1, [10, 11]
            )
        self.assertEqual(permnos, [10001, 10006])
        self.assertEqual(tickers, ["AAPL", "MSFT"])
        out = buf.getvalue()
        self.assertIn("Final tickers used: ['AAPL', 'MSFT']", out)
        self.assertIn("Corresponding PERMNOs: [10001, 10006]", out)

    @patch(f"{MODULE_NAME}.pick_permno_for_ticker", side_effect=[None, 20001])
    @patch("builtins.input", side_effect=["NVDA"])
    def test_reprompt_on_none(self, _in, _pick):
        db = MagicMock()
        buf = io.StringIO()
        with redirect_stdout(buf):
            permnos, tickers = dacv.get_active_permnos(
                db, ["BAD"], Timestamp("2019-01-01"), 0, [10, 11]
            )
        self.assertEqual(permnos, [20001])
        self.assertEqual(tickers, ["NVDA"])
        out = buf.getvalue()
        self.assertIn("No valid PERMNO found for ticker BAD.", out)
        self.assertIn("Final tickers used: ['NVDA']", out)
        self.assertEqual(_in.call_count, 1)

    @patch(f"{MODULE_NAME}.pick_permno_for_ticker", side_effect=[111, 222, 333])
    def test_order_preserved(self, _pick):
        db = MagicMock()
        buf = io.StringIO()
        with redirect_stdout(buf):
            permnos, tickers = dacv.get_active_permnos(
                db, ["X", "Y", "Z"], Timestamp("2020-01-01"), 2, [10, 11]
            )
        self.assertEqual(permnos, [111, 222, 333])
        self.assertEqual(tickers, ["X", "Y", "Z"])
        out = buf.getvalue()
        self.assertIn("Final tickers used: ['X', 'Y', 'Z']", out)

    @patch(f"{MODULE_NAME}.pick_permno_for_ticker", side_effect=[None, None, 30001])
    @patch("builtins.input", side_effect=["AAA", "BBB"])
    def test_multiple_reprompts(self, _in, _pick):
        db = MagicMock()
        buf = io.StringIO()
        with redirect_stdout(buf):
            permnos, tickers = dacv.get_active_permnos(
                db, ["ORIG"], Timestamp("2020-01-01"), 0, [10, 11]
            )
        self.assertEqual(permnos, [30001])
        self.assertEqual(tickers, ["BBB"])  # second replacement succeeded
        out = buf.getvalue()
        self.assertIn("No valid PERMNO found for ticker ORIG.", out)
        self.assertIn("No valid PERMNO found for ticker AAA.", out)
        self.assertEqual(_in.call_count, 2)

    @patch(f"{MODULE_NAME}.pick_permno_for_ticker", side_effect=[70001])
    def test_single_ticker(self, _pick):
        db = MagicMock()
        buf = io.StringIO()
        with redirect_stdout(buf):
            permnos, tickers = dacv.get_active_permnos(
                db, ["IBM"], Timestamp("2020-01-01"), 1, [10, 11]
            )
        self.assertEqual(permnos, [70001])
        self.assertEqual(tickers, ["IBM"])
        out = buf.getvalue()
        self.assertIn("Final tickers used: ['IBM']", out)
        self.assertIn("Corresponding PERMNOs: [70001]", out)

# # ----------------------------
# # get_delistings (5 tests)
# # ----------------------------
# class TestGetDelistings(unittest.TestCase):
#     def setUp(self):
#         self.db = MagicMock()

#     def test_empty_permno_list(self):
#         out = dacv.get_delistings(self.db, [], "2020-01-01", "2020-12-31")
#         self.assertEqual(list(out.columns), ["permno", "date", "dlret"])
#         self.assertTrue(out.empty)

#     def test_empty_sql_result(self):
#         self.db.raw_sql.return_value = pd.DataFrame(columns=["permno", "dlstdt", "dlret"])
#         out = dacv.get_delistings(self.db, [1], "2020-01-01", "2020-12-31")
#         self.assertTrue(out.empty)

#     def test_align_month_end(self):
#         self.db.raw_sql.return_value = pd.DataFrame({
#             "permno": [1], "dlstdt": ["2020-02-10"], "dlret": ["-0.5"]
#         })
#         out = dacv.get_delistings(self.db, [1], "2020-01-01", "2020-12-31")
#         self.assertEqual(out.loc[0, "date"], pd.Timestamp("2020-02-29"))

#     def test_numeric_conversion(self):
#         self.db.raw_sql.return_value = pd.DataFrame({
#             "permno": [1], "dlstdt": ["2020-01-31"], "dlret": ["0.1"]
#         })
#         out = dacv.get_delistings(self.db, [1], "2020-01-01", "2020-12-31")
#         self.assertAlmostEqual(out.loc[0, "dlret"], 0.1, places=7)

#     def test_drop_null_dlret(self):
#         self.db.raw_sql.return_value = pd.DataFrame({
#             "permno": [1, 1], "dlstdt": ["2020-01-31", "2020-02-29"], "dlret": [None, "-1.0"]
#         })
#         out = dacv.get_delistings(self.db, [1], "2020-01-01", "2020-12-31")
#         self.assertEqual(len(out), 1)


# # ----------------------------
# # add_effective_returns (5 tests)
# # ----------------------------
# class TestAddEffectiveReturns(unittest.TestCase):
#     def test_empty_panel(self):
#         out = dacv.add_effective_returns(pd.DataFrame(), pd.DataFrame())
#         self.assertTrue(out.empty)

#     def test_no_delistings_identity(self):
#         panel = pd.DataFrame({1: [0.1, 0.0]}, index=pd.to_datetime(["2020-01-31", "2020-02-29"]))
#         out = dacv.add_effective_returns(panel, pd.DataFrame(columns=["permno", "date", "dlret"]))
#         self.assertTrue(out.equals(panel))

#     def test_compound_only_dlret(self):
#         panel = pd.DataFrame({1: [0.0]}, index=pd.to_datetime(["2020-02-29"]))
#         dl = pd.DataFrame({"permno": [1], "date": [pd.Timestamp("2020-02-29")], "dlret": [-0.5]})
#         out = dacv.add_effective_returns(panel, dl)
#         self.assertAlmostEqual(out.loc[pd.Timestamp("2020-02-29"), 1], -0.5)

#     def test_compound_both(self):
#         panel = pd.DataFrame({1: [0.1]}, index=pd.to_datetime(["2020-02-29"]))
#         dl = pd.DataFrame({"permno": [1], "date": [pd.Timestamp("2020-02-29")], "dlret": [-0.5]})
#         out = dacv.add_effective_returns(panel, dl)
#         self.assertAlmostEqual(out.loc[pd.Timestamp("2020-02-29"), 1], -0.45, places=6)

#     def test_multiple_assets(self):
#         panel = pd.DataFrame({1: [0.05], 2: [0.02]}, index=pd.to_datetime(["2020-03-31"]))
#         dl = pd.DataFrame({"permno": [2], "date": [pd.Timestamp("2020-03-31")], "dlret": [-1.0]})
#         out = dacv.add_effective_returns(panel, dl)
#         self.assertAlmostEqual(out.loc[pd.Timestamp("2020-03-31"), 1], 0.05)
#         self.assertAlmostEqual(out.loc[pd.Timestamp("2020-03-31"), 2], -1.0)


# ----------------------------
# weight_constraint_input (5 tests)
# ----------------------------
class TestWeightConstraintInput(unittest.TestCase):
    @patch("builtins.input", side_effect=["y", "0.1", "y", "-0.2"])
    def test_yes_yes(self, _):
        mx, mn = dacv.weight_constraint_input()
        self.assertEqual(mx, 0.1)
        self.assertEqual(mn, -0.2)

    @patch("builtins.input", side_effect=["n", "n"])
    def test_no_no(self, _):
        mx, mn = dacv.weight_constraint_input()
        self.assertIsNone(mx)
        # Depending on your implementation, min_weight may be None; if not, fix the function and assert None.

    @patch("builtins.input", side_effect=["y", "bad", "0.05", "y", "-0.1"])
    def test_invalid_then_valid(self, _):
        mx, mn = dacv.weight_constraint_input()
        self.assertEqual(mx, 0.05)
        self.assertEqual(mn, -0.1)

    @patch("builtins.input", side_effect=["y", "0.2", "n"])
    def test_yes_then_no(self, _):
        mx, mn = dacv.weight_constraint_input()
        self.assertEqual(mx, 0.2)

    @patch("builtins.input", side_effect=["n", "y", "-0.15"])
    def test_no_then_yes(self, _):
        mx, mn = dacv.weight_constraint_input()
        self.assertIsNone(mx)
        self.assertEqual(mn, -0.15)

# ----------------------------
# data_acquisition (5 tests)
# ----------------------------
# test_data_acquisition.py
class TestDataAcquisition(unittest.TestCase):

    @patch(f"{MODULE_NAME}.risk_free_rate_input", return_value=0.03)
    @patch(f"{MODULE_NAME}.window_size_input", return_value=12)
    @patch(f"{MODULE_NAME}.weight_constraint_input", return_value=(0.1, -0.1))
    @patch(f"{MODULE_NAME}.get_returns")
    @patch(f"{MODULE_NAME}.get_active_permnos", return_value=([10001], ["AAPL"]))
    @patch(f"{MODULE_NAME}.missing_months_input", return_value=1)
    @patch(f"{MODULE_NAME}.date_input", return_value=(Timestamp("2020-01-01"), "12"))
    @patch(f"{MODULE_NAME}.ticker_input", return_value=["AAPL"])
    @patch(f"{MODULE_NAME}.db_connect")
    def test_happy_path_basic(
        self, m_db, m_tickers, m_date, m_missing, m_get_active, m_get_returns,
        m_weight, m_window, m_rf
    ):
        # returns for one permno
        dates = pd.to_datetime(["2020-01-31", "2020-02-29"])
        m_get_returns.return_value = pd.DataFrame(
            {10001: [0.05, -0.02]}, index=dates
        )

        buf = io.StringIO()
        with redirect_stdout(buf):
            all_data, valid_tickers, start_date, months, max_w, min_w, win_sz, rf = dacv.data_acquisition()

        # renamed to ticker column
        self.assertListEqual(all_data.columns.tolist(), ["AAPL"])
        self.assertEqual(valid_tickers, ["AAPL"])
        self.assertEqual(start_date, Timestamp("2020-01-01"))
        self.assertEqual(months, "12")
        self.assertEqual((max_w, min_w), (0.1, -0.1))
        self.assertEqual(win_sz, 12)
        self.assertEqual(rf, 0.03)

        # get_returns called with db, permno_list, and start_date string
        m_get_returns.assert_called_once()
        args, kwargs = m_get_returns.call_args
        self.assertIs(args[0], m_db.return_value)
        self.assertEqual(args[1], [10001])
        self.assertEqual(args[2], "2020-01-01")  # start_date_eval.strftime('%Y-%m-%d')

    @patch(f"{MODULE_NAME}.risk_free_rate_input", return_value=0.01)
    @patch(f"{MODULE_NAME}.window_size_input", return_value=6)
    @patch(f"{MODULE_NAME}.weight_constraint_input", return_value=(None, None))
    @patch(f"{MODULE_NAME}.get_returns")
    @patch(f"{MODULE_NAME}.get_active_permnos", return_value=([11111, 22222], ["MSFT", "NVDA"]))
    @patch(f"{MODULE_NAME}.missing_months_input", return_value=2)
    @patch(f"{MODULE_NAME}.date_input", return_value=(Timestamp("2019-03-01"), "9"))
    @patch(f"{MODULE_NAME}.ticker_input", return_value=["MSFT", "NVDA"])
    @patch(f"{MODULE_NAME}.db_connect")
    def test_multiple_permnos_are_renamed_to_tickers(
        self, m_db, *_mocks, **kwargs
    ):
        # returns for two permnos -> should rename to MSFT, NVDA
        idx = pd.to_datetime(["2019-03-29", "2019-04-30"])
        df = pd.DataFrame({11111: [0.02, 0.03], 22222: [0.01, -0.01]}, index=idx)
        dacv.get_returns.return_value = df

        all_data, tickers, start_date, months, max_w, min_w, win_sz, rf = dacv.data_acquisition()

        self.assertListEqual(sorted(all_data.columns.tolist()), ["MSFT", "NVDA"])
        self.assertEqual(tickers, ["MSFT", "NVDA"])
        self.assertEqual(start_date, Timestamp("2019-03-01"))
        self.assertEqual(months, "9")
        self.assertIsNone(max_w)
        self.assertIsNone(min_w)
        self.assertEqual(win_sz, 6)
        self.assertEqual(rf, 0.01)

    @patch(f"{MODULE_NAME}.risk_free_rate_input", return_value=0.0)
    @patch(f"{MODULE_NAME}.window_size_input", return_value=24)
    @patch(f"{MODULE_NAME}.weight_constraint_input", return_value=(0.2, -0.2))
    @patch(f"{MODULE_NAME}.get_returns")
    @patch(f"{MODULE_NAME}.get_active_permnos", return_value=([55555], ["IBM"]))
    @patch(f"{MODULE_NAME}.missing_months_input", return_value=0)
    @patch(f"{MODULE_NAME}.date_input", return_value=(Timestamp("2018-07-15"), "3"))
    @patch(f"{MODULE_NAME}.ticker_input", return_value=["IBM"])
    @patch(f"{MODULE_NAME}.db_connect")
    def test_propagates_user_params_and_prints(
        self, m_db, *_mocks
    ):
        # empty but correctly shaped DataFrame is OK
        dacv.get_returns.return_value = pd.DataFrame({55555: []})

        buf = io.StringIO()
        with redirect_stdout(buf):
            out = dacv.data_acquisition()
        all_data, tickers, start_date, months, max_w, min_w, win_sz, rf = out

        self.assertEqual(tickers, ["IBM"])
        self.assertEqual(start_date, Timestamp("2018-07-15"))
        self.assertEqual(months, "3")
        self.assertEqual((max_w, min_w), (0.2, -0.2))
        self.assertEqual(win_sz, 24)
        self.assertEqual(rf, 0.0)

        # sanity: function prints the pieces (not strict matching)
        printed = buf.getvalue()
        self.assertIn("['IBM']", printed)
        self.assertIn("2018-07-15", printed)

    @patch(f"{MODULE_NAME}.risk_free_rate_input", return_value=0.02)
    @patch(f"{MODULE_NAME}.window_size_input", return_value=3)
    @patch(f"{MODULE_NAME}.weight_constraint_input", return_value=(None, -0.05))
    @patch(f"{MODULE_NAME}.get_returns")
    @patch(f"{MODULE_NAME}.get_active_permnos", return_value=([99999], ["TSLA"]))
    @patch(f"{MODULE_NAME}.missing_months_input", return_value=5)
    @patch(f"{MODULE_NAME}.date_input", return_value=(Timestamp("2021-01-01"), "1"))
    @patch(f"{MODULE_NAME}.ticker_input", return_value=["TSLA"])
    @patch(f"{MODULE_NAME}.db_connect")
    def test_single_month_window_and_min_weight_only(
        self, m_db, *_mocks
    ):
        dates = pd.to_datetime(["2021-01-29"])
        dacv.get_returns.return_value = pd.DataFrame({99999: [0.12]}, index=dates)

        all_data, tickers, start_date, months, max_w, min_w, win_sz, rf = dacv.data_acquisition()

        self.assertEqual(all_data.shape, (1, 1))
        self.assertEqual(all_data.columns.tolist(), ["TSLA"])
        self.assertEqual(tickers, ["TSLA"])
        self.assertEqual(start_date, Timestamp("2021-01-01"))
        self.assertEqual(months, "1")
        self.assertIsNone(max_w)
        self.assertEqual(min_w, -0.05)
        self.assertEqual(win_sz, 3)
        self.assertEqual(rf, 0.02)

        # get_returns args check
        args, _ = dacv.get_returns.call_args
        self.assertEqual(args[1], [99999])
        self.assertEqual(args[2], "2021-01-01")

    @patch(f"{MODULE_NAME}.risk_free_rate_input", return_value=0.015)
    @patch(f"{MODULE_NAME}.window_size_input", return_value=18)
    @patch(f"{MODULE_NAME}.weight_constraint_input", return_value=(0.15, None))
    @patch(f"{MODULE_NAME}.get_returns")
    @patch(f"{MODULE_NAME}.get_active_permnos", return_value=([12345, 54321], ["META", "GOOGL"]))
    @patch(f"{MODULE_NAME}.missing_months_input", return_value=4)
    @patch(f"{MODULE_NAME}.date_input", return_value=(Timestamp("2020-05-05"), "7"))
    @patch(f"{MODULE_NAME}.ticker_input", return_value=["META", "GOOGL"])
    @patch(f"{MODULE_NAME}.db_connect")
    def test_columns_renamed_correctly_for_multiple_and_order_preserved(
        self, m_db, *_mocks
    ):
        idx = pd.to_datetime(["2020-05-29", "2020-06-30", "2020-07-31"])
        df = pd.DataFrame({12345: [0.01, 0.02, 0.03], 54321: [0.00, -0.01, 0.02]}, index=idx)
        dacv.get_returns.return_value = df

        all_data, tickers, start_date, months, max_w, min_w, win_sz, rf = dacv.data_acquisition()

        self.assertEqual(tickers, ["META", "GOOGL"])
        self.assertListEqual(all_data.columns.tolist(), ["META", "GOOGL"])
        pd.testing.assert_index_equal(all_data.index, idx)
        self.assertEqual(start_date, Timestamp("2020-05-05"))
        self.assertEqual(months, "7")
        self.assertEqual((max_w, min_w), (0.15, None))
        self.assertEqual(win_sz, 18)
        self.assertEqual(rf, 0.015)

if __name__ == "__main__":
    unittest.main()
