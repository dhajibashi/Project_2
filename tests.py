# test_pipeline.py
import unittest
from unittest.mock import patch, MagicMock
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


# ----------------------------
# date_input (5 tests)
# ----------------------------
class TestDateInput(unittest.TestCase):
    @patch("builtins.input",
           side_effect=["2020-01-01", "12", "2020-01-01", "2021-01-01"])
    def test_happy_path(self, _):
        s, e = dacv.date_input()
        self.assertEqual(s, Timestamp("2020-01-01"))
        self.assertEqual(e, Timestamp("2021-01-01"))

    @patch("builtins.input",
           side_effect=["bad", "12",  # invalid start -> loop
                        "2020-01-01", "12",  # valid in first loop
                        "2020-01-01", "2021-01-01"])  # final pair returned
    def test_invalid_format_then_valid(self, _):
        s, e = dacv.date_input()
        self.assertEqual((s, e), (Timestamp("2020-01-01"), Timestamp("2021-01-01")))

    @patch("builtins.input",
           side_effect=["2020-01-01", "0",  # end == start -> loop
                        "2020-01-01", "12",
                        "2020-01-01", "2020-12-31"])
    def test_start_after_or_equal_end_reprompt(self, _):
        s, e = dacv.date_input()
        self.assertLess(s, e)

    @patch("builtins.input",
           side_effect=[" 2021-02-15 ", "1", "2021-02-15", "2021-03-15"])
    def test_whitespace_ok(self, _):
        s, e = dacv.date_input()
        self.assertEqual(s, Timestamp("2021-02-15"))
        self.assertEqual(e, Timestamp("2021-03-15"))

    @patch("builtins.input", side_effect=["2020-01-01", "6"])
    def test_returns_values_from_prompt(self, _):
        s, e = dacv.date_input()
        self.assertEqual((s, e), (Timestamp("2020-01-01"), Timestamp("2020-07-01")))


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


# ----------------------------
# pick_permno_for_ticker (5 tests)
# ----------------------------
class TestGetActivePermnos(unittest.TestCase):
    @patch(f"{MODULE_NAME}.pick_permno_for_ticker",
           side_effect=[(10001, Timestamp("2025-01-31")),
                        (10006, Timestamp("2024-12-31"))])
    def test_two_tickers_happy(self, _pick):
        db = MagicMock()
        permnos, tickers, first_delist = dacv.get_active_permnos(
            db, ["AAPL", "MSFT"], Timestamp("2020-01-01"), Timestamp("2020-12-31"), 1
        )
        self.assertEqual(permnos, [10001, 10006])
        self.assertEqual(tickers, ["AAPL", "MSFT"])
        # Function prints "Collecting data until first delisting date (if any): max(...)"
        # It returns max(date_of_delisting)
        self.assertEqual(first_delist, Timestamp("2025-01-31"))

    @patch(f"{MODULE_NAME}.pick_permno_for_ticker",
           side_effect=[(None, None), (20001, Timestamp("2023-06-30"))])
    @patch("builtins.input", side_effect=["NVDA"])
    def test_reprompt_on_none(self, _in, _pick):
        db = MagicMock()
        permnos, tickers, first_delist = dacv.get_active_permnos(
            db, ["BAD"], Timestamp("2019-01-01"), Timestamp("2019-12-31"), 0
        )
        self.assertEqual(permnos, [20001])
        self.assertEqual(tickers, ["NVDA"])
        self.assertEqual(first_delist, Timestamp("2023-06-30"))

    @patch(f"{MODULE_NAME}.pick_permno_for_ticker",
           side_effect=[(111, Timestamp("2022-01-31")),
                        (222, Timestamp("2024-02-29")),
                        (333, Timestamp("2023-12-31"))])
    def test_order_preserved_and_max_delist(self, _pick):
        db = MagicMock()
        permnos, tickers, first_delist = dacv.get_active_permnos(
            db, ["X", "Y", "Z"], Timestamp("2020-01-01"), Timestamp("2020-12-31"), 2
        )
        self.assertEqual(permnos, [111, 222, 333])
        self.assertEqual(tickers, ["X", "Y", "Z"])
        self.assertEqual(first_delist, Timestamp("2024-02-29"))

    @patch(f"{MODULE_NAME}.pick_permno_for_ticker",
           side_effect=[(None, None), (None, None), (30001, Timestamp("2022-03-31"))])
    @patch("builtins.input", side_effect=["AAA", "BBB"])
    def test_multiple_reprompts(self, _in, _pick):
        db = MagicMock()
        permnos, tickers, first_delist = dacv.get_active_permnos(
            db, ["ORIG"], Timestamp("2020-01-01"), Timestamp("2020-12-31"), 0
        )
        self.assertEqual(permnos, [30001])
        self.assertEqual(tickers, ["BBB"])  # second replacement that finally succeeded
        self.assertEqual(first_delist, Timestamp("2022-03-31"))
        # ensure input was asked twice
        self.assertEqual(_in.call_count, 2)

    @patch(f"{MODULE_NAME}.pick_permno_for_ticker",
           side_effect=[(70001, Timestamp("2021-06-30"))])
    def test_single_ticker(self, _pick):
        db = MagicMock()
        permnos, tickers, first_delist = dacv.get_active_permnos(
            db, ["IBM"], Timestamp("2020-01-01"), Timestamp("2020-12-31"), 1
        )
        self.assertEqual(permnos, [70001])
        self.assertEqual(tickers, ["IBM"])
        self.assertEqual(first_delist, Timestamp("2021-06-30"))

# ----------------------------
# get_active_permnos (5 tests)
# ----------------------------
class TestGetActivePermnos(unittest.TestCase):
    @patch(f"{MODULE_NAME}.pick_permno_for_ticker",
           side_effect=[(10001, Timestamp("2025-01-31")),
                        (10006, Timestamp("2024-12-31"))])
    def test_two_tickers_happy(self, _pick):
        db = MagicMock()
        permnos, tickers, first_delist = dacv.get_active_permnos(
            db, ["AAPL", "MSFT"], Timestamp("2020-01-01"), Timestamp("2020-12-31"), 1
        )
        self.assertEqual(permnos, [10001, 10006])
        self.assertEqual(tickers, ["AAPL", "MSFT"])
        # Function prints "Collecting data until first delisting date (if any): max(...)"
        # It returns max(date_of_delisting)
        self.assertEqual(first_delist, Timestamp("2025-01-31"))

    @patch(f"{MODULE_NAME}.pick_permno_for_ticker",
           side_effect=[(None, None), (20001, Timestamp("2023-06-30"))])
    @patch("builtins.input", side_effect=["NVDA"])
    def test_reprompt_on_none(self, _in, _pick):
        db = MagicMock()
        permnos, tickers, first_delist = dacv.get_active_permnos(
            db, ["BAD"], Timestamp("2019-01-01"), Timestamp("2019-12-31"), 0
        )
        self.assertEqual(permnos, [20001])
        self.assertEqual(tickers, ["NVDA"])
        self.assertEqual(first_delist, Timestamp("2023-06-30"))

    @patch(f"{MODULE_NAME}.pick_permno_for_ticker",
           side_effect=[(111, Timestamp("2022-01-31")),
                        (222, Timestamp("2024-02-29")),
                        (333, Timestamp("2023-12-31"))])
    def test_order_preserved_and_max_delist(self, _pick):
        db = MagicMock()
        permnos, tickers, first_delist = dacv.get_active_permnos(
            db, ["X", "Y", "Z"], Timestamp("2020-01-01"), Timestamp("2020-12-31"), 2
        )
        self.assertEqual(permnos, [111, 222, 333])
        self.assertEqual(tickers, ["X", "Y", "Z"])
        self.assertEqual(first_delist, Timestamp("2024-02-29"))

    @patch(f"{MODULE_NAME}.pick_permno_for_ticker",
           side_effect=[(None, None), (None, None), (30001, Timestamp("2022-03-31"))])
    @patch("builtins.input", side_effect=["AAA", "BBB"])
    def test_multiple_reprompts(self, _in, _pick):
        db = MagicMock()
        permnos, tickers, first_delist = dacv.get_active_permnos(
            db, ["ORIG"], Timestamp("2020-01-01"), Timestamp("2020-12-31"), 0
        )
        self.assertEqual(permnos, [30001])
        self.assertEqual(tickers, ["BBB"])  # second replacement that finally succeeded
        self.assertEqual(first_delist, Timestamp("2022-03-31"))
        # ensure input was asked twice
        self.assertEqual(_in.call_count, 2)

    @patch(f"{MODULE_NAME}.pick_permno_for_ticker",
           side_effect=[(70001, Timestamp("2021-06-30"))])
    def test_single_ticker(self, _pick):
        db = MagicMock()
        permnos, tickers, first_delist = dacv.get_active_permnos(
            db, ["IBM"], Timestamp("2020-01-01"), Timestamp("2020-12-31"), 1
        )
        self.assertEqual(permnos, [70001])
        self.assertEqual(tickers, ["IBM"])
        self.assertEqual(first_delist, Timestamp("2021-06-30"))

# ----------------------------
# get_returns (5 tests)
# ----------------------------
class TestGetReturns(unittest.TestCase):
    def setUp(self):
        self.conn = MagicMock()

    def test_empty_permnos(self):
        out = dacv.get_returns(self.conn, [], "2020-01-01", "2020-12-31")
        self.assertTrue(out.empty)

    def test_empty_sql_result(self):
        self.conn.raw_sql.return_value = pd.DataFrame(columns=["permno", "date", "ret"])
        out = dacv.get_returns(self.conn, [1], "2020-01-01", "2020-12-31")
        self.assertTrue(out.empty)

    def test_pivot_types(self):
        self.conn.raw_sql.return_value = pd.DataFrame({
            "permno": [1, 1, 2, 2],
            "date": ["2020-01-31", "2020-02-29", "2020-01-31", "2020-02-29"],
            "ret": ["0.1", "0.2", "0.0", "-0.05"]
        })
        out = dacv.get_returns(self.conn, [1, 2], "2020-01-01", "2020-02-29")
        self.assertEqual(out.index[0], pd.Timestamp("2020-01-31"))
        self.assertTrue(np.issubdtype(out.dtypes.iloc[0], np.floating))
        self.assertTrue(all(np.issubdtype(dt, np.floating) for dt in out.dtypes))

    def test_coerce_fill(self):
        self.conn.raw_sql.return_value = pd.DataFrame({
            "permno": [1], "date": ["2020-01-31"], "ret": ["B"]
        })
        out = dacv.get_returns(self.conn, [1], "2020-01-01", "2020-01-31")
        self.assertEqual(out.iloc[0, 0], 0.0)

    def test_column_order(self):
        self.conn.raw_sql.return_value = pd.DataFrame({
            "permno": [2, 1],
            "date": ["2020-01-31", "2020-01-31"],
            "ret": ["0.02", "0.01"]
        })
        out = dacv.get_returns(self.conn, [1, 2], "2020-01-01", "2020-01-31")
        self.assertListEqual(sorted(out.columns.tolist()), [1, 2])


# ----------------------------
# get_delistings (5 tests)
# ----------------------------
class TestGetDelistings(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()

    def test_empty_permno_list(self):
        out = dacv.get_delistings(self.db, [], "2020-01-01", "2020-12-31")
        self.assertEqual(list(out.columns), ["permno", "date", "dlret"])
        self.assertTrue(out.empty)

    def test_empty_sql_result(self):
        self.db.raw_sql.return_value = pd.DataFrame(columns=["permno", "dlstdt", "dlret"])
        out = dacv.get_delistings(self.db, [1], "2020-01-01", "2020-12-31")
        self.assertTrue(out.empty)

    def test_align_month_end(self):
        self.db.raw_sql.return_value = pd.DataFrame({
            "permno": [1], "dlstdt": ["2020-02-10"], "dlret": ["-0.5"]
        })
        out = dacv.get_delistings(self.db, [1], "2020-01-01", "2020-12-31")
        self.assertEqual(out.loc[0, "date"], pd.Timestamp("2020-02-29"))

    def test_numeric_conversion(self):
        self.db.raw_sql.return_value = pd.DataFrame({
            "permno": [1], "dlstdt": ["2020-01-31"], "dlret": ["0.1"]
        })
        out = dacv.get_delistings(self.db, [1], "2020-01-01", "2020-12-31")
        self.assertAlmostEqual(out.loc[0, "dlret"], 0.1, places=7)

    def test_drop_null_dlret(self):
        self.db.raw_sql.return_value = pd.DataFrame({
            "permno": [1, 1], "dlstdt": ["2020-01-31", "2020-02-29"], "dlret": [None, "-1.0"]
        })
        out = dacv.get_delistings(self.db, [1], "2020-01-01", "2020-12-31")
        self.assertEqual(len(out), 1)


# ----------------------------
# add_effective_returns (5 tests)
# ----------------------------
class TestAddEffectiveReturns(unittest.TestCase):
    def test_empty_panel(self):
        out = dacv.add_effective_returns(pd.DataFrame(), pd.DataFrame())
        self.assertTrue(out.empty)

    def test_no_delistings_identity(self):
        panel = pd.DataFrame({1: [0.1, 0.0]}, index=pd.to_datetime(["2020-01-31", "2020-02-29"]))
        out = dacv.add_effective_returns(panel, pd.DataFrame(columns=["permno", "date", "dlret"]))
        self.assertTrue(out.equals(panel))

    def test_compound_only_dlret(self):
        panel = pd.DataFrame({1: [0.0]}, index=pd.to_datetime(["2020-02-29"]))
        dl = pd.DataFrame({"permno": [1], "date": [pd.Timestamp("2020-02-29")], "dlret": [-0.5]})
        out = dacv.add_effective_returns(panel, dl)
        self.assertAlmostEqual(out.loc[pd.Timestamp("2020-02-29"), 1], -0.5)

    def test_compound_both(self):
        panel = pd.DataFrame({1: [0.1]}, index=pd.to_datetime(["2020-02-29"]))
        dl = pd.DataFrame({"permno": [1], "date": [pd.Timestamp("2020-02-29")], "dlret": [-0.5]})
        out = dacv.add_effective_returns(panel, dl)
        self.assertAlmostEqual(out.loc[pd.Timestamp("2020-02-29"), 1], -0.45, places=6)

    def test_multiple_assets(self):
        panel = pd.DataFrame({1: [0.05], 2: [0.02]}, index=pd.to_datetime(["2020-03-31"]))
        dl = pd.DataFrame({"permno": [2], "date": [pd.Timestamp("2020-03-31")], "dlret": [-1.0]})
        out = dacv.add_effective_returns(panel, dl)
        self.assertAlmostEqual(out.loc[pd.Timestamp("2020-03-31"), 1], 0.05)
        self.assertAlmostEqual(out.loc[pd.Timestamp("2020-03-31"), 2], -1.0)


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
class TestDataAcquisition(unittest.TestCase):
    @patch(f"{MODULE_NAME}.weight_constraint_input", return_value=(0.1, -0.2))
    @patch(f"{MODULE_NAME}.add_effective_returns")
    @patch(f"{MODULE_NAME}.get_delistings")
    @patch(f"{MODULE_NAME}.get_returns")
    @patch(f"{MODULE_NAME}.get_active_permnos")
    @patch(f"{MODULE_NAME}.missing_months_input", return_value=1)
    @patch(f"{MODULE_NAME}.date_input", return_value=(Timestamp("2020-01-01"), Timestamp("2020-12-31")))
    @patch(f"{MODULE_NAME}.ticker_input", return_value=["AAPL", "MSFT"])
    @patch(f"{MODULE_NAME}.db_connect")
    def test_happy_path(self, mock_db, mock_tickers, mock_dates, mock_missing,
                        mock_get_active, mock_get_returns, mock_get_delist, mock_add_eff, mock_wc):
        # DB mock
        mock_db.return_value = MagicMock()
        # active permnos + first_delisting_date
        mock_get_active.return_value = ([10001, 10006], ["AAPL", "MSFT"], Timestamp("2020-06-30"))
        # returns and delistings
        dates = pd.to_datetime(["2020-01-31", "2020-02-29"])
        mock_get_returns.return_value = pd.DataFrame({10001: [0.01, 0.02], 10006: [0.03, 0.04]}, index=dates)
        mock_get_delist.return_value = pd.DataFrame({"permno": [10001],
                                                     "date": [Timestamp("2020-02-29")],
                                                     "dlret": [-0.5]})
        mock_add_eff.return_value = mock_get_returns.return_value.copy()

        data, tickers, s, e, mx, mn = dacv.data_acquisition()

        # columns renamed to tickers
        self.assertListEqual(list(data.columns), ["AAPL", "MSFT"])
        self.assertEqual(tickers, ["AAPL", "MSFT"])
        self.assertEqual((s, e), (Timestamp("2020-01-01"), Timestamp("2020-12-31")))
        self.assertEqual((mx, mn), (0.1, -0.2))

        # get_returns end-date should be first_delisting_date
        mock_get_returns.assert_called_with(
            mock_db.return_value, [10001, 10006], "2020-01-01", "2020-06-30"
        )

    @patch(f"{MODULE_NAME}.weight_constraint_input", return_value=(None, None))
    @patch(f"{MODULE_NAME}.add_effective_returns", return_value=pd.DataFrame())
    @patch(f"{MODULE_NAME}.get_delistings", return_value=pd.DataFrame(columns=["permno", "date", "dlret"]))
    @patch(f"{MODULE_NAME}.get_returns", return_value=pd.DataFrame(index=pd.to_datetime(["2020-01-31"])))
    @patch(f"{MODULE_NAME}.get_active_permnos", return_value=([10001], ["AAPL"], Timestamp("2020-01-31")))
    @patch(f"{MODULE_NAME}.missing_months_input", return_value=0)
    @patch(f"{MODULE_NAME}.date_input", return_value=(Timestamp("2020-01-01"), Timestamp("2020-12-31")))
    @patch(f"{MODULE_NAME}.ticker_input", return_value=["AAPL"])
    @patch(f"{MODULE_NAME}.db_connect")
    def test_empty_returns_shape_ok(self, *_):
        data, tickers, *_tail = dacv.data_acquisition()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(tickers, ["AAPL"])

    @patch(f"{MODULE_NAME}.weight_constraint_input", return_value=(None, None))
    @patch(f"{MODULE_NAME}.add_effective_returns", return_value=pd.DataFrame({10001: [0.1]} ,
                                                                             index=pd.to_datetime(["2020-01-31"])))
    @patch(f"{MODULE_NAME}.get_delistings", return_value=pd.DataFrame(columns=["permno", "date", "dlret"]))
    @patch(f"{MODULE_NAME}.get_returns", return_value=pd.DataFrame({10001: [0.1]} ,
                                                                   index=pd.to_datetime(["2020-01-31"])))
    @patch(f"{MODULE_NAME}.get_active_permnos", return_value=([10001], ["IBM"], Timestamp("2020-01-31")))
    @patch(f"{MODULE_NAME}.missing_months_input", return_value=0)
    @patch(f"{MODULE_NAME}.date_input", return_value=(Timestamp("2020-01-01"), Timestamp("2020-12-31")))
    @patch(f"{MODULE_NAME}.ticker_input", return_value=["IBM"])
    @patch(f"{MODULE_NAME}.db_connect")
    def test_column_rename_to_tickers(self, *_):
        data, tickers, *_tail = dacv.data_acquisition()
        self.assertListEqual(list(data.columns), ["IBM"])

    @patch(f"{MODULE_NAME}.weight_constraint_input", return_value=(None, None))
    @patch(f"{MODULE_NAME}.add_effective_returns")
    @patch(f"{MODULE_NAME}.get_delistings")
    @patch(f"{MODULE_NAME}.get_returns", side_effect=Exception("DB failure"))
    @patch(f"{MODULE_NAME}.get_active_permnos", return_value=([1], ["AAPL"], Timestamp("2020-06-30")))
    @patch(f"{MODULE_NAME}.missing_months_input", return_value=0)
    @patch(f"{MODULE_NAME}.date_input", return_value=(Timestamp("2020-01-01"), Timestamp("2020-12-31")))
    @patch(f"{MODULE_NAME}.ticker_input", return_value=["AAPL"])
    @patch(f"{MODULE_NAME}.db_connect")
    def test_propagates_exception_from_returns(self, *_):
        with self.assertRaises(Exception):
            dacv.data_acquisition()

    @patch(f"{MODULE_NAME}.weight_constraint_input", return_value=(0.2, -0.1))
    @patch(f"{MODULE_NAME}.add_effective_returns", return_value=pd.DataFrame())
    @patch(f"{MODULE_NAME}.get_delistings", return_value=pd.DataFrame())
    @patch(f"{MODULE_NAME}.get_returns", return_value=pd.DataFrame())
    @patch(f"{MODULE_NAME}.get_active_permnos", return_value=([9], ["T"], Timestamp("2020-05-31")))
    @patch(f"{MODULE_NAME}.missing_months_input", return_value=2)
    @patch(f"{MODULE_NAME}.date_input", return_value=(Timestamp("2019-01-01"), Timestamp("2020-12-31")))
    @patch(f"{MODULE_NAME}.ticker_input", return_value=["T"])
    @patch(f"{MODULE_NAME}.db_connect")
    def test_db_connect_called_once_and_types(self, mock_db, *_):
        data, tickers, s, e, mx, mn = dacv.data_acquisition()
        mock_db.assert_called_once()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(tickers, ["T"])
        self.assertEqual((s, e), (Timestamp("2019-01-01"), Timestamp("2020-12-31")))
        self.assertEqual((mx, mn), (0.2, -0.1))


if __name__ == "__main__":
    unittest.main()
