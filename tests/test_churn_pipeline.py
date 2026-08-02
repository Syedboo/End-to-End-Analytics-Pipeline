"""Tests for leakage-safe customer churn modelling utilities."""

from __future__ import annotations

import unittest

import pandas as pd

from src.churn.config import ChurnConfig
from src.churn.features import build_customer_snapshots, prepare_churn_transactions
from src.churn.model import fit_churn_models


class ChurnPipelineTests(unittest.TestCase):
    """Validate core churn date-window and scoring behaviour."""

    def setUp(self) -> None:
        self.config = ChurnConfig(
            observation_window_days=120,
            prediction_window_days=30,
            gap_days=0,
            min_history_days=10,
            min_distinct_order_days=2,
            min_reorder_window_days=10,
            contact_capacity=2,
        )
        rows = []
        for customer, dates, future, price in [
            ("CID_A", ["2024-01-01", "2024-01-15", "2024-02-01"], "2024-03-20", 100.0),
            ("CID_B", ["2024-01-01", "2024-01-15", "2024-02-01"], None, 200.0),
            ("CID_C", ["2024-02-20"], None, 300.0),
        ]:
            all_dates = list(dates)
            if future:
                all_dates.append(future)
            for idx, date in enumerate(all_dates):
                rows.append(
                    {
                        "Title": f"{customer}_{idx}",
                        "CustomerID": customer,
                        "Customer Name": customer.replace("CID", "Customer"),
                        "SalesIn": date,
                        "Sell Price": price,
                        "Estimated Sell Price": price,
                        "VA Amount": price * 0.4,
                        "Purchases": price * 0.3,
                        "Job Status": "z-Closed",
                        "Product Type": "Leaflet",
                        "Work Type": "Digital",
                        "Rep": "Rep A",
                        "Region": "North",
                        "Industry": "Retail",
                        "Currency": "Stg",
                    }
                )
        self.df = pd.DataFrame(rows)

    def _snapshots(self) -> pd.DataFrame:
        prepared = prepare_churn_transactions(self.df, self.config)
        return build_customer_snapshots(
            prepared,
            self.config,
            snapshot_dates=[pd.Timestamp("2024-03-01")],
        )

    def test_snapshot_features_do_not_use_future_orders(self) -> None:
        snapshots = self._snapshots().set_index("CustomerID")
        self.assertEqual(snapshots.loc["CID_A", "Revenue Lifetime"], 300.0)
        self.assertEqual(snapshots.loc["CID_A", "Future Order Count"], 1)
        self.assertEqual(snapshots.loc["CID_A", "Churn Label"], 0)

    def test_churn_label_uses_prediction_window(self) -> None:
        snapshots = self._snapshots().set_index("CustomerID")
        self.assertTrue(bool(snapshots.loc["CID_B", "Eligible For Training"]))
        self.assertEqual(snapshots.loc["CID_B", "Churn Label"], 1)

    def test_single_order_customer_is_cold_start_not_training_label(self) -> None:
        snapshots = self._snapshots().set_index("CustomerID")
        self.assertFalse(bool(snapshots.loc["CID_C", "Eligible For Training"]))
        self.assertTrue(bool(snapshots.loc["CID_C", "Cold Start Flag"]))
        self.assertTrue(pd.isna(snapshots.loc["CID_C", "Churn Label"]))

    def test_prediction_output_contains_business_priority_columns(self) -> None:
        snapshots = self._snapshots()
        result = fit_churn_models(snapshots, snapshots.copy(), self.config)
        self.assertFalse(result.predictions.empty)
        self.assertIn("Priority Rank", result.predictions.columns)
        self.assertIn("Priority Score", result.predictions.columns)
        self.assertIn("Priority Score Raw", result.predictions.columns)
        self.assertIn("Max Reorder Days", result.predictions.columns)
        self.assertIn("Days Beyond Max Reorder Gap", result.predictions.columns)
        self.assertIn("Top three churn reasons", result.predictions.columns)
        self.assertIn("Recommended retention action", result.predictions.columns)
        self.assertGreaterEqual(result.predictions["Priority Score"].min(), 0.0)
        self.assertLessEqual(result.predictions["Priority Score"].max(), 100.0)


if __name__ == "__main__":
    unittest.main()