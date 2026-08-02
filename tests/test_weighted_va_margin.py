import unittest

import numpy as np
import pandas as pd

from src.eda import build_business_tables
from src.feature_engineering import build_customer_lifecycle_features
from src.calculation_validation import validate_monthly_summary
from src.utils import weighted_va_margin


class WeightedVAMarginTests(unittest.TestCase):
    def _sample_frame(self):
        return pd.DataFrame(
            {
                "Title": ["J001", "J002"],
                "CustomerID": ["CID_001", "CID_001"],
                "Customer Name": ["CUST_024", "CUST_024"],
                "Industry": ["Retail", "Retail"],
                "Region": ["NI", "NI"],
                "Product Type": ["Postcards/Tags", "Postcards/Tags"],
                "Work Type": ["Litho", "Litho"],
                "Binding Type": ["None", "None"],
                "Rep": ["Rep A", "Rep A"],
                "SalesIn": pd.to_datetime(["2026-01-05", "2026-01-20"]),
                "Sales Month": pd.to_datetime(["2026-01-01", "2026-01-01"]),
                "Sell Price": [60000.0, 8350.0],
                "VA Amount": [1000.0, 156.0],
                # Deliberately wrong row-level margins prove grouped outputs do
                # not average VA% and instead use total VA / total revenue.
                "Profit Margin": [-1.126, -1.126],
                "Mup%": [0.10, 0.20],
                "Quantity": [1000, 200],
                "Press hrs": [10.0, 2.0],
                "Impressions": [10000, 2000],
                "Labour": [500.0, 100.0],
                "Paper": [250.0, 50.0],
                "Purchases": [30000.0, 4000.0],
            }
        )

    def test_business_tables_use_weighted_va_margin(self):
        tables = build_business_tables(self._sample_frame())
        result = tables["product_types_profitability"].set_index("Product Type")

        self.assertAlmostEqual(
            result.loc["Postcards/Tags", "VA_Margin"],
            1156 / 68350,
            places=8,
        )

    def test_monthly_validation_uses_weighted_va_margin(self):
        frame = self._sample_frame()
        tables = build_business_tables(frame)
        validation = validate_monthly_summary(frame, tables["monthly_sales_profitability_trend"])
        margin_row = validation.set_index("Metric").loc["Monthly VA_Margin"]

        self.assertEqual(margin_row["Status"], "PASS")
        self.assertEqual(margin_row["Failures"], 0)

    def test_customer_lifecycle_uses_weighted_average_va_margin(self):
        lifecycle = build_customer_lifecycle_features(
            self._sample_frame(),
            reference_date=pd.Timestamp("2026-02-01"),
        ).set_index("CustomerID")

        self.assertAlmostEqual(
            lifecycle.loc["CID_001", "Average VA Margin"],
            1156 / 68350,
            places=8,
        )

    def test_zero_total_sell_price_returns_nan(self):
        margin = weighted_va_margin(
            pd.Series([100.0, -50.0]),
            pd.Series([0.0, 0.0]),
        )

        self.assertTrue(np.isnan(margin))


if __name__ == "__main__":
    unittest.main()
