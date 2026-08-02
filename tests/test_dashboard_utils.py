import unittest

import pandas as pd

from src.dashboard_utils import filter_dashboard_data


class DashboardFilterTests(unittest.TestCase):
    def test_filters_year_month_and_quality_status(self):
        frame = pd.DataFrame(
            {
                "Year": [2025, 2025, 2026, 2026],
                "Month": [1, 2, 1, 2],
                "Quality Status": ["PASS", "FAIL", "WARNING", "PASS"],
                "Sell Price": [100, 200, 300, 400],
            }
        )

        filtered = filter_dashboard_data(
            frame,
            selected_years=("2025", "2026"),
            selected_months=("1",),
            include_flagged_rows=False,
        )

        self.assertEqual(filtered["Sell Price"].tolist(), [100])

    def test_keeps_flagged_rows_when_requested(self):
        frame = pd.DataFrame(
            {
                "Year": [2026, 2026],
                "Month": [1, 1],
                "Quality Status": ["PASS", "FAIL"],
                "Sell Price": [100, 200],
            }
        )

        filtered = filter_dashboard_data(
            frame,
            selected_years=("2026",),
            selected_months=("1",),
            include_flagged_rows=True,
        )

        self.assertEqual(filtered["Sell Price"].tolist(), [100, 200])


if __name__ == "__main__":
    unittest.main()
