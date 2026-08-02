import unittest

import pandas as pd

from src.dashboard_utils import (
    build_data_confidence_summary,
    build_executive_focus_cards,
    build_pricing_review_table,
    build_recommended_actions,
)


class BoardDashboardTests(unittest.TestCase):
    def _jobs(self):
        return pd.DataFrame(
            {
                "Title": ["J1", "J2", "J3"],
                "Customer Name": ["CUST_A", "CUST_A", "CUST_B"],
                "Product Type": ["Books", "Books", "Cards"],
                "Work Type": ["Litho", "Litho", "Digital"],
                "Rep": ["REP_1", "REP_1", "REP_2"],
                "Sell Price": [1000.0, 0.0, 500.0],
                "VA Amount": [400.0, -100.0, 50.0],
                "Purchases": [600.0, 200.0, 700.0],
                "Quality Status": ["PASS", "FAIL", "WARNING"],
                "Reason": ["PASS", "Zero Revenue; Negative Margin", "Sell Price < Purchase Cost"],
            }
        )

    def test_data_confidence_summary_counts_statuses(self):
        summary, headline = build_data_confidence_summary(self._jobs())

        counts = summary.set_index("Quality Status")["Rows"].to_dict()
        self.assertEqual(counts["PASS"], 1)
        self.assertEqual(counts["WARNING"], 1)
        self.assertEqual(counts["FAIL"], 1)
        self.assertEqual(headline["label"], "Needs attention")

    def test_pricing_review_table_groups_review_rows(self):
        review = build_pricing_review_table(self._jobs(), top_n=10)

        self.assertFalse(review.empty)
        self.assertIn("Below_Purchase_Jobs", review.columns)
        self.assertGreaterEqual(review["Below_Purchase_Jobs"].sum(), 1)
        self.assertGreaterEqual(review["Negative_Margin_Jobs"].sum(), 1)

    def test_executive_focus_cards_show_action_themes(self):
        lifecycle = pd.DataFrame(
            {
                "Customer Name": ["CUST_A", "CUST_B"],
                "Churn Risk": ["Due for reorder", "Active cadence"],
                "Value at Risk": [1200.0, 0.0],
            }
        )
        product_table = pd.DataFrame(
            {
                "Product Type": ["Cards", "Books"],
                "Revenue": [5000.0, 10000.0],
                "VA_Margin": [0.10, 0.40],
            }
        )

        cards = build_executive_focus_cards(
            self._jobs(),
            lifecycle,
            {"product_types_profitability": product_table},
        )

        self.assertEqual(list(cards["Theme"]), ["Customer retention", "Pricing review", "Product mix"])
        self.assertIn("1 customer needs", cards.loc[cards["Theme"].eq("Customer retention"), "Headline"].iloc[0])
        self.assertEqual(cards.loc[cards["Theme"].eq("Customer retention"), "Value"].iloc[0], 1200.0)
        self.assertEqual(cards.loc[cards["Theme"].eq("Product mix"), "Value"].iloc[0], 5000.0)

    def test_recommended_actions_combines_customer_and_pricing_actions(self):
        lifecycle = pd.DataFrame(
            {
                "Customer Name": ["CUST_A"],
                "Churn Risk": ["Due for reorder"],
                "Churn Reason": ["No order for 70 days."],
                "Priority Rank": [1],
                "Value at Risk": [1200.0],
            }
        )
        product_table = pd.DataFrame(
            {
                "Product Type": ["Cards"],
                "Revenue": [5000.0],
                "VA_Margin": [0.10],
            }
        )

        actions = build_recommended_actions(
            self._jobs(),
            lifecycle,
            {"product_types_profitability": product_table},
            top_n=10,
        )

        self.assertIn("Customer retention", set(actions["Area"]))
        self.assertIn("Pricing review", set(actions["Area"]))
        self.assertIn("Data confidence", set(actions["Area"]))


if __name__ == "__main__":
    unittest.main()
