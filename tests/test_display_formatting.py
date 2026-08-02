import unittest

from src.utils import display_format_for_column


class DisplayFormattingTests(unittest.TestCase):
    def test_count_and_day_columns_are_not_currency(self):
        self.assertEqual(display_format_for_column("Order Count", 999), "{:,.0f}")
        self.assertEqual(display_format_for_column("Distinct Order Days", 999), "{:,.0f}")
        self.assertEqual(display_format_for_column("Days Since Last Order", 999), "{:,.0f}")
        self.assertEqual(display_format_for_column("Reorder Cadence Days", 999), "{:,.1f}")

    def test_financial_columns_remain_currency(self):
        self.assertEqual(display_format_for_column("Customer Lifetime Revenue", 1000), "£{:,.0f}")
        self.assertEqual(display_format_for_column("Average Order Value", 1000), "£{:,.0f}")
        self.assertEqual(display_format_for_column("Customer Lifetime VA", 1000), "£{:,.0f}")

    def test_percent_columns_remain_percent(self):
        self.assertEqual(display_format_for_column("Average VA Margin", 1), "{:.1%}")
        self.assertEqual(display_format_for_column("Markup", 35), "{:.1f}%")
        self.assertEqual(display_format_for_column("Revenue Share", 1), "{:.1%}")
        self.assertEqual(display_format_for_column("VA Share", 1), "{:.1%}")


if __name__ == "__main__":
    unittest.main()
