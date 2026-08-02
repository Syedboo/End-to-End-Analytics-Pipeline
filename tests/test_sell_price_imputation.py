import unittest

import numpy as np
import pandas as pd

from src.data_cleaning import clean_data
from src.sell_price_imputation import estimate_sell_price


class SellPriceFormulaImputationTests(unittest.TestCase):
    def test_cleaning_imputes_missing_sell_price_from_business_formula(self):
        raw = pd.DataFrame(
            {
                "Title": ["J1", "J2", "J3", "J4", "J5"],
                "Sell Price": [np.nan, 500.0, 0.0, "\u00a3-", "\u00a3 -"],
                "Purchases": [200.0, 250.0, 240.0, 400.0, 50.0],
                "Rebate": [25.0, 0.0, 10.0, 20.0, 5.0],
                "VA Amount": [75.0, 250.0, 50.0, 80.0, 45.0],
                "VA%": [np.nan, 0.50, np.nan, np.nan, np.nan],
                "Mup%": [10.0, 10.0, 10.0, 10.0, 10.0],
            }
        )

        cleaned, result = clean_data(raw)

        self.assertEqual(cleaned.loc[0, "Sell Price"], 300.0)
        self.assertTrue(bool(cleaned.loc[0, "Sell Price Formula Imputed"]))
        self.assertTrue(bool(cleaned.loc[0, "Sell Price Was Imputed"]))
        self.assertFalse(bool(cleaned.loc[1, "Sell Price Formula Imputed"]))
        self.assertEqual(cleaned.loc[2, "Sell Price"], 300.0)
        self.assertTrue(bool(cleaned.loc[2, "Sell Price Formula Imputed"]))
        self.assertEqual(cleaned.loc[3, "Sell Price"], 500.0)
        self.assertTrue(bool(cleaned.loc[3, "Sell Price Formula Imputed"]))
        self.assertEqual(cleaned.loc[4, "Sell Price"], 100.0)
        self.assertTrue(bool(cleaned.loc[4, "Sell Price Formula Imputed"]))
        self.assertEqual(cleaned.loc[0, "sell_price_source_missing"], 1)
        self.assertEqual(cleaned.loc[3, "sell_price_source_missing"], 1)
        self.assertEqual(cleaned.loc[4, "sell_price_source_missing"], 1)
        self.assertAlmostEqual(cleaned.loc[0, "VA%"], 75.0 / 300.0)
        self.assertTrue(
            any("Purchases plus Rebate plus VA Amount" in action for action in result.cleaning_actions)
        )

    def test_sell_price_metadata_preserves_formula_imputed_source(self):
        frame = pd.DataFrame(
            {
                "Sell Price": [300.0, 500.0],
                "Purchases": [200.0, 250.0],
                "Rebate": [25.0, 0.0],
                "VA Amount": [75.0, 250.0],
                "Sell Price Formula Imputed": [True, False],
            }
        )

        result = estimate_sell_price(frame).data

        self.assertEqual(result.loc[0, "Sell Price Source"], "Formula Imputed")
        self.assertEqual(result.loc[0, "Sell Price Confidence"], "Formula")
        self.assertTrue(bool(result.loc[0, "Sell Price Was Imputed"]))
        self.assertEqual(result.loc[0, "Estimated Sell Price"], 300.0)
        self.assertEqual(result.loc[1, "Sell Price Source"], "Actual")
        self.assertFalse(bool(result.loc[1, "Sell Price Was Imputed"]))


if __name__ == "__main__":
    unittest.main()
