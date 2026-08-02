import unittest

import pandas as pd

from src.feature_engineering import build_customer_lifecycle_features


class CustomerLifecycleTests(unittest.TestCase):
    def _row(self, customer_id, customer_name, date, sell_price=1000, va_amount=500):
        return {
            'Title': f'{customer_id}-{date}',
            'CustomerID': customer_id,
            'Customer Name': customer_name,
            'SalesIn': pd.Timestamp(date),
            'Sell Price': sell_price,
            'VA Amount': va_amount,
            'Profit Margin': va_amount / sell_price,
        }

    def test_churn_risk_uses_distinct_dates_and_median_cadence(self):
        rows = []
        rows.extend(
            [
                self._row('CID_ACTIVE', 'CUST_ACTIVE', '2026-01-01'),
                self._row('CID_ACTIVE', 'CUST_ACTIVE', '2026-01-01'),
                self._row('CID_ACTIVE', 'CUST_ACTIVE', '2026-02-01'),
                self._row('CID_ACTIVE', 'CUST_ACTIVE', '2026-03-01'),
                self._row('CID_ACTIVE', 'CUST_ACTIVE', '2026-04-01'),
                self._row('CID_ACTIVE', 'CUST_ACTIVE', '2026-05-01'),
            ]
        )
        rows.extend(
            [
                self._row('CID_DUE', 'CUST_DUE', '2026-02-14'),
                self._row('CID_DUE', 'CUST_DUE', '2026-03-15'),
                self._row('CID_DUE', 'CUST_DUE', '2026-04-15'),
            ]
        )
        rows.extend(
            [
                self._row('CID_HIGH', 'CUST_HIGH', '2026-01-01'),
                self._row('CID_HIGH', 'CUST_HIGH', '2026-01-31'),
                self._row('CID_HIGH', 'CUST_HIGH', '2026-03-01'),
            ]
        )
        rows.append(self._row('CID_SINGLE', 'CUST_SINGLE', '2026-05-01'))

        result = build_customer_lifecycle_features(
            pd.DataFrame(rows),
            reference_date=pd.Timestamp('2026-05-25'),
        ).set_index('CustomerID')

        self.assertEqual(result.loc['CID_ACTIVE', 'Distinct Order Days'], 5)
        self.assertGreater(result.loc['CID_ACTIVE', 'Median Reorder Days'], 20)
        self.assertEqual(result.loc['CID_ACTIVE', 'Churn Risk'], 'Active cadence')
        self.assertEqual(result.loc['CID_ACTIVE', 'Churn Confidence'], 'High')
        self.assertEqual(result.loc['CID_DUE', 'Churn Risk'], 'Due for reorder')
        self.assertEqual(result.loc['CID_HIGH', 'Churn Risk'], 'Likely churn')
        self.assertEqual(result.loc['CID_SINGLE', 'Churn Risk'], 'Single-order customer')
        self.assertNotIn('Status', result.columns)
        self.assertEqual(result.loc['CID_DUE', 'Active Months'], 3)
        self.assertEqual(result.loc['CID_DUE', 'Average Monthly VA'], 500)
        self.assertEqual(result.loc['CID_DUE', 'Average Annual VA'], 6000)
        self.assertEqual(result.loc['CID_DUE', 'Value at Risk'], 6000)
        self.assertGreater(result.loc['CID_DUE', 'Priority Score'], 0)
        self.assertLessEqual(result.loc['CID_DUE', 'Priority Score'], 100)
        self.assertGreater(result.loc['CID_HIGH', 'Priority Score Raw'], 0)
        self.assertLessEqual(result.loc['CID_HIGH', 'Priority Score'], 100)
        self.assertIn('Max Reorder Days', result.columns)
        self.assertGreater(result.loc['CID_HIGH', 'Days Beyond Max Reorder Gap'], 0)
        self.assertIn('longest historical reorder gap', result.loc['CID_HIGH', 'Churn Reason'])
        self.assertEqual(result.loc['CID_ACTIVE', 'Value at Risk'], 0)


if __name__ == '__main__':
    unittest.main()
