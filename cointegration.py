"""
Cointegration Testing Module
Identifies cointegrated pairs using Engle-Granger methodology
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


class CointegrationTester:
    """Tests for cointegration between asset pairs"""

    def __init__(self, use_log=True, alpha=0.05):
        """
        Args:
            use_log: Use log prices (recommended for financial data)
            alpha: Significance level for statistical tests
        """
        self.use_log = use_log
        self.alpha = alpha

    def test_pair(self, prices: pd.DataFrame):
        """
        Test if two assets are cointegrated using Engle-Granger method

        Args:
            prices: DataFrame with exactly 2 columns (asset prices)

        Returns:
            dict with test results and OLS parameters
        """
        if prices.shape[1] != 2:
            raise ValueError("DataFrame must have exactly 2 columns")

        col1, col2 = prices.columns.tolist()
        df = prices.dropna().copy()

        s1 = df[col1].astype(float)
        s2 = df[col2].astype(float)

        # Apply log transformation if requested
        if self.use_log:
            s1 = np.log(s1)
            s2 = np.log(s2)

        # Step 1: Check that both series are non-stationary (I(1))
        p1 = adfuller(s1.values, regression="c", autolag="AIC")[1]
        p2 = adfuller(s2.values, regression="c", autolag="AIC")[1]

        # Step 2: Run OLS regression S1 = beta0 + beta1*S2 + residuals
        X = sm.add_constant(s2.values)
        model = sm.OLS(s1.values, X).fit()
        beta0, beta1 = model.params
        residuals = s1.values - (beta0 + beta1 * s2.values)

        # Step 3: Test if residuals are stationary
        p_res = adfuller(residuals, regression="n", autolag="AIC")[1]

        # Cointegration conditions
        cond_nonstat = (p1 > self.alpha) and (p2 > self.alpha)
        cond_res_stat = (p_res < self.alpha)
        is_cointegrated = cond_nonstat and cond_res_stat

        return {
            "asset1": col1,
            "asset2": col2,
            "is_cointegrated": is_cointegrated,
            "adf_p_asset1": p1,
            "adf_p_asset2": p2,
            "adf_p_residuals": p_res,
            "beta0": beta0,
            "beta1": beta1,
            "r_squared": model.rsquared
        }

    def find_cointegrated_pairs(self, prices: pd.DataFrame, min_corr=0.7):
        """
        Find all cointegrated pairs in a dataset

        Args:
            prices: DataFrame with multiple asset columns
            min_corr: Minimum correlation threshold (pre-filter)

        Returns:
            List of cointegrated pairs with their statistics
        """
        assets = prices.columns.tolist()
        n = len(assets)
        results = []

        # Calculate correlation matrix for pre-filtering
        corr_matrix = prices.corr()

        print(f"Searching for cointegrated pairs among {n} assets...")
        print(f"Total pairs to test: {n * (n - 1) // 2}")

        tested = 0
        for i in range(n):
            for j in range(i + 1, n):
                # Pre-filter by correlation
                if abs(corr_matrix.iloc[i, j]) < min_corr:
                    continue

                asset1, asset2 = assets[i], assets[j]
                pair_prices = prices[[asset1, asset2]]

                result = self.test_pair(pair_prices)
                tested += 1

                if result["is_cointegrated"]:
                    results.append(result)
                    print(f"✓ Found: {asset1} - {asset2} (p-value: {result['adf_p_residuals']:.4f})")

        print(f"\nTested {tested} pairs with correlation > {min_corr}")
        print(f"Found {len(results)} cointegrated pairs")

        return results