"""
Data Handler Module
Handles data loading, cleaning, and temporal splits
"""
import pandas as pd
import numpy as np


class DataHandler:
    """Manages price data and creates chronological splits"""

    def __init__(self, prices: pd.DataFrame, train_pct=0.6, test_pct=0.2, val_pct=0.2):
        """
        Args:
            prices: DataFrame with datetime index and asset columns
            train_pct: Training set percentage (default 60%)
            test_pct: Testing set percentage (default 20%)
            val_pct: Validation set percentage (default 20%)
        """
        if abs(train_pct + test_pct + val_pct - 1.0) > 1e-6:
            raise ValueError("Percentages must sum to 1.0")

        self.prices = prices.dropna(how='any')
        self.train_pct = train_pct
        self.test_pct = test_pct
        self.val_pct = val_pct

        self._create_splits()

    def _create_splits(self):
        """Create chronological train/test/validation splits"""
        n = len(self.prices)

        train_end = int(n * self.train_pct)
        test_end = train_end + int(n * self.test_pct)

        self.train_data = self.prices.iloc[:train_end]
        self.test_data = self.prices.iloc[train_end:test_end]
        self.val_data = self.prices.iloc[test_end:]

        print(f"Data splits created:")
        print(f"  Training:   {self.train_data.index[0]} to {self.train_data.index[-1]} ({len(self.train_data)} days)")
        print(f"  Testing:    {self.test_data.index[0]} to {self.test_data.index[-1]} ({len(self.test_data)} days)")
        print(f"  Validation: {self.val_data.index[0]} to {self.val_data.index[-1]} ({len(self.val_data)} days)")

    def get_pair_data(self, asset1: str, asset2: str, split='train'):
        """
        Extract price data for a specific pair

        Args:
            asset1: First asset ticker
            asset2: Second asset ticker
            split: 'train', 'test', or 'val'

        Returns:
            DataFrame with two columns (asset1, asset2)
        """
        if split == 'train':
            data = self.train_data
        elif split == 'test':
            data = self.test_data
        elif split == 'val':
            data = self.val_data
        else:
            raise ValueError("split must be 'train', 'test', or 'val'")

        return data[[asset1, asset2]].copy()

    def get_all_data(self, asset1: str, asset2: str):
        """Get complete dataset for a pair (all splits combined)"""
        return self.prices[[asset1, asset2]].copy()