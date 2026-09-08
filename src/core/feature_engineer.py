# src/core/feature_engineer.py
"""
Feature Engineering Module
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

class FeatureEngineer:
    """
    Create and engineer features for analysis and ML
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.date_col = self.config.get('data', {}).get('columns', {}).get('purchase_date', 'purchase_date')
        self.customer_col = self.config.get('data', {}).get('columns', {}).get('customer_id', 'customer_id')
        self.amount_col = self.config.get('data', {}).get('columns', {}).get('amount', 'amount')
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features from raw data
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        try:
            self.logger.info("Creating features...")
            
            # Copy to avoid modifying original
            df_copy = df.copy()
            
            # Ensure date column is datetime
            if self.date_col in df_copy.columns:
                df_copy[self.date_col] = pd.to_datetime(df_copy[self.date_col])
            
            # Create date-based features
            df_copy = self._create_date_features(df_copy)
            
            # Create customer-based features
            df_copy = self._create_customer_features(df_copy)
            
            # Create transaction-based features
            df_copy = self._create_transaction_features(df_copy)
            
            # Create monetary features
            df_copy = self._create_monetary_features(df_copy)
            
            # Create interaction features
            df_copy = self._create_interaction_features(df_copy)
            
            self.logger.info(f"Features created: {len(df_copy.columns)} total columns")
            
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {str(e)}")
            return df
    
    def _create_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create date-based features"""
        try:
            if self.date_col not in df.columns:
                return df
            
            # Extract date components
            df['year'] = df[self.date_col].dt.year
            df['month'] = df[self.date_col].dt.month
            df['day'] = df[self.date_col].dt.day
            df['day_of_week'] = df[self.date_col].dt.dayofweek
            df['quarter'] = df[self.date_col].dt.quarter
            df['week_of_year'] = df[self.date_col].dt.isocalendar().week
            
            # Is weekend
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Is holiday (simplified - can be expanded)
            df['is_holiday'] = df[self.date_col].apply(self._is_holiday).astype(int)
            
            # Day of month
            df['day_of_month'] = df[self.date_col].dt.day
            
            # Month name
            df['month_name'] = df[self.date_col].dt.month_name()
            
            # Season
            df['season'] = df['month'].apply(self._get_season)
            
            self.logger.info(f"Created date features")
            return df
            
        except Exception as e:
            self.logger.warning(f"Date features creation failed: {str(e)}")
            return df
    
    def _create_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create customer-based features"""
        try:
            if self.customer_col not in df.columns:
                return df
            
            # Customer tenure (days since first purchase)
            first_purchase = df.groupby(self.customer_col)[self.date_col].transform('min')
            df['customer_tenure_days'] = (df[self.date_col] - first_purchase).dt.days
            
            # Customer tenure in months
            df['customer_tenure_months'] = df['customer_tenure_days'] / 30.44
            
            # Customer lifetime (total days from first to last purchase)
            last_purchase = df.groupby(self.customer_col)[self.date_col].transform('max')
            df['customer_lifetime_days'] = (last_purchase - first_purchase).dt.days
            
            # Number of purchases per customer (using transform)
            purchase_count = df.groupby(self.customer_col).size()
            df['customer_purchase_count'] = df[self.customer_col].map(purchase_count)
            
            # Average days between purchases
            df['avg_days_between_purchases'] = df.groupby(self.customer_col)[self.date_col].transform(
                lambda x: x.diff().dt.days.mean() if len(x) > 1 else 0
            )
            
            self.logger.info(f"Created customer features")
            return df
            
        except Exception as e:
            self.logger.warning(f"Customer features creation failed: {str(e)}")
            return df
    
    def _create_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create transaction-based features"""
        try:
            if self.amount_col not in df.columns:
                return df
            
            # Transaction amount features
            df['amount_log'] = np.log1p(df[self.amount_col])
            
            # Amount normalized by customer
            customer_mean = df.groupby(self.customer_col)[self.amount_col].transform('mean')
            df['amount_vs_customer_avg'] = df[self.amount_col] / (customer_mean + 1)
            
            # Cumulative sum per customer
            df['cumulative_amount'] = df.groupby(self.customer_col)[self.amount_col].cumsum()
            
            # Cumulative count per customer
            df['cumulative_count'] = df.groupby(self.customer_col).cumcount() + 1
            
            # Amount rank per customer
            df['amount_rank'] = df.groupby(self.customer_col)[self.amount_col].rank(method='dense')
            
            # Recent purchase amount (last 30 days)
            days_since_last = df.groupby(self.customer_col)[self.date_col].transform(
                lambda x: (x.max() - x).dt.days
            )
            df['amount_last_30_days'] = df.apply(
                lambda row: row[self.amount_col] if days_since_last.iloc[0] <= 30 else 0,
                axis=1
            )
            
            self.logger.info(f"Created transaction features")
            return df
            
        except Exception as e:
            self.logger.warning(f"Transaction features creation failed: {str(e)}")
            return df
    
    def _create_monetary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create monetary features"""
        try:
            if self.amount_col not in df.columns:
                return df
            
            # Amount categories
            df['amount_category'] = pd.cut(
                df[self.amount_col],
                bins=[0, 50, 200, 500, 1000, float('inf')],
                labels=['very_low', 'low', 'medium', 'high', 'very_high']
            )
            
            # Total amount per customer (using transform)
            total_amount = df.groupby(self.customer_col)[self.amount_col].transform('sum')
            df['customer_total_amount'] = total_amount
            
            # Average amount per customer
            avg_amount = df.groupby(self.customer_col)[self.amount_col].transform('mean')
            df['customer_avg_amount'] = avg_amount
            
            # Max amount per customer
            max_amount = df.groupby(self.customer_col)[self.amount_col].transform('max')
            df['customer_max_amount'] = max_amount
            
            # Min amount per customer
            min_amount = df.groupby(self.customer_col)[self.amount_col].transform('min')
            df['customer_min_amount'] = min_amount
            
            # Amount std per customer
            std_amount = df.groupby(self.customer_col)[self.amount_col].transform('std')
            df['customer_amount_std'] = std_amount.fillna(0)
            
            # Coefficient of variation
            df['customer_amount_cv'] = df['customer_amount_std'] / (df['customer_avg_amount'] + 1)
            
            self.logger.info(f"Created monetary features")
            return df
            
        except Exception as e:
            self.logger.warning(f"Monetary features creation failed: {str(e)}")
            return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between existing features"""
        try:
            # Interaction between amount and recency (if both exist)
            if self.amount_col in df.columns and self.date_col in df.columns:
                # Recency score (days since purchase)
                ref_date = df[self.date_col].max()
                df['recency_days'] = (ref_date - df[self.date_col]).dt.days
                
                # Amount × Recency interaction
                df['amount_recency_interaction'] = df[self.amount_col] / (df['recency_days'] + 1)
            
            # Frequency × Monetary interaction
            if 'customer_purchase_count' in df.columns and 'customer_total_amount' in df.columns:
                df['frequency_monetary_interaction'] = df['customer_total_amount'] / (df['customer_purchase_count'] + 1)
            
            # Average order value if exists
            if 'customer_total_amount' in df.columns and 'customer_purchase_count' in df.columns:
                df['avg_order_value'] = df['customer_total_amount'] / (df['customer_purchase_count'] + 1)
            
            self.logger.info(f"Created interaction features")
            return df
            
        except Exception as e:
            self.logger.warning(f"Interaction features creation failed: {str(e)}")
            return df
    
    def _is_holiday(self, date: pd.Timestamp) -> bool:
        """Check if date is a holiday (simplified)"""
        # Add your holiday logic here
        holidays = [
            '2026-01-01', '2026-05-01', '2026-07-05', 
            '2026-12-25', '2026-12-26'
        ]
        return date.strftime('%Y-%m-%d') in holidays
    
    def _get_season(self, month: int) -> str:
        """Get season from month"""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of all feature names"""
        return df.columns.tolist()