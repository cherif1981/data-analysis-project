# src/analyzers/rfm_enhanced.py
"""
Enhanced RFM Analysis with 7 Customer Segments
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from enum import Enum

class CustomerSegment(Enum):
    """Customer segmentation based on RFM scores"""
    CHAMPIONS = "Champions"
    LOYAL = "Loyal Customers"
    POTENTIAL_LOYALISTS = "Potential Loyalists"
    NEW_CUSTOMERS = "New Customers"
    AT_RISK = "At Risk"
    CANT_LOSE = "Can't Lose Them"
    LOST = "Lost Customers"

class RFMAnalyzerEnhanced:
    """
    Enhanced RFM Analysis with 7 customer segments
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.segments = {
            CustomerSegment.CHAMPIONS: {
                'r_range': (4, 5),
                'f_range': (4, 5),
                'm_range': (4, 5),
                'description': 'Best customers - high value, frequent, recent',
                'strategy': 'Reward and retain - VIP program'
            },
            CustomerSegment.LOYAL: {
                'r_range': (3, 5),
                'f_range': (3, 5),
                'm_range': (3, 5),
                'description': 'Regular, loyal customers with good value',
                'strategy': 'Strengthen relationship - loyalty program'
            },
            CustomerSegment.POTENTIAL_LOYALISTS: {
                'r_range': (3, 5),
                'f_range': (1, 3),
                'm_range': (1, 3),
                'description': 'Recent but not frequent, potential to become loyal',
                'strategy': 'Increase frequency - targeted offers'
            },
            CustomerSegment.NEW_CUSTOMERS: {
                'r_range': (4, 5),
                'f_range': (1, 2),
                'm_range': (1, 2),
                'description': 'New customers with potential',
                'strategy': 'Engage and nurture - welcome program'
            },
            CustomerSegment.AT_RISK: {
                'r_range': (1, 2),
                'f_range': (3, 5),
                'm_range': (3, 5),
                'description': 'High value but not recent - at risk of churn',
                'strategy': 'Re-engage - win-back campaign'
            },
            CustomerSegment.CANT_LOSE: {
                'r_range': (1, 2),
                'f_range': (4, 5),
                'm_range': (4, 5),
                'description': 'High value, high frequency but not recent',
                'strategy': 'Urgent re-engagement - special offers'
            },
            CustomerSegment.LOST: {
                'r_range': (1, 1),
                'f_range': (1, 3),
                'm_range': (1, 3),
                'description': 'Low recency, low frequency, low monetary',
                'strategy': 'Reactivation campaign or let go'
            }
        }
    
    def calculate_rfm(self, df: pd.DataFrame, reference_date: str = None) -> pd.DataFrame:
        """Calculate RFM scores"""
        try:
            # Prepare data
            if reference_date:
                ref_date = pd.to_datetime(reference_date)
            else:
                ref_date = df['purchase_date'].max() + pd.Timedelta(days=1)
            
            # Calculate RFM
            rfm = df.groupby('customer_id').agg({
                'purchase_date': lambda x: (ref_date - x.max()).days,
                'customer_id': 'count',
                'amount': 'sum'
            }).rename(columns={
                'purchase_date': 'recency',
                'customer_id': 'frequency',
                'amount': 'monetary'
            })
            
            # Calculate RFM scores (1-5)
            rfm['r_score'] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
            rfm['f_score'] = pd.qcut(rfm['frequency'], 5, labels=[1, 2, 3, 4, 5])
            rfm['m_score'] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])
            
            # Convert scores to int
            rfm['r_score'] = rfm['r_score'].astype(int)
            rfm['f_score'] = rfm['f_score'].astype(int)
            rfm['m_score'] = rfm['m_score'].astype(int)
            
            # Assign segments
            rfm['segment'] = rfm.apply(self._assign_segment, axis=1)
            
            # Calculate segment value
            rfm['segment_value'] = rfm.apply(self._calculate_segment_value, axis=1)
            
            return rfm
            
        except Exception as e:
            self.logger.error(f"RFM calculation failed: {str(e)}")
            raise
    
    def _assign_segment(self, row: pd.Series) -> str:
        """Assign customer segment based on RFM scores"""
        r, f, m = row['r_score'], row['f_score'], row['m_score']
        
        if r >= 4 and f >= 4 and m >= 4:
            return CustomerSegment.CHAMPIONS.value
        elif r >= 3 and f >= 3 and m >= 3:
            return CustomerSegment.LOYAL.value
        elif r >= 3 and f >= 1 and m >= 1:
            return CustomerSegment.POTENTIAL_LOYALISTS.value
        elif r >= 4 and f <= 2 and m <= 2:
            return CustomerSegment.NEW_CUSTOMERS.value
        elif r <= 2 and f >= 3 and m >= 3:
            return CustomerSegment.CANT_LOSE.value
        elif r <= 2 and f >= 1 and m >= 1:
            return CustomerSegment.AT_RISK.value
        else:
            return CustomerSegment.LOST.value
    
    def _calculate_segment_value(self, row: pd.Series) -> float:
        """Calculate segment value based on scores"""
        return (row['r_score'] + row['f_score'] + row['m_score']) / 3
    
    def get_segment_analysis(self, rfm_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive segment analysis"""
        analysis = {
            'segment_summary': {},
            'segment_recommendations': {},
            'segment_value_stats': {},
            'total_customers': len(rfm_df)
        }
        
        # Calculate segment statistics
        for segment in CustomerSegment:
            seg_data = rfm_df[rfm_df['segment'] == segment.value]
            if not seg_data.empty:
                analysis['segment_summary'][segment.value] = {
                    'count': len(seg_data),
                    'percentage': (len(seg_data) / len(rfm_df) * 100),
                    'avg_recency': seg_data['recency'].mean(),
                    'avg_frequency': seg_data['frequency'].mean(),
                    'avg_monetary': seg_data['monetary'].mean(),
                    'avg_value': seg_data['segment_value'].mean(),
                    'total_monetary': seg_data['monetary'].sum()
                }
                
                # Get strategy
                analysis['segment_recommendations'][segment.value] = \
                    self.segments[segment]['strategy']
        
        # Value distribution
        analysis['segment_value_stats'] = {
            'min': rfm_df['segment_value'].min(),
            'max': rfm_df['segment_value'].max(),
            'mean': rfm_df['segment_value'].mean(),
            'std': rfm_df['segment_value'].std()
        }
        
        return analysis
    
    def get_customer_insights(self, customer_id: str, rfm_df: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed insights for a specific customer"""
        customer_data = rfm_df[rfm_df.index == customer_id]
        
        if customer_data.empty:
            return {'error': 'Customer not found'}
        
        row = customer_data.iloc[0]
        
        return {
            'customer_id': customer_id,
            'segment': row['segment'],
            'recency': row['recency'],
            'frequency': row['frequency'],
            'monetary': row['monetary'],
            'r_score': row['r_score'],
            'f_score': row['f_score'],
            'm_score': row['m_score'],
            'segment_value': row['segment_value'],
            'recommendation': self.segments[CustomerSegment(row['segment'])]['strategy']
        }