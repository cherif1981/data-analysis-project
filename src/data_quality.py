# src/data_quality.py
"""
Data Quality Reporting
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime

class DataQualityReporter:
    """Generate comprehensive data quality reports"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.report = {}
    
    def generate_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate complete data quality report"""
        self.logger.info("Generating data quality report...")
        
        self.report = {
            'generated_at': datetime.now().isoformat(),
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'duplicates': self._check_duplicates(df),
            'missing_values': self._check_missing(df),
            'invalid_dates': self._check_dates(df),
            'negative_amounts': self._check_negative_amounts(df),
            'invalid_customer_ids': self._check_customer_ids(df),
            'outliers': self._check_outliers(df),
            'wrong_data_types': self._check_data_types(df),
            'column_stats': self._column_stats(df)
        }
        
        # Calculate quality score
        self.report['quality_score'] = self._calculate_quality_score()
        
        # Generate recommendations
        self.report['recommendations'] = self._generate_recommendations()
        
        self.logger.info(f"Quality score: {self.report['quality_score']:.1f}%")
        return self.report
    
    def _check_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for duplicate rows"""
        duplicates = df.duplicated().sum()
        return {
            'count': int(duplicates),
            'percentage': (duplicates / len(df) * 100) if len(df) > 0 else 0
        }
    
    def _check_missing(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for missing values"""
        total_missing = df.isnull().sum().sum()
        missing_by_column = df.isnull().sum().to_dict()
        missing_percentage = (total_missing / (len(df) * len(df.columns)) * 100) if len(df) > 0 else 0
        
        return {
            'total': int(total_missing),
            'percentage': missing_percentage,
            'by_column': {k: int(v) for k, v in missing_by_column.items() if v > 0}
        }
    
    def _check_dates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for invalid dates"""
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'day' in col.lower()]
        invalid_dates = 0
        
        for col in date_columns:
            try:
                # Try to convert to datetime
                temp = pd.to_datetime(df[col], errors='coerce')
                invalid = temp.isnull().sum()
                invalid_dates += invalid
            except:
                invalid_dates += len(df)
        
        return {
            'count': int(invalid_dates),
            'percentage': (invalid_dates / len(df) * 100) if len(df) > 0 else 0,
            'date_columns': date_columns
        }
    
    def _check_negative_amounts(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for negative amounts"""
        amount_columns = [col for col in df.columns if 'amount' in col.lower() or 'price' in col.lower() or 'value' in col.lower()]
        negative_count = 0
        
        for col in amount_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                negative = (df[col] < 0).sum()
                negative_count += negative
        
        return {
            'count': int(negative_count),
            'percentage': (negative_count / len(df) * 100) if len(df) > 0 else 0
        }
    
    def _check_customer_ids(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for invalid customer IDs"""
        customer_columns = [col for col in df.columns if 'customer' in col.lower() or 'client' in col.lower() or 'user' in col.lower()]
        invalid_count = 0
        
        for col in customer_columns:
            # Check for nulls or empty strings
            invalid = df[col].isnull().sum()
            if df[col].dtype == 'object':
                invalid += (df[col].astype(str).str.strip() == '').sum()
            invalid_count += invalid
        
        return {
            'count': int(invalid_count),
            'percentage': (invalid_count / len(df) * 100) if len(df) > 0 else 0,
            'customer_columns': customer_columns
        }
    
    def _check_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for outliers using IQR method"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}
        total_outliers = 0
        
        for col in numeric_cols:
            if len(df[col].dropna()) > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                
                outliers_count = ((df[col] < lower) | (df[col] > upper)).sum()
                if outliers_count > 0:
                    outliers[col] = int(outliers_count)
                    total_outliers += outliers_count
        
        return {
            'total': int(total_outliers),
            'percentage': (total_outliers / len(df) * 100) if len(df) > 0 else 0,
            'by_column': outliers
        }
    
    def _check_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for wrong data types"""
        wrong_types = {}
        
        for col in df.columns:
            # Check if numeric column has strings
            if 'id' not in col.lower() and 'date' not in col.lower():
                if df[col].dtype == 'object':
                    # Check if it can be numeric
                    numeric_test = pd.to_numeric(df[col], errors='coerce')
                    if numeric_test.notna().sum() > len(df) * 0.5:
                        wrong_types[col] = 'should_be_numeric'
        
        return {
            'count': len(wrong_types),
            'columns': wrong_types
        }
    
    def _column_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistics for each column"""
        stats = {}
        
        for col in df.columns:
            col_stats = {
                'dtype': str(df[col].dtype),
                'nulls': int(df[col].isnull().sum()),
                'unique': int(df[col].nunique()),
                'null_percentage': (df[col].isnull().sum() / len(df) * 100) if len(df) > 0 else 0
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                col_stats.update({
                    'min': float(df[col].min()) if not df[col].isnull().all() else None,
                    'max': float(df[col].max()) if not df[col].isnull().all() else None,
                    'mean': float(df[col].mean()) if not df[col].isnull().all() else None,
                    'std': float(df[col].std()) if not df[col].isnull().all() else None
                })
            
            stats[col] = col_stats
        
        return stats
    
    def _calculate_quality_score(self) -> float:
        """Calculate overall quality score"""
        score = 100.0
        
        # Deduct for duplicates
        dup_pct = self.report['duplicates']['percentage']
        score -= dup_pct * 0.5
        
        # Deduct for missing values
        missing_pct = self.report['missing_values']['percentage']
        score -= missing_pct * 0.3
        
        # Deduct for invalid dates
        date_pct = self.report['invalid_dates']['percentage']
        score -= date_pct * 0.5
        
        # Deduct for negative amounts
        neg_pct = self.report['negative_amounts']['percentage']
        score -= neg_pct * 0.5
        
        # Deduct for outliers (but not too much)
        outlier_pct = self.report['outliers']['percentage']
        score -= outlier_pct * 0.1
        
        return max(0, min(100, score))
    
    def _generate_recommendations(self) -> List[str]:
        """Generate data quality recommendations"""
        recommendations = []
        
        if self.report['duplicates']['count'] > 0:
            recommendations.append(f"Remove {self.report['duplicates']['count']} duplicate rows")
        
        if self.report['missing_values']['total'] > 0:
            missing_cols = list(self.report['missing_values']['by_column'].keys())[:3]
            recommendations.append(f"Handle missing values in columns: {', '.join(missing_cols)}")
        
        if self.report['invalid_dates']['count'] > 0:
            recommendations.append(f"Fix {self.report['invalid_dates']['count']} invalid dates")
        
        if self.report['negative_amounts']['count'] > 0:
            recommendations.append(f"Review {self.report['negative_amounts']['count']} negative amounts")
        
        if self.report['outliers']['total'] > 0:
            recommendations.append(f"Investigate outliers in {len(self.report['outliers']['by_column'])} columns")
        
        if self.report['wrong_data_types']['count'] > 0:
            recommendations.append(f"Fix data types for: {', '.join(self.report['wrong_data_types']['columns'].keys())}")
        
        if not recommendations:
            recommendations.append("✅ Data quality looks good! No major issues detected.")
        
        return recommendations
    
    def print_report(self):
        """Print quality report to console"""
        report = self.report
        
        print("\n" + "=" * 70)
        print("📊 DATA QUALITY REPORT")
        print("=" * 70)
        
        print(f"\n📈 Summary:")
        print(f"   Total Rows:    {report['total_rows']:,}")
        print(f"   Total Columns: {report['total_columns']}")
        print(f"   Quality Score: {report['quality_score']:.1f}%")
        
        print(f"\n🔍 Issues Found:")
        print(f"   Duplicates:     {report['duplicates']['count']:,} ({report['duplicates']['percentage']:.2f}%)")
        print(f"   Missing Values: {report['missing_values']['total']:,} ({report['missing_values']['percentage']:.2f}%)")
        print(f"   Invalid Dates:  {report['invalid_dates']['count']:,} ({report['invalid_dates']['percentage']:.2f}%)")
        print(f"   Negative Amts:  {report['negative_amounts']['count']:,} ({report['negative_amounts']['percentage']:.2f}%)")
        print(f"   Outliers:       {report['outliers']['total']:,} ({report['outliers']['percentage']:.2f}%)")
        print(f"   Wrong Types:    {report['wrong_data_types']['count']}")
        
        if report['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in report['recommendations']:
                print(f"   - {rec}")
        
        print("=" * 70)