# src/core/validator.py
"""
التحقق من صحة البيانات
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Optional

class DataValidator:
    """التحقق من جودة البيانات"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_schema(self, df: pd.DataFrame, expected_columns: List[str]) -> bool:
        """
        التحقق من وجود الأعمدة المطلوبة
        
        Args:
            df: DataFrame للتحقق
            expected_columns: قائمة الأعمدة المطلوبة
            
        Returns:
            True إذا كانت الأعمدة موجودة، وإلا False
        """
        # التحقق من وجود الأعمدة
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            self.logger.warning(f"⚠️ الأعمدة المفقودة: {missing_columns}")
            self.logger.info(f"📋 الأعمدة الموجودة: {df.columns.tolist()}")
            
            # محاولة إيجاد أعمدة بديلة
            for col in missing_columns:
                if col == 'customer_id':
                    # البحث عن عمود يحتوي على معرف العميل
                    possible_cols = [c for c in df.columns if 'customer' in c.lower() or 'client' in c.lower() or 'id' in c.lower()]
                    if possible_cols:
                        df[col] = df[possible_cols[0]]
                        self.logger.info(f"✅ تم إنشاء عمود {col} من {possible_cols[0]}")
                
                elif col == 'purchase_date':
                    # البحث عن عمود يحتوي على تاريخ
                    possible_cols = [c for c in df.columns if 'date' in c.lower() or 'day' in c.lower() or 'purchase' in c.lower()]
                    if possible_cols:
                        df[col] = pd.to_datetime(df[possible_cols[0]])
                        self.logger.info(f"✅ تم إنشاء عمود {col} من {possible_cols[0]}")
                
                elif col == 'amount':
                    # البحث عن عمود يحتوي على مبلغ
                    possible_cols = [c for c in df.columns if 'amount' in c.lower() or 'price' in c.lower() or 'total' in c.lower() or 'value' in c.lower()]
                    if possible_cols:
                        df[col] = df[possible_cols[0]]
                        self.logger.info(f"✅ تم إنشاء عمود {col} من {possible_cols[0]}")
            
            # التحقق مرة أخرى
            missing_columns = [col for col in expected_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"لا يمكن العثور على الأعمدة المطلوبة: {missing_columns}")
        
        self.logger.info("✅ تم التحقق من صحة البيانات")
        return True
    
    def check_missing_values(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """
        فحص القيم المفقودة
        
        Args:
            df: DataFrame للفحص
            threshold: النسبة المئوية المسموح بها للقيم المفقودة
            
        Returns:
            DataFrame يحتوي على تقرير القيم المفقودة
        """
        missing_report = pd.DataFrame({
            'column': df.columns,
            'missing_count': df.isnull().sum().values,
            'missing_percentage': (df.isnull().sum() / len(df) * 100).values
        })
        
        # عرض الأعمدة التي تحتوي على قيم مفقودة
        columns_with_missing = missing_report[missing_report['missing_count'] > 0]
        if not columns_with_missing.empty:
            self.logger.warning("⚠️ الأعمدة التي تحتوي على قيم مفقودة:")
            for _, row in columns_with_missing.iterrows():
                self.logger.warning(f"   - {row['column']}: {row['missing_count']} ({row['missing_percentage']:.1f}%)")
        
        # التحقق من الأعمدة التي تتجاوز الحد المسموح
        high_missing = missing_report[missing_report['missing_percentage'] > threshold * 100]
        if not high_missing.empty:
            self.logger.error(f"❌ الأعمدة التي تتجاوز نسبة المفقود {threshold*100}%:")
            for _, row in high_missing.iterrows():
                self.logger.error(f"   - {row['column']}: {row['missing_percentage']:.1f}%")
            raise ValueError("نسبة القيم المفقودة عالية جداً")
        
        return missing_report
    
    def check_outliers(self, df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.DataFrame:
        """
        فحص القيم الشاذة
        
        Args:
            df: DataFrame للفحص
            column: اسم العمود للفحص
            method: طريقة الفحص ('iqr' أو 'zscore')
            
        Returns:
            DataFrame يحتوي على القيم الشاذة
        """
        if column not in df.columns:
            self.logger.warning(f"⚠️ العمود {column} غير موجود")
            return pd.DataFrame()
        
        # اختيار طريقة الفحص
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[column].dropna()))
            outliers = df[np.abs(stats.zscore(df[column])) > 3]
        
        else:
            raise ValueError(f"طريقة غير معروفة: {method}")
        
        if not outliers.empty:
            self.logger.warning(f"⚠️ تم العثور على {len(outliers)} قيمة شاذة في عمود {column}")
            self.logger.warning(f"   - النطاق الطبيعي: [{lower_bound:.2f}, {upper_bound:.2f}]")
            self.logger.warning(f"   - القيم الشاذة: {outliers[column].tolist()[:5]}...")
        else:
            self.logger.info(f"✅ لا توجد قيم شاذة في عمود {column}")
        
        return outliers