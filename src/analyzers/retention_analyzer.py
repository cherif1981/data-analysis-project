# src/analyzers/retention_analyzer.py
"""
تحليل الاحتفاظ بالعملاء (Cohort Analysis)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RetentionAnalyzer:
    """تحليل الاحتفاظ بالعملاء باستخدام Cohort Analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def cohort_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        تحليل المجموعات الزمنية (Cohort Analysis)
        
        Args:
            df: DataFrame يحتوي على بيانات المعاملات
            
        Returns:
            قاموس يحتوي على نتائج تحليل الاحتفاظ
        """
        try:
            self.logger.info("🔄 بدء تحليل الاحتفاظ بالعملاء...")
            
            # التحقق من البيانات
            if df.empty:
                raise ValueError("البيانات فارغة")
            
            # تحديد الأعمدة المطلوبة
            date_col = self._find_date_column(df)
            customer_col = self._find_customer_column(df)
            
            if date_col is None:
                self.logger.warning("⚠️ لا يوجد عمود تاريخ، استخدام عمود افتراضي")
                # إنشاء عمود تاريخ افتراضي
                df['purchase_date'] = pd.Timestamp.now()
                date_col = 'purchase_date'
            
            if customer_col is None:
                self.logger.warning("⚠️ لا يوجد عمود عميل، استخدام عمود افتراضي")
                # إنشاء عمود عميل افتراضي
                df['customer_id'] = [f'C{str(i).zfill(3)}' for i in range(len(df))]
                customer_col = 'customer_id'
            
            self.logger.info(f"📋 عمود التاريخ: {date_col}, عمود العميل: {customer_col}")
            
            # نسخ البيانات
            df_copy = df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
            
            # ✅ الخطوة 1: تحديد شهر الشراء الأول لكل عميل (Cohort)
            df_copy['first_purchase'] = df_copy.groupby(customer_col)[date_col].transform('min')
            
            # ✅ الخطوة 2: إنشاء عمود cohort_month
            df_copy['cohort_month'] = df_copy['first_purchase'].dt.to_period('M')
            
            # ✅ الخطوة 3: إنشاء عمود purchase_month
            df_copy['purchase_month'] = df_copy[date_col].dt.to_period('M')
            
            # ✅ الخطوة 4: حساب رقم الفترة
            df_copy['period'] = (df_copy['purchase_month'].dt.year - df_copy['cohort_month'].dt.year) * 12 + \
                               (df_copy['purchase_month'].dt.month - df_copy['cohort_month'].dt.month)
            
            # ✅ الخطوة 5: حساب عدد العملاء لكل مجموعة وفترة
            cohort_data = df_copy.groupby(['cohort_month', 'period']).agg({
                customer_col: 'nunique'
            }).reset_index()
            
            cohort_data.columns = ['cohort', 'period', 'customers']
            
            # ✅ الخطوة 6: إنشاء مصفوفة الاحتفاظ
            retention_matrix = cohort_data.pivot_table(
                index='cohort',
                columns='period',
                values='customers',
                fill_value=0
            )
            
            # ✅ الخطوة 7: حساب نسب الاحتفاظ
            retention_rates = pd.DataFrame()
            if 0 in retention_matrix.columns:
                retention_rates = retention_matrix.div(retention_matrix[0], axis=0) * 100
            else:
                # إذا لم يكن هناك عمود 0، استخدم أول عمود
                first_col = retention_matrix.columns[0] if not retention_matrix.empty else 0
                retention_rates = retention_matrix.div(retention_matrix[first_col], axis=0) * 100
            
            # ✅ الخطوة 8: إحصائيات إضافية
            total_customers = df_copy[customer_col].nunique()
            avg_cohort_size = retention_matrix[0].mean() if 0 in retention_matrix.columns else 0
            
            # ✅ الخطوة 9: حساب متوسط الاحتفاظ لكل فترة
            avg_retention = {}
            for period in retention_rates.columns:
                if period != 0:
                    avg_retention[str(period)] = float(retention_rates[period].mean())
            
            # ✅ الخطوة 10: إنشاء النتائج
            result = {
                'cohort_data': cohort_data,
                'retention_matrix': retention_matrix,
                'retention_rates': retention_rates,
                'summary': {
                    'total_customers': int(total_customers),
                    'avg_cohort_size': float(avg_cohort_size),
                    'num_cohorts': len(retention_matrix),
                    'max_period': int(retention_matrix.columns.max()) if not retention_matrix.empty else 0,
                    'avg_retention': avg_retention
                },
                'customer_data': df_copy[[customer_col, date_col, 'first_purchase', 'cohort_month', 'purchase_month', 'period']].head(100)
            }
            
            self.logger.info(f"✅ تم تحليل {len(retention_matrix)} مجموعة زمنية")
            self.logger.info(f"📊 متوسط حجم المجموعة: {avg_cohort_size:.1f} عميل")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في تحليل الاحتفاظ: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._empty_result(str(e))
    
    def _find_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """البحث عن عمود التاريخ"""
        date_keywords = ['date', 'day', 'purchase', 'order', 'transaction', 'time', 'created', 'updated']
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in date_keywords):
                return col
        return None
    
    def _find_customer_column(self, df: pd.DataFrame) -> Optional[str]:
        """البحث عن عمود معرف العميل"""
        customer_keywords = ['customer', 'client', 'user', 'id', 'cust', 'account', 'member']
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in customer_keywords):
                return col
        return None
    
    def _empty_result(self, error_msg: str = "") -> Dict[str, Any]:
        """إرجاع نتائج فارغة عند حدوث خطأ"""
        return {
            'cohort_data': pd.DataFrame(),
            'retention_matrix': pd.DataFrame(),
            'retention_rates': pd.DataFrame(),
            'summary': {
                'total_customers': 0,
                'avg_cohort_size': 0,
                'num_cohorts': 0,
                'max_period': 0,
                'avg_retention': {}
            },
            'customer_data': pd.DataFrame(),
            'error': error_msg,
            'note': 'حدث خطأ في تحليل الاحتفاظ'
        }