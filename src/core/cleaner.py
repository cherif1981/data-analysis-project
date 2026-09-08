# src/core/cleaner.py
"""
تنظيف البيانات
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

class DataCleaner:
    """تنظيف البيانات وإزالة الأخطاء"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def handle_missing(self, df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
        """
        معالجة القيم المفقودة
        
        Args:
            df: DataFrame للمعالجة
            strategy: استراتيجية المعالجة ('drop', 'fill_mean', 'fill_median', 'fill_mode')
            
        Returns:
            DataFrame بعد المعالجة
        """
        # حساب عدد القيم المفقودة قبل المعالجة
        missing_before = df.isnull().sum().sum()
        
        if missing_before == 0:
            self.logger.info("✅ لا توجد قيم مفقودة")
            return df
        
        self.logger.info(f"🔍 عدد القيم المفقودة: {missing_before}")
        
        # تطبيق الاستراتيجية
        if strategy == 'drop':
            df = df.dropna()
            self.logger.info(f"✅ تم حذف الصفوف التي تحتوي على قيم مفقودة")
        
        elif strategy == 'fill_mean':
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].fillna(df[col].mean())
            self.logger.info(f"✅ تم ملء القيم المفقودة بالمتوسط")
        
        elif strategy == 'fill_median':
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].fillna(df[col].median())
            self.logger.info(f"✅ تم ملء القيم المفقودة بالوسيط")
        
        elif strategy == 'fill_mode':
            for col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
            self.logger.info(f"✅ تم ملء القيم المفقودة بالمنوال")
        
        else:
            raise ValueError(f"استراتيجية غير معروفة: {strategy}")
        
        # حساب عدد القيم المفقودة بعد المعالجة
        missing_after = df.isnull().sum().sum()
        self.logger.info(f"✅ تم معالجة {missing_before - missing_after} قيمة مفقودة")
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        إزالة التكرارات
        
        Args:
            df: DataFrame للمعالجة
            
        Returns:
            DataFrame بعد إزالة التكرارات
        """
        duplicates_before = len(df)
        df = df.drop_duplicates()
        duplicates_removed = duplicates_before - len(df)
        
        if duplicates_removed > 0:
            self.logger.info(f"✅ تم إزالة {duplicates_removed} صف مكرر")
        else:
            self.logger.info("✅ لا توجد تكرارات")
        
        return df
    
    def standardize_dates(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        توحيد التنسيقات الزمنية
        
        Args:
            df: DataFrame للمعالجة
            column: اسم عمود التاريخ
            
        Returns:
            DataFrame بعد توحيد التواريخ
        """
        if column not in df.columns:
            self.logger.warning(f"⚠️ العمود {column} غير موجود")
            return df
        
        try:
            # تحويل إلى datetime
            df[column] = pd.to_datetime(df[column], errors='coerce')
            
            # التحقق من القيم المفقودة بعد التحويل
            missing_dates = df[column].isnull().sum()
            if missing_dates > 0:
                self.logger.warning(f"⚠️ {missing_dates} تاريخ غير صالح")
                # حذف التواريخ غير الصالحة
                df = df.dropna(subset=[column])
            
            self.logger.info(f"✅ تم توحيد التنسيق الزمني للعمود {column}")
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في توحيد التواريخ: {str(e)}")
        
        return df
    
    def clean_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        تنظيف الأعمدة النصية
        
        Args:
            df: DataFrame للمعالجة
            
        Returns:
            DataFrame بعد تنظيف النصوص
        """
        for col in df.select_dtypes(include=['object']).columns:
            # إزالة المسافات الزائدة
            df[col] = df[col].str.strip()
            
            # توحيد حالة الأحرف
            df[col] = df[col].str.lower()
            
            # إزالة الأحرف الخاصة
            df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)
        
        self.logger.info("✅ تم تنظيف الأعمدة النصية")
        return df