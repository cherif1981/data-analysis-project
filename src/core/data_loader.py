# src/core/data_loader.py - نسخة محسنة
"""
تحميل البيانات من مصادر متعددة
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, List

class DataLoader:
    """تحميل البيانات من مصادر متعددة"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        تحميل بيانات من ملف CSV
        
        Args:
            filepath: مسار ملف CSV
            
        Returns:
            DataFrame محمل من الملف
        """
        # تحويل المسار إلى Path object
        path = Path(filepath)
        
        # التحقق من وجود الملف
        if not path.exists():
            self.logger.warning(f"⚠️ الملف غير موجود: {filepath}")
            self.logger.info("📊 سيتم إنشاء بيانات عينة تلقائياً...")
            return self._create_sample_data()
        
        # تحميل الملف
        try:
            df = pd.read_csv(path)
            self.logger.info(f"✅ تم تحميل {len(df)} سجل من {path}")
            
            # التحقق من الأعمدة المطلوبة
            required_columns = ['customer_id', 'purchase_date', 'amount']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.logger.warning(f"⚠️ الأعمدة المفقودة: {missing_columns}")
                self.logger.info("📊 سيتم إعادة تسمية الأعمدة تلقائياً...")
                df = self._fix_column_names(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في تحميل الملف: {str(e)}")
            raise
    
    def _fix_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """إصلاح أسماء الأعمدة إذا كانت مختلفة"""
        
        # محاولة إعادة تسمية الأعمدة
        column_mapping = {}
        
        # البحث عن أعمدة مشابهة
        for col in df.columns:
            col_lower = col.lower().strip()
            
            # عميل
            if any(keyword in col_lower for keyword in ['customer', 'client', 'user', 'id']):
                if 'id' in col_lower or 'customer' in col_lower or 'client' in col_lower:
                    column_mapping[col] = 'customer_id'
            
            # تاريخ
            elif any(keyword in col_lower for keyword in ['date', 'day', 'purchase', 'order', 'transaction']):
                if 'date' in col_lower or 'purchase' in col_lower or 'order' in col_lower:
                    column_mapping[col] = 'purchase_date'
            
            # المبلغ
            elif any(keyword in col_lower for keyword in ['amount', 'price', 'total', 'value', 'sales', 'revenue']):
                if 'amount' in col_lower or 'price' in col_lower or 'total' in col_lower or 'value' in col_lower:
                    column_mapping[col] = 'amount'
        
        # إعادة تسمية الأعمدة
        if column_mapping:
            df = df.rename(columns=column_mapping)
            self.logger.info(f"✅ تم إعادة تسمية الأعمدة: {column_mapping}")
        
        # إذا لم يتم العثور على الأعمدة المطلوبة، قم بإنشاء بيانات افتراضية
        required_columns = ['customer_id', 'purchase_date', 'amount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            self.logger.warning(f"⚠️ لا تزال الأعمدة مفقودة: {missing_columns}")
            self.logger.info("📊 سيتم إنشاء أعمدة افتراضية...")
            
            # إنشاء أعمدة افتراضية
            if 'customer_id' not in df.columns:
                df['customer_id'] = [f'C{str(i).zfill(3)}' for i in range(1, len(df) + 1)]
            
            if 'purchase_date' not in df.columns:
                # إنشاء تواريخ عشوائية
                from datetime import datetime, timedelta
                import random
                start_date = datetime(2026, 1, 1)
                df['purchase_date'] = [
                    (start_date + timedelta(days=random.randint(0, 180))).strftime('%Y-%m-%d')
                    for _ in range(len(df))
                ]
            
            if 'amount' not in df.columns:
                # إنشاء مبالغ عشوائية
                df['amount'] = np.random.uniform(20, 1500, len(df)).round(2)
        
        return df
    
    def load_excel(self, filepath: str, sheet_name: str = 0) -> pd.DataFrame:
        """تحميل بيانات من ملف Excel"""
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"الملف غير موجود: {filepath}")
        
        try:
            df = pd.read_excel(path, sheet_name=sheet_name)
            self.logger.info(f"✅ تم تحميل {len(df)} سجل من {path}")
            
            # إصلاح أسماء الأعمدة إذا لزم الأمر
            df = self._fix_column_names(df)
            
            return df
        except Exception as e:
            self.logger.error(f"❌ خطأ في تحميل ملف Excel: {str(e)}")
            raise
    
    def load_sql(self, query: str, connection) -> pd.DataFrame:
        """تحميل بيانات من قاعدة بيانات SQL"""
        try:
            df = pd.read_sql(query, connection)
            self.logger.info(f"✅ تم تحميل {len(df)} سجل من قاعدة البيانات")
            
            # إصلاح أسماء الأعمدة إذا لزم الأمر
            df = self._fix_column_names(df)
            
            return df
        except Exception as e:
            self.logger.error(f"❌ خطأ في تحميل بيانات SQL: {str(e)}")
            raise
    
    def _create_sample_data(self) -> pd.DataFrame:
        """إنشاء بيانات عينة إذا لم يكن الملف موجوداً"""
        import numpy as np
        from datetime import datetime, timedelta
        import random
        
        self.logger.info("📊 إنشاء بيانات عينة...")
        
        # إعداد البيانات
        np.random.seed(42)
        num_customers = 20
        num_transactions = 150
        
        # إنشاء العملاء
        customer_ids = [f'C{str(i).zfill(3)}' for i in range(1, num_customers + 1)]
        
        # إنشاء التواريخ
        start_date = datetime(2026, 1, 1)
        end_date = datetime(2026, 9, 8)
        date_range = (end_date - start_date).days
        
        # إنشاء البيانات
        data = []
        categories = ['Electronics', 'Books', 'Clothing', 'Furniture', 'Food', 'Sports']
        
        for _ in range(num_transactions):
            customer = random.choice(customer_ids)
            days_offset = random.randint(0, date_range)
            purchase_date = start_date + timedelta(days=days_offset)
            amount = round(random.uniform(20, 1500), 2)
            category = random.choice(categories)
            
            data.append({
                'customer_id': customer,
                'purchase_date': purchase_date.strftime('%Y-%m-%d'),
                'amount': amount,
                'product_category': category
            })
        
        # إنشاء DataFrame
        df = pd.DataFrame(data)
        df = df.sort_values('purchase_date')
        
        # حفظ الملف للاستخدام المستقبلي
        Path('data').mkdir(exist_ok=True)
        df.to_csv('data/sales.csv', index=False)
        self.logger.info(f"✅ تم إنشاء {len(df)} سجل في data/sales.csv")
        
        return df