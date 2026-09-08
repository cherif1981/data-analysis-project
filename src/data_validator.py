# src/data_validator.py
"""
نظام متقدم للتحقق من جودة البيانات
يقوم بالتحقق من:
- هيكل البيانات (Schema)
- القيم المفقودة
- الصفوف المكررة
- التواريخ غير الصالحة
- القيم السالبة
- معرفات العملاء غير الصالحة
ويخرج تقريراً مفصلاً بجودة البيانات
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import Dict, Any, List, Tuple

class DataValidator:
    """فئة متخصصة للتحقق من جودة البيانات"""

    def __init__(self, df: pd.DataFrame, schema: Dict[str, str] = None):
        """
        تهيئة المدقق
        :param df: إطار البيانات المراد التحقق منه
        :param schema: قاموس يحدد نوع كل عمود متوقع (اختياري)
        """
        self.df = df
        self.schema = schema or {}
        self.report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': 0,
            'duplicates': 0,
            'invalid_dates': 0,
            'negative_values': 0,
            'invalid_customer_ids': 0,
            'schema_errors': [],
            'details': {}
        }
        self._run_all_checks()

    def _run_all_checks(self):
        """تنفيذ جميع خطوات التحقق بالتسلسل"""
        self._check_schema()
        self._check_missing_values()
        self._check_duplicates()
        self._check_invalid_dates()
        self._check_negative_values()
        self._check_invalid_customer_ids()
        self._calculate_quality_score()

    # ------------------- 1. التحقق من الهيكل (Schema) -------------------
    def _check_schema(self):
        """التحقق من تطابق الأعمدة مع الهيكل المتوقع"""
        if not self.schema:
            return
        errors = []
        for col, expected_type in self.schema.items():
            if col not in self.df.columns:
                errors.append(f"العمود '{col}' مفقود")
                continue
            actual_type = str(self.df[col].dtype)
            if expected_type not in actual_type:
                errors.append(f"العمود '{col}' نوعه {actual_type} ولكن المتوقع {expected_type}")
        self.report['schema_errors'] = errors

    # ------------------- 2. القيم المفقودة -------------------
    def _check_missing_values(self):
        """حساب عدد القيم المفقودة في جميع الأعمدة"""
        total_missing = int(self.df.isnull().sum().sum())
        self.report['missing_values'] = total_missing
        # تفاصيل إضافية
        missing_by_col = self.df.isnull().sum().to_dict()
        self.report['details']['missing_by_column'] = {
            k: int(v) for k, v in missing_by_col.items() if v > 0
        }

    # ------------------- 3. الصفوف المكررة -------------------
    def _check_duplicates(self):
        """حساب عدد الصفوف المكررة"""
        duplicates = int(self.df.duplicated().sum())
        self.report['duplicates'] = duplicates

    # ------------------- 4. التواريخ غير الصالحة -------------------
    def _check_invalid_dates(self):
        """التحقق من التواريخ غير الصالحة في الأعمدة التي تحتوي على 'date' أو 'day'"""
        date_cols = [col for col in self.df.columns if 'date' in col.lower() or 'day' in col.lower()]
        invalid_count = 0
        for col in date_cols:
            try:
                # محاولة تحويل العمود إلى تاريخ
                temp = pd.to_datetime(self.df[col], errors='coerce')
                invalid = temp.isnull().sum()
                invalid_count += invalid
            except:
                invalid_count += len(self.df)
        self.report['invalid_dates'] = int(invalid_count)
        self.report['details']['date_columns'] = date_cols

    # ------------------- 5. القيم السالبة -------------------
    def _check_negative_values(self):
        """التحقق من القيم السالبة في الأعمدة الرقمية التي تحتوي على 'amount' أو 'price' أو 'value'"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        target_cols = [col for col in numeric_cols if any(k in col.lower() for k in ['amount', 'price', 'value'])]
        negative_count = 0
        for col in target_cols:
            neg = (self.df[col] < 0).sum()
            negative_count += neg
        self.report['negative_values'] = int(negative_count)

    # ------------------- 6. معرفات العملاء غير الصالحة -------------------
    def _check_invalid_customer_ids(self):
        """التحقق من معرفات العملاء (فارغة أو مكررة أو غير صالحة)"""
        customer_cols = [col for col in self.df.columns if 'customer' in col.lower() or 'client' in col.lower() or 'user' in col.lower()]
        invalid_count = 0
        for col in customer_cols:
            # القيم الفارغة
            invalid = self.df[col].isnull().sum()
            # القيم النصية الفارغة
            if self.df[col].dtype == 'object':
                invalid += (self.df[col].astype(str).str.strip() == '').sum()
            invalid_count += invalid
        self.report['invalid_customer_ids'] = int(invalid_count)
        self.report['details']['customer_columns'] = customer_cols

    # ------------------- 7. حساب درجة الجودة -------------------
    def _calculate_quality_score(self):
        """حساب درجة الجودة الإجمالية بناءً على عدد المشكلات"""
        total_issues = (
            self.report['missing_values'] +
            self.report['duplicates'] +
            self.report['invalid_dates'] +
            self.report['negative_values'] +
            self.report['invalid_customer_ids']
        )
        # كل مشكلة تؤثر على درجة الجودة بنسبة معينة
        max_possible_issues = self.report['total_rows'] * 5  # تقدير
        penalty = min(100, (total_issues / max(1, self.report['total_rows'])) * 100)
        score = max(0, 100 - penalty * 0.5)
        self.report['quality_score'] = round(score, 1)

    # ------------------- طباعة التقرير -------------------
    def print_report(self):
        """طباعة تقرير جودة البيانات بشكل منسق"""
        r = self.report
        print("\n" + "=" * 60)
        print("📊  DATA QUALITY REPORT".center(60))
        print("=" * 60)
        print(f"\n  📌 Rows              : {r['total_rows']:,}")
        print(f"  📌 Columns           : {r['total_columns']}")
        print(f"  📌 Missing values    : {r['missing_values']:,}")
        print(f"  📌 Duplicates        : {r['duplicates']:,}")
        print(f"  📌 Invalid dates     : {r['invalid_dates']:,}")
        print(f"  📌 Negative values   : {r['negative_values']:,}")
        print(f"  📌 Invalid Customer IDs: {r['invalid_customer_ids']:,}")

        # الأخطاء في الهيكل إن وجدت
        if r['schema_errors']:
            print("\n  ⚠️  Schema Errors:")
            for err in r['schema_errors']:
                print(f"     - {err}")

        print("\n" + "-" * 60)
        print(f"  🎯 Quality Score    : {r['quality_score']}%")
        status = "✅ PASS" if r['quality_score'] >= 80 else "⚠️  REVIEW"
        print(f"  📈 Status            : {status}")
        print("=" * 60 + "\n")


# ------------------- مثال على الاستخدام -------------------
if __name__ == "__main__":
    # تحميل بيانات من ملف CSV (يمكنك تغيير المسار)
    # df = pd.read_csv("data/sample_data.csv")

    # --- بيانات وهمية للاختبار ---
    data = {
        'customer_id': ['C001', 'C002', 'C003', '', 'C005', 'C006', 'C007', 'C008', 'C009', 'C010'],
        'transaction_date': ['2025-01-01', '2025-02-15', 'invalid_date', '2025-03-20', '2025-04-10', '2025-05-05', '2025-06-12', '2025-07-18', '2025-08-22', '2025-09-30'],
        'amount': [100.5, 200.0, -50.0, 150.0, 300.0, 400.0, -20.0, 250.0, 100.0, 50.0],
        'product': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A']
    }
    df = pd.DataFrame(data)

    # إضافة بعض الصفوف المكررة
    df = pd.concat([df, df.iloc[[0, 2]]], ignore_index=True)

    # تعريف الهيكل المتوقع (اختياري)
    schema = {
        'customer_id': 'object',
        'transaction_date': 'datetime64',
        'amount': 'float',
        'product': 'object'
    }

    # إنشاء المدقق وتشغيله
    validator = DataValidator(df, schema=schema)
    validator.print_report()
